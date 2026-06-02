"""OFAC name matcher using a cascading exact → fuzzy → FAISS composite scorer.

Each detected entity is matched against its category-specific OFAC name list
(SANCTIONED_PERSON searches only individual rows; SANCTIONED_ORG searches only
org/vessel/aircraft rows).  The four-stage cascade is:

1. Exact match (case-insensitive) → composite = 1.0
2. rapidfuzz token_sort_ratio >= FUZZY_THRESHOLD → proceed to composite scoring
3. sentence-transformers embedding + FAISS L1 nearest-neighbour search
4. Composite = fuzzy_weight * fuzzy_norm + semantic_weight * semantic_norm
              + (1 - fuzzy_weight - semantic_weight) * exact_substring_bonus

Routing:
    composite >= CONFIRMED_THRESHOLD (0.90) → CONFIRMED
    REVIEW_THRESHOLD (0.65) <= composite < 0.90 → REVIEW
    composite < REVIEW_THRESHOLD → NO_MATCH
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

from ner_ofac import config
from ner_ofac.analysis.faiss_index import search_index

logger = logging.getLogger(__name__)

# Weight for the exact-substring bonus (remainder after fuzzy + semantic).
_EXACT_BONUS_WEIGHT = 1.0 - config.FUZZY_WEIGHT - config.SEMANTIC_WEIGHT


@dataclass
class MatchResult:
    """Result of matching a single detected entity against the OFAC SDN list.

    Attributes:
        entity_text: Surface text of the detected entity.
        entity_type: Comprehend entity type (e.g. ``"SANCTIONED_PERSON"``).
        matched_name: Best-matching OFAC name, or ``None`` if no match found.
        composite_score: Final composite similarity score in ``[0.0, 1.0]``.
        stage: Matching stage that produced the result (``"exact"``,
            ``"fuzzy"``, ``"semantic"``, or ``"none"``).
        routing: One of ``"CONFIRMED"``, ``"REVIEW"``, or ``"NO_MATCH"``.
    """

    entity_text: str
    entity_type: str
    matched_name: Optional[str]
    composite_score: float
    stage: str
    routing: str


def _route(composite_score: float) -> str:
    """Map a composite score to a routing label.

    Args:
        composite_score: Score in ``[0.0, 1.0]``.

    Returns:
        ``"CONFIRMED"``, ``"REVIEW"``, or ``"NO_MATCH"``.
    """
    if composite_score >= config.CONFIRMED_THRESHOLD:
        return "CONFIRMED"
    if composite_score >= config.REVIEW_THRESHOLD:
        return "REVIEW"
    return "NO_MATCH"


def _exact_substring_bonus(entity_text: str, ofac_name: str) -> float:
    """Return 1.0 if entity_text is a case-insensitive substring of ofac_name, else 0.0.

    Args:
        entity_text: The detected entity surface string.
        ofac_name: An OFAC SDN name to compare against.

    Returns:
        ``1.0`` if *entity_text* appears within *ofac_name* (case-insensitive),
        ``0.0`` otherwise.
    """
    return 1.0 if entity_text.lower() in ofac_name.lower() else 0.0


def _compute_composite(
    entity_text: str,
    ofac_name: str,
    fuzzy_score_0_100: float,
    semantic_distance: float,
    max_distance: float,
) -> float:
    """Compute the composite match score from fuzzy, semantic, and substring signals.

    Args:
        entity_text: The detected entity surface string.
        ofac_name: The OFAC name being scored.
        fuzzy_score_0_100: rapidfuzz token_sort_ratio score in ``[0, 100]``.
        semantic_distance: L1 distance returned by FAISS (lower = more similar).
        max_distance: The maximum L1 distance seen in the current search batch,
            used to normalise ``semantic_distance`` to ``[0, 1]``.

    Returns:
        Composite score in ``[0.0, 1.0]``.
    """
    fuzzy_norm = fuzzy_score_0_100 / 100.0
    # Convert distance to similarity: 0 distance → 1.0 similarity.
    if max_distance > 0:
        semantic_norm = 1.0 - (semantic_distance / max_distance)
    else:
        semantic_norm = 1.0

    bonus = _exact_substring_bonus(entity_text, ofac_name)

    composite = (
        config.FUZZY_WEIGHT * fuzzy_norm
        + config.SEMANTIC_WEIGHT * semantic_norm
        + _EXACT_BONUS_WEIGHT * bonus
    )
    return min(composite, 1.0)


def match_entity(
    entity_text: str,
    entity_type: str,
    ofac_names: List[str],
    faiss_index,
    name_list: List[str],
    model: SentenceTransformer,
) -> MatchResult:
    """Match a single detected entity against its category-specific OFAC name list.

    Runs the four-stage cascade and returns the best ``MatchResult``.

    Args:
        entity_text: Surface text of the detected entity (e.g. ``"JOHN DOE"``).
        entity_type: Entity type label used for logging/routing
            (e.g. ``"SANCTIONED_PERSON"``).
        ofac_names: The complete list of OFAC names for this entity category.
            Must match the names used to build *faiss_index*.
        faiss_index: A ``faiss.IndexFlatL1`` index built from *name_list*.
        name_list: The ordered list of names aligned with *faiss_index*.
        model: A loaded ``SentenceTransformer`` used to embed the query.

    Returns:
        A ``MatchResult`` with the best match found (or ``NO_MATCH`` if no
        candidate surpasses ``REVIEW_THRESHOLD``).
    """
    if not entity_text or not ofac_names:
        return MatchResult(
            entity_text=entity_text,
            entity_type=entity_type,
            matched_name=None,
            composite_score=0.0,
            stage="none",
            routing="NO_MATCH",
        )

    entity_lower = entity_text.lower()

    # ------------------------------------------------------------------
    # Stage 1: Exact match (case-insensitive)
    # ------------------------------------------------------------------
    for name in ofac_names:
        if entity_lower == name.lower():
            logger.debug("Exact match for '%s' → '%s'", entity_text, name)
            return MatchResult(
                entity_text=entity_text,
                entity_type=entity_type,
                matched_name=name,
                composite_score=1.0,
                stage="exact",
                routing="CONFIRMED",
            )

    # ------------------------------------------------------------------
    # Stage 2: Fuzzy candidates using rapidfuzz token_sort_ratio
    # ------------------------------------------------------------------
    fuzzy_candidates: List[Tuple[str, float]] = []
    for name in ofac_names:
        score = fuzz.token_sort_ratio(entity_text, name)
        if score >= config.FUZZY_THRESHOLD:
            fuzzy_candidates.append((name, float(score)))

    # ------------------------------------------------------------------
    # Stage 3: FAISS semantic search
    # ------------------------------------------------------------------
    query_embedding = model.encode([entity_text], convert_to_numpy=True)[0].astype(np.float32)
    k = min(5, len(name_list))
    semantic_results = search_index(faiss_index, name_list, query_embedding, k=k)

    # ------------------------------------------------------------------
    # Stage 4: Composite scoring on union of fuzzy + semantic candidates
    # ------------------------------------------------------------------
    # Collect all candidates for composite scoring.
    candidate_names = set(n for n, _ in fuzzy_candidates) | set(n for n, _ in semantic_results)

    if not candidate_names:
        return MatchResult(
            entity_text=entity_text,
            entity_type=entity_type,
            matched_name=None,
            composite_score=0.0,
            stage="none",
            routing="NO_MATCH",
        )

    # Build lookup maps.
    fuzzy_map = {n: s for n, s in fuzzy_candidates}
    semantic_map = {n: d for n, d in semantic_results}

    # Determine max distance for normalisation.
    all_distances = [d for _, d in semantic_results]
    max_distance = max(all_distances) if all_distances else 1.0

    best_score = 0.0
    best_name = None
    best_stage = "none"

    for name in candidate_names:
        f_score = fuzzy_map.get(name, fuzz.token_sort_ratio(entity_text, name))
        s_dist = semantic_map.get(name, max_distance)
        composite = _compute_composite(entity_text, name, f_score, s_dist, max_distance)
        if composite > best_score:
            best_score = composite
            best_name = name
            best_stage = "fuzzy" if name in fuzzy_map else "semantic"

    routing = _route(best_score)
    logger.debug(
        "match_entity '%s' → '%s' composite=%.3f routing=%s stage=%s",
        entity_text,
        best_name,
        best_score,
        routing,
        best_stage,
    )

    return MatchResult(
        entity_text=entity_text,
        entity_type=entity_type,
        matched_name=best_name,
        composite_score=best_score,
        stage=best_stage,
        routing=routing,
    )
