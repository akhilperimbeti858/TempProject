"""Context expander for REVIEW-tier entity matches.

For entities routed as REVIEW (composite score between REVIEW_THRESHOLD and
CONFIRMED_THRESHOLD), this module expands the entity's context window by
±CONTEXT_WINDOW_WORDS words and re-runs the fuzzy + semantic scoring on the
expanded text.  This can push a borderline match into the CONFIRMED tier.

The expanded text is used only for re-scoring; the original entity text is
preserved in the returned MatchResult.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from ner_ofac import config
from ner_ofac.analysis.faiss_index import search_index
from ner_ofac.analysis.ofac_matcher import MatchResult, _compute_composite, _route

logger = logging.getLogger(__name__)


def expand_context(
    sentence: str,
    begin_offset: int,
    end_offset: int,
    window: int = config.CONTEXT_WINDOW_WORDS,
) -> str:
    """Extract an expanded text window around an entity span.

    Splits *sentence* on whitespace, locates the word(s) that overlap the
    character span ``[begin_offset, end_offset)``, then returns the surrounding
    ``±window`` words joined by spaces.

    Args:
        sentence: The full sentence string containing the entity.
        begin_offset: Character index of the first character of the entity span.
        end_offset: Character index one past the last character of the entity span.
        window: Number of words to include on each side of the entity span.
            Defaults to ``config.CONTEXT_WINDOW_WORDS``.

    Returns:
        The expanded context string.  If the span cannot be located in the
        token list, the original *sentence* is returned unchanged.

    Example::

        >>> sentence = "The ACME Corp entity was sanctioned last year."
        >>> expand_context(sentence, 4, 13, window=2)
        'The ACME Corp entity was'
    """
    tokens = sentence.split()
    if not tokens:
        return sentence

    # Build character-offset spans for each token.
    token_spans: List[tuple] = []
    pos = 0
    for tok in tokens:
        start = sentence.find(tok, pos)
        end = start + len(tok)
        token_spans.append((start, end))
        pos = end

    # Find which token indices overlap with the entity span.
    overlap_indices = [
        i
        for i, (ts, te) in enumerate(token_spans)
        if ts < end_offset and te > begin_offset
    ]

    if not overlap_indices:
        logger.debug(
            "expand_context: entity span [%d,%d) not found in tokens; returning full sentence",
            begin_offset,
            end_offset,
        )
        return sentence

    first_tok = overlap_indices[0]
    last_tok = overlap_indices[-1]

    left = max(0, first_tok - window)
    right = min(len(tokens) - 1, last_tok + window)

    expanded = " ".join(tokens[left : right + 1])
    logger.debug(
        "expand_context: span [%d,%d) → tokens [%d,%d] → '%s'",
        begin_offset,
        end_offset,
        left,
        right,
        expanded,
    )
    return expanded


def recheck_review(
    entity: MatchResult,
    sentence: str,
    begin_offset: int,
    end_offset: int,
    ofac_names: List[str],
    faiss_index,
    name_list: List[str],
    model: SentenceTransformer,
) -> MatchResult:
    """Re-score a REVIEW-tier match using the expanded context window.

    Expands the context around the entity span and re-runs fuzzy + semantic
    scoring against the previously matched OFAC name.  If the new composite
    score is higher than the original, it is used to update routing.

    Args:
        entity: The original ``MatchResult`` with routing == ``"REVIEW"``.
        sentence: The full sentence text in which the entity was found.
        begin_offset: Character index of the start of the entity span.
        end_offset: Character index one past the end of the entity span.
        ofac_names: Category-specific OFAC name list.
        faiss_index: FAISS index built from *name_list*.
        name_list: Ordered name list aligned with *faiss_index*.
        model: Loaded ``SentenceTransformer`` for query encoding.

    Returns:
        An updated ``MatchResult``.  If the expanded context yields a higher
        composite score the new score and routing are applied; otherwise the
        original ``MatchResult`` is returned unchanged.
    """
    expanded_text = expand_context(sentence, begin_offset, end_offset)
    logger.debug("recheck_review: expanded text = '%s'", expanded_text)

    if not entity.matched_name:
        return entity

    # Re-run fuzzy scoring with the expanded text against the best OFAC name.
    from rapidfuzz import fuzz as _fuzz

    f_score = _fuzz.token_sort_ratio(expanded_text, entity.matched_name)

    # Re-run semantic scoring.
    query_embedding = model.encode([expanded_text], convert_to_numpy=True)[0].astype(np.float32)
    k = min(5, len(name_list))
    semantic_results = search_index(faiss_index, name_list, query_embedding, k=k)
    semantic_map = {n: d for n, d in semantic_results}

    all_distances = [d for _, d in semantic_results]
    max_distance = max(all_distances) if all_distances else 1.0

    s_dist = semantic_map.get(entity.matched_name, max_distance)
    new_composite = _compute_composite(
        expanded_text, entity.matched_name, f_score, s_dist, max_distance
    )

    if new_composite > entity.composite_score:
        new_routing = _route(new_composite)
        logger.debug(
            "recheck_review: '%s' composite improved %.3f → %.3f routing=%s",
            entity.entity_text,
            entity.composite_score,
            new_composite,
            new_routing,
        )
        return MatchResult(
            entity_text=entity.entity_text,
            entity_type=entity.entity_type,
            matched_name=entity.matched_name,
            composite_score=new_composite,
            stage="context_expanded",
            routing=new_routing,
        )

    logger.debug(
        "recheck_review: '%s' composite unchanged at %.3f",
        entity.entity_text,
        entity.composite_score,
    )
    return entity
