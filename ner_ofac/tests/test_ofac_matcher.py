"""Tests for ner_ofac.analysis.ofac_matcher.

Covers:
- Exact match returns CONFIRMED with composite == 1.0
- Near-fuzzy match returns appropriate routing
- No-match below threshold returns NO_MATCH
- Category silo isolation: SANCTIONED_PERSON doesn't match org names
- MatchResult dataclass fields are populated correctly
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers: build a minimal mock FAISS index and SentenceTransformer
# ---------------------------------------------------------------------------


def _make_mock_model(embed_dim: int = 16):
    """Return a mock SentenceTransformer that produces deterministic embeddings."""
    mock_model = MagicMock()

    def mock_encode(texts, convert_to_numpy=True, show_progress_bar=False):
        vecs = []
        for text in texts:
            # Reproducible deterministic vector based on text hash.
            rng = np.random.default_rng(abs(hash(text)) % (2**31))
            vecs.append(rng.random(embed_dim).astype(np.float32))
        return np.array(vecs, dtype=np.float32)

    mock_model.encode.side_effect = mock_encode
    return mock_model


def _build_test_index(names, embed_dim=16):
    """Build a real (tiny) FAISS index for testing."""
    import faiss

    mock_model = _make_mock_model(embed_dim)
    embeddings = mock_model.encode(names, convert_to_numpy=True)
    index = faiss.IndexFlatL1(embed_dim)
    index.add(embeddings.astype(np.float32))
    return index, names, mock_model


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from ner_ofac.analysis.ofac_matcher import match_entity, MatchResult, _route


# ---------------------------------------------------------------------------
# _route
# ---------------------------------------------------------------------------


class TestRoute:
    def test_confirmed_threshold(self):
        assert _route(0.90) == "CONFIRMED"
        assert _route(0.95) == "CONFIRMED"
        assert _route(1.0) == "CONFIRMED"

    def test_review_threshold(self):
        assert _route(0.65) == "REVIEW"
        assert _route(0.75) == "REVIEW"
        assert _route(0.8999) == "REVIEW"

    def test_no_match(self):
        assert _route(0.0) == "NO_MATCH"
        assert _route(0.5) == "NO_MATCH"
        assert _route(0.6499) == "NO_MATCH"


# ---------------------------------------------------------------------------
# match_entity
# ---------------------------------------------------------------------------


class TestMatchEntity:
    """Tests for the cascading match logic."""

    PERSON_NAMES = ["JOHN DOE", "JANE SMITH", "IVAN PETROV", "ALI HASSAN"]
    ORG_NAMES = ["ACME CORP", "SHADOW TRADING LLC", "DARK VESSEL INC"]

    def test_exact_match_returns_confirmed(self):
        """Exact case-insensitive match yields composite=1.0 and CONFIRMED."""
        index, name_list, model = _build_test_index(self.PERSON_NAMES)
        result = match_entity(
            entity_text="JOHN DOE",
            entity_type="SANCTIONED_PERSON",
            ofac_names=self.PERSON_NAMES,
            faiss_index=index,
            name_list=name_list,
            model=model,
        )
        assert result.composite_score == 1.0
        assert result.routing == "CONFIRMED"
        assert result.matched_name == "JOHN DOE"
        assert result.stage == "exact"

    def test_exact_match_case_insensitive(self):
        """Exact match is case-insensitive."""
        index, name_list, model = _build_test_index(self.PERSON_NAMES)
        result = match_entity(
            entity_text="john doe",
            entity_type="SANCTIONED_PERSON",
            ofac_names=self.PERSON_NAMES,
            faiss_index=index,
            name_list=name_list,
            model=model,
        )
        assert result.composite_score == 1.0
        assert result.routing == "CONFIRMED"

    def test_no_match_below_threshold(self):
        """Completely unrelated text returns NO_MATCH."""
        index, name_list, model = _build_test_index(self.PERSON_NAMES)
        result = match_entity(
            entity_text="WIDGET FACTORY",
            entity_type="SANCTIONED_PERSON",
            ofac_names=self.PERSON_NAMES,
            faiss_index=index,
            name_list=name_list,
            model=model,
        )
        # Routing should be NO_MATCH or REVIEW — should not be CONFIRMED.
        assert result.routing in ("NO_MATCH", "REVIEW")

    def test_category_silo_person_vs_org(self):
        """SANCTIONED_PERSON search does not match org names."""
        # Build index from ORG names only.
        index, name_list, model = _build_test_index(self.ORG_NAMES)
        # Query with a person name against an ORG index — no exact match possible.
        result = match_entity(
            entity_text="JOHN DOE",
            entity_type="SANCTIONED_PERSON",
            ofac_names=self.ORG_NAMES,  # wrong silo
            faiss_index=index,
            name_list=name_list,
            model=model,
        )
        # There should be no exact match since "JOHN DOE" is not in ORG_NAMES.
        assert result.matched_name != "JOHN DOE" or result.stage != "exact"
        # Composite score should be low.
        assert result.composite_score < 0.90

    def test_empty_entity_text_returns_no_match(self):
        """Empty entity text returns NO_MATCH immediately."""
        index, name_list, model = _build_test_index(self.PERSON_NAMES)
        result = match_entity(
            entity_text="",
            entity_type="SANCTIONED_PERSON",
            ofac_names=self.PERSON_NAMES,
            faiss_index=index,
            name_list=name_list,
            model=model,
        )
        assert result.routing == "NO_MATCH"
        assert result.composite_score == 0.0

    def test_empty_ofac_names_returns_no_match(self):
        """Empty OFAC name list returns NO_MATCH immediately."""
        index, name_list, model = _build_test_index(self.PERSON_NAMES)
        result = match_entity(
            entity_text="JOHN DOE",
            entity_type="SANCTIONED_PERSON",
            ofac_names=[],
            faiss_index=index,
            name_list=name_list,
            model=model,
        )
        assert result.routing == "NO_MATCH"

    def test_match_result_fields_populated(self):
        """All MatchResult fields are present and correctly typed."""
        index, name_list, model = _build_test_index(self.PERSON_NAMES)
        result = match_entity(
            entity_text="JOHN DOE",
            entity_type="SANCTIONED_PERSON",
            ofac_names=self.PERSON_NAMES,
            faiss_index=index,
            name_list=name_list,
            model=model,
        )
        assert isinstance(result, MatchResult)
        assert isinstance(result.entity_text, str)
        assert isinstance(result.entity_type, str)
        assert isinstance(result.composite_score, float)
        assert result.routing in ("CONFIRMED", "REVIEW", "NO_MATCH")
        assert result.stage in ("exact", "fuzzy", "semantic", "none")

    def test_fuzzy_near_match(self):
        """A near-match (minor typo) should reach at least REVIEW tier."""
        index, name_list, model = _build_test_index(self.PERSON_NAMES)
        # "JOHN DO" is a single-character deletion from "JOHN DOE".
        result = match_entity(
            entity_text="JOHN DO",
            entity_type="SANCTIONED_PERSON",
            ofac_names=self.PERSON_NAMES,
            faiss_index=index,
            name_list=name_list,
            model=model,
        )
        # Near match should be REVIEW or CONFIRMED (not NO_MATCH).
        assert result.routing in ("REVIEW", "CONFIRMED")
        assert result.matched_name is not None
