"""Tests for ner_ofac.training.feedback_loop.

Covers:
- Auto-label logic: CONFIRMED → positive, high-confidence no-match → negative
- Deduplication via seen_hashes.txt
- REVIEW items are placed in the human review queue, not training data
- New hashes are appended to seen_hashes.txt after a run
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from ner_ofac.training.feedback_loop import (
    _entity_hash,
    _load_seen_hashes,
    _save_seen_hashes,
    run_feedback,
)


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


class TestHashHelpers:
    def test_entity_hash_is_deterministic(self):
        h1 = _entity_hash("JOHN DOE", "SANCTIONED_PERSON")
        h2 = _entity_hash("JOHN DOE", "SANCTIONED_PERSON")
        assert h1 == h2

    def test_different_entities_have_different_hashes(self):
        h1 = _entity_hash("JOHN DOE", "SANCTIONED_PERSON")
        h2 = _entity_hash("JANE SMITH", "SANCTIONED_PERSON")
        assert h1 != h2

    def test_same_text_different_type_different_hash(self):
        h1 = _entity_hash("ACME", "SANCTIONED_PERSON")
        h2 = _entity_hash("ACME", "SANCTIONED_ORG")
        assert h1 != h2

    def test_case_insensitive(self):
        h1 = _entity_hash("JOHN DOE", "SANCTIONED_PERSON")
        h2 = _entity_hash("john doe", "SANCTIONED_PERSON")
        assert h1 == h2


class TestSeenHashes:
    def test_load_empty_file(self, tmp_path):
        path = str(tmp_path / "seen.txt")
        open(path, "w").close()
        result = _load_seen_hashes(path)
        assert result == set()

    def test_load_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.txt")
        result = _load_seen_hashes(path)
        assert result == set()

    def test_load_and_save_roundtrip(self, tmp_path):
        path = str(tmp_path / "seen.txt")
        hashes = ["abc123", "def456"]
        _save_seen_hashes(hashes, path)
        loaded = _load_seen_hashes(path)
        assert "abc123" in loaded
        assert "def456" in loaded

    def test_save_appends(self, tmp_path):
        path = str(tmp_path / "seen.txt")
        _save_seen_hashes(["aaa"], path)
        _save_seen_hashes(["bbb"], path)
        loaded = _load_seen_hashes(path)
        assert "aaa" in loaded
        assert "bbb" in loaded


# ---------------------------------------------------------------------------
# run_feedback
# ---------------------------------------------------------------------------


def _make_analysis_results(specs):
    """Build a list of analysis result dicts from (routing, composite, comprehend_score) tuples."""
    results = []
    for i, (routing, composite, comp_score) in enumerate(specs):
        results.append(
            {
                "row_index": i,
                "text": f"Sentence {i} containing an entity.",
                "entity_text": f"ENTITY_{i}",
                "entity_type": "SANCTIONED_PERSON",
                "matched_name": f"OFAC_NAME_{i}",
                "composite_score": composite,
                "comprehend_score": comp_score,
                "routing": routing,
                "BeginOffset": 10,
                "EndOffset": 20,
            }
        )
    return results


def _make_sentences_df(n):
    """Build a mock pandas-like object with to_dict."""
    import pandas as pd

    rows = [
        {"row_index": i, "text": f"Sentence {i} containing an entity.", "filename": f"doc_{i}.txt"}
        for i in range(n)
    ]
    return pd.DataFrame(rows)


class TestRunFeedback:
    def test_confirmed_goes_to_positive_annotations(self, tmp_path):
        """CONFIRMED results produce positive annotation rows."""
        results = _make_analysis_results([
            ("CONFIRMED", 0.95, 0.90),
            ("NO_MATCH", 0.30, 0.70),
        ])
        sentences_df = _make_sentences_df(2)
        seen_path = str(tmp_path / "seen.txt")

        with patch("ner_ofac.training.feedback_loop.upload_annotations") as mock_upload, \
             patch("ner_ofac.training.feedback_loop.update_flywheel_iteration") as mock_fly:
            mock_upload.return_value = "s3://bucket/test"
            count = run_feedback(
                results,
                sentences_df,
                flywheel_arn="arn:test",
                s3_bucket="bucket",
                output_dir=str(tmp_path),
                seen_hashes_path=seen_path,
            )

        assert count >= 1
        mock_upload.assert_called_once()
        mock_fly.assert_called_once()

    def test_review_goes_to_human_queue_not_training(self, tmp_path):
        """REVIEW items are written to the human queue file, not to training annotations."""
        results = _make_analysis_results([
            ("REVIEW", 0.70, 0.80),
        ])
        sentences_df = _make_sentences_df(1)
        seen_path = str(tmp_path / "seen.txt")

        with patch("ner_ofac.training.feedback_loop.upload_annotations") as mock_upload, \
             patch("ner_ofac.training.feedback_loop.update_flywheel_iteration") as mock_fly:
            count = run_feedback(
                results,
                sentences_df,
                flywheel_arn="arn:test",
                s3_bucket="bucket",
                output_dir=str(tmp_path),
                seen_hashes_path=seen_path,
            )

        # No training annotations from REVIEW items alone.
        assert count == 0
        mock_upload.assert_not_called()
        mock_fly.assert_not_called()

        # But a human review queue file should exist.
        files = os.listdir(tmp_path)
        review_files = [f for f in files if "human_review" in f]
        assert len(review_files) == 1

    def test_dedup_via_seen_hashes(self, tmp_path):
        """Entities already in seen_hashes.txt are skipped."""
        results = _make_analysis_results([
            ("CONFIRMED", 0.95, 0.92),
        ])
        sentences_df = _make_sentences_df(1)
        seen_path = str(tmp_path / "seen.txt")

        # Pre-populate the seen_hashes file with the hash of ENTITY_0.
        pre_hash = _entity_hash("ENTITY_0", "SANCTIONED_PERSON")
        _save_seen_hashes([pre_hash], seen_path)

        with patch("ner_ofac.training.feedback_loop.upload_annotations") as mock_upload, \
             patch("ner_ofac.training.feedback_loop.update_flywheel_iteration") as mock_fly:
            count = run_feedback(
                results,
                sentences_df,
                flywheel_arn="arn:test",
                s3_bucket="bucket",
                output_dir=str(tmp_path),
                seen_hashes_path=seen_path,
            )

        # Entity was already seen; should not be re-processed.
        assert count == 0
        mock_upload.assert_not_called()

    def test_new_hashes_appended_after_run(self, tmp_path):
        """After a successful run, new entity hashes are written to seen_hashes.txt."""
        results = _make_analysis_results([
            ("CONFIRMED", 0.95, 0.92),
        ])
        sentences_df = _make_sentences_df(1)
        seen_path = str(tmp_path / "seen.txt")

        with patch("ner_ofac.training.feedback_loop.upload_annotations") as mock_upload, \
             patch("ner_ofac.training.feedback_loop.update_flywheel_iteration"):
            mock_upload.return_value = "s3://bucket/test"
            run_feedback(
                results,
                sentences_df,
                flywheel_arn="arn:test",
                s3_bucket="bucket",
                output_dir=str(tmp_path),
                seen_hashes_path=seen_path,
            )

        loaded = _load_seen_hashes(seen_path)
        expected_hash = _entity_hash("ENTITY_0", "SANCTIONED_PERSON")
        assert expected_hash in loaded

    def test_high_confidence_no_match_auto_negative(self, tmp_path):
        """High Comprehend confidence + low composite → auto-labelled negative (counted in new_hashes)."""
        results = _make_analysis_results([
            ("NO_MATCH", 0.30, 0.90),  # comprehend_score >= 0.85, composite < 0.65
        ])
        sentences_df = _make_sentences_df(1)
        seen_path = str(tmp_path / "seen.txt")

        with patch("ner_ofac.training.feedback_loop.upload_annotations") as mock_upload, \
             patch("ner_ofac.training.feedback_loop.update_flywheel_iteration"):
            count = run_feedback(
                results,
                sentences_df,
                flywheel_arn="arn:test",
                s3_bucket="bucket",
                output_dir=str(tmp_path),
                seen_hashes_path=seen_path,
            )

        # Negative examples don't produce annotation rows but hash is tracked.
        loaded = _load_seen_hashes(seen_path)
        expected_hash = _entity_hash("ENTITY_0", "SANCTIONED_PERSON")
        assert expected_hash in loaded
