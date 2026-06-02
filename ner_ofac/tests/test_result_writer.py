"""Tests for ner_ofac.analysis.result_writer.

Covers:
- write_confirmed: appends CONFIRMED rows to CSV, creates header if absent,
  ignores non-CONFIRMED rows.
- write_review_queue: creates a timestamped file, contains REVIEW items only.
- write_ner_results: produces a CSV with the correct column set.
"""

import csv
import os
import tempfile

import pytest

from ner_ofac.analysis.result_writer import (
    write_confirmed,
    write_review_queue,
    write_ner_results,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_results(routings):
    """Build a minimal list of result dicts with the given routing labels."""
    results = []
    for i, routing in enumerate(routings):
        results.append(
            {
                "row_index": i,
                "text": f"Sentence {i}.",
                "entity_text": f"ENTITY_{i}",
                "entity_type": "SANCTIONED_PERSON",
                "comprehend_score": 0.95,
                "matched_name": f"OFAC NAME {i}",
                "composite_score": 0.92 if routing == "CONFIRMED" else (0.70 if routing == "REVIEW" else 0.40),
                "stage": "exact" if routing == "CONFIRMED" else "fuzzy",
                "routing": routing,
            }
        )
    return results


# ---------------------------------------------------------------------------
# write_confirmed
# ---------------------------------------------------------------------------


class TestWriteConfirmed:
    def test_appends_confirmed_rows(self, tmp_path):
        """Only CONFIRMED rows are written."""
        path = str(tmp_path / "confirmed.csv")
        results = _make_results(["CONFIRMED", "REVIEW", "NO_MATCH", "CONFIRMED"])
        count = write_confirmed(results, path=path)
        assert count == 2
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 2
        assert all(r["entity_type"] == "SANCTIONED_PERSON" for r in rows)

    def test_creates_header_if_file_absent(self, tmp_path):
        """Header row is written when the CSV file does not yet exist."""
        path = str(tmp_path / "new_confirmed.csv")
        results = _make_results(["CONFIRMED"])
        write_confirmed(results, path=path)
        with open(path, newline="", encoding="utf-8") as fh:
            first_line = fh.readline().strip()
        assert "entity_text" in first_line
        assert "composite_score" in first_line

    def test_appends_to_existing_file(self, tmp_path):
        """Second call appends rows without writing a second header."""
        path = str(tmp_path / "confirmed.csv")
        results1 = _make_results(["CONFIRMED"])
        results2 = _make_results(["CONFIRMED"])
        write_confirmed(results1, path=path)
        write_confirmed(results2, path=path)
        with open(path, newline="", encoding="utf-8") as fh:
            lines = [l for l in fh if l.strip()]
        # 1 header + 2 data rows.
        assert len(lines) == 3

    def test_no_confirmed_returns_zero(self, tmp_path):
        """Returns 0 when there are no CONFIRMED results."""
        path = str(tmp_path / "confirmed.csv")
        results = _make_results(["REVIEW", "NO_MATCH"])
        count = write_confirmed(results, path=path)
        assert count == 0
        assert not os.path.exists(path)


# ---------------------------------------------------------------------------
# write_review_queue
# ---------------------------------------------------------------------------


class TestWriteReviewQueue:
    def test_creates_timestamped_file(self, tmp_path):
        """A file with 'review_queue_' prefix is created in output_dir."""
        results = _make_results(["REVIEW", "CONFIRMED", "REVIEW"])
        output_dir = str(tmp_path)
        filepath = write_review_queue(results, output_dir=output_dir)
        assert filepath != ""
        assert os.path.isfile(filepath)
        assert "review_queue_" in os.path.basename(filepath)

    def test_contains_only_review_items(self, tmp_path):
        """Only REVIEW-routing items appear in the file."""
        results = _make_results(["CONFIRMED", "REVIEW", "NO_MATCH", "REVIEW"])
        output_dir = str(tmp_path)
        filepath = write_review_queue(results, output_dir=output_dir)
        with open(filepath, encoding="utf-8") as fh:
            content = fh.read()
        assert "ENTITY_1" in content  # routing=REVIEW
        assert "ENTITY_3" in content  # routing=REVIEW
        assert "ENTITY_0" not in content  # routing=CONFIRMED
        assert "ENTITY_2" not in content  # routing=NO_MATCH

    def test_no_review_items_returns_empty_string(self, tmp_path):
        """Returns empty string and creates no file if there are no REVIEW items."""
        results = _make_results(["CONFIRMED", "NO_MATCH"])
        output_dir = str(tmp_path)
        filepath = write_review_queue(results, output_dir=output_dir)
        assert filepath == ""
        files = os.listdir(output_dir)
        assert len(files) == 0


# ---------------------------------------------------------------------------
# write_ner_results
# ---------------------------------------------------------------------------


class TestWriteNerResults:
    def test_produces_correct_columns(self, tmp_path):
        """The NER results CSV contains the expected column headers."""
        results = _make_results(["CONFIRMED", "REVIEW", "NO_MATCH"])
        output_dir = str(tmp_path)
        filepath = write_ner_results(results, output_dir=output_dir)
        assert os.path.isfile(filepath)
        with open(filepath, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
        assert "entity_text" in fieldnames
        assert "entity_type" in fieldnames
        assert "composite_score" in fieldnames
        assert "routing" in fieldnames
        assert "matched_name" in fieldnames

    def test_row_count_matches_input(self, tmp_path):
        """One CSV row is written per result dict."""
        results = _make_results(["CONFIRMED", "REVIEW", "NO_MATCH"])
        output_dir = str(tmp_path)
        filepath = write_ner_results(results, output_dir=output_dir)
        with open(filepath, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 3

    def test_creates_timestamped_filename(self, tmp_path):
        """Output filename contains 'ner_results_' prefix."""
        results = _make_results(["CONFIRMED"])
        output_dir = str(tmp_path)
        filepath = write_ner_results(results, output_dir=output_dir)
        assert "ner_results_" in os.path.basename(filepath)
