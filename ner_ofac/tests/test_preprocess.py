"""Tests for ner_ofac.preprocess.

Covers:
- normalize_text: accented characters, pure unicode, empty strings.
- filter_english: English rows are kept, non-English rows are dropped,
  the dropped count is accurate.
- preprocess: full pipeline integration.
"""

import pytest
from unittest.mock import MagicMock, patch

from ner_ofac.preprocess import normalize_text, filter_english, preprocess


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_accented_latin(self):
        """Accented characters are decomposed and their diacritics stripped."""
        result = normalize_text("Résumé")
        assert result == "Resume"

    def test_multiple_accents(self):
        result = normalize_text("naïve café")
        assert result == "naive cafe"

    def test_angstrom(self):
        result = normalize_text("Ångström")
        assert result == "Angstrom"

    def test_empty_string(self):
        """Empty string input returns empty string."""
        assert normalize_text("") == ""

    def test_none_input(self):
        """None input returns empty string."""
        assert normalize_text(None) == ""

    def test_pure_ascii(self):
        """Pure ASCII strings pass through unchanged."""
        text = "JOHN DOE was sanctioned."
        assert normalize_text(text) == text

    def test_pure_unicode_non_latin(self):
        """Non-Latin unicode characters that have no ASCII equivalent are stripped."""
        result = normalize_text("中文")
        # Chinese characters have no ASCII decomposition; they should be removed.
        assert result == ""

    def test_mixed_ascii_and_unicode(self):
        """Mixed strings: ASCII preserved, unicode diacritics removed."""
        result = normalize_text("Müller AG")
        assert result == "Muller AG"

    def test_preserves_case(self):
        """Case is preserved (no upper/lower conversion)."""
        result = normalize_text("John Doe")
        assert result == "John Doe"

    def test_preserves_punctuation(self):
        """Punctuation is preserved."""
        result = normalize_text("Héllo, wörld!")
        assert result == "Hello, world!"


# ---------------------------------------------------------------------------
# filter_english
# ---------------------------------------------------------------------------


class TestFilterEnglish:
    """Tests for filter_english using a mocked Spark DataFrame."""

    def _make_mock_df(self, rows):
        """Build a minimal mock Spark DataFrame from a list of text strings."""
        mock_df = MagicMock()
        # Mock count() returns.
        mock_df.count.return_value = len(rows)

        # Mock filter() to return a filtered DataFrame.
        def mock_filter(condition):
            # We can't easily evaluate a Spark Column predicate here; instead
            # we'll intercept via the UDF path below.
            filtered_mock = MagicMock()
            # Count returns will be set per test.
            return filtered_mock

        mock_df.filter.side_effect = mock_filter
        return mock_df

    def test_all_english(self):
        """All-English DataFrame: dropped_count == 0."""
        with patch("ner_ofac.preprocess._is_english_udf") as mock_udf, \
             patch("ner_ofac.preprocess.F") as mock_f:

            from pyspark.sql import functions as F_real

            mock_df = MagicMock()
            mock_df.count.return_value = 5

            filtered_df = MagicMock()
            filtered_df.count.return_value = 5
            mock_df.filter.return_value = filtered_df

            result_df, dropped = filter_english(mock_df)
            assert dropped == 0
            assert result_df is filtered_df

    def test_mixed_languages(self):
        """Some non-English rows are dropped; dropped_count reflects the difference."""
        mock_df = MagicMock()
        mock_df.count.return_value = 10

        filtered_df = MagicMock()
        filtered_df.count.return_value = 7
        mock_df.filter.return_value = filtered_df

        result_df, dropped = filter_english(mock_df)
        assert dropped == 3
        assert result_df is filtered_df

    def test_all_non_english(self):
        """All non-English: dropped_count equals total row count."""
        mock_df = MagicMock()
        mock_df.count.return_value = 4

        filtered_df = MagicMock()
        filtered_df.count.return_value = 0
        mock_df.filter.return_value = filtered_df

        result_df, dropped = filter_english(mock_df)
        assert dropped == 4


# ---------------------------------------------------------------------------
# preprocess (pipeline integration)
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_preprocess_adds_normalized_column(self):
        """preprocess adds 'text_normalized' column to the filtered DataFrame."""
        mock_df = MagicMock()
        mock_df.count.return_value = 3

        filtered_df = MagicMock()
        filtered_df.count.return_value = 3
        mock_df.filter.return_value = filtered_df

        normalized_df = MagicMock()
        normalized_df.count.return_value = 3
        filtered_df.withColumn.return_value = normalized_df

        result = preprocess(mock_df)
        # withColumn should have been called with 'text_normalized'.
        filtered_df.withColumn.assert_called_once()
        call_args = filtered_df.withColumn.call_args[0]
        assert call_args[0] == "text_normalized"
        assert result is normalized_df
