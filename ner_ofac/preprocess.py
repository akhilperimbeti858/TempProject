"""Preprocessing module for the NER OFAC pipeline.

Provides two cleaning steps applied before Comprehend inference:

1. **Language filtering** — keeps only English sentences using ``langdetect``.
   Non-English rows are dropped and the count is logged.
2. **Text normalization** — applies Unicode NFKD decomposition followed by
   ASCII encoding (with ``errors='ignore'``) to remove diacritics and
   non-ASCII characters.  Case and punctuation are intentionally preserved
   so that Comprehend entity boundaries remain valid.

The ``preprocess`` function composes both steps into a single pipeline.
"""

import logging
import unicodedata
from typing import Tuple

from langdetect import detect, LangDetectException
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, BooleanType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-Python helpers
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize a single string with Unicode NFKD → ASCII transliteration.

    Applies ``unicodedata.normalize("NFKD", text)`` to decompose accented
    characters into their base letter + combining mark form, then encodes to
    ASCII with ``errors='ignore'`` to strip the non-ASCII combining marks.
    Case and punctuation are unchanged.

    Args:
        text: Raw input string, potentially containing accented or non-ASCII
            characters.

    Returns:
        A cleaned string containing only ASCII characters, with diacritics
        removed.  Returns an empty string if ``text`` is ``None`` or empty.

    Example::

        >>> normalize_text("Ångström Résumé")
        'Angstrom Resume'
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return nfkd.encode("ascii", errors="ignore").decode("ascii")


def _is_english(text: str) -> bool:
    """Return ``True`` if ``langdetect`` classifies *text* as English.

    Args:
        text: The sentence to classify.

    Returns:
        ``True`` when the detected language code is ``"en"``, ``False`` for
        all other languages or when detection fails.
    """
    if not text or not text.strip():
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        logger.debug("langdetect failed for text: %r — treating as non-English", text[:80])
        return False


# ---------------------------------------------------------------------------
# Spark UDFs
# ---------------------------------------------------------------------------

_is_english_udf = F.udf(_is_english, BooleanType())
_normalize_udf = F.udf(normalize_text, StringType())


# ---------------------------------------------------------------------------
# DataFrame-level functions
# ---------------------------------------------------------------------------


def filter_english(df: DataFrame) -> Tuple[DataFrame, int]:
    """Keep only rows whose ``text`` column is detected as English.

    Applies ``langdetect`` to each row via a Spark UDF.  Dropped rows (those
    classified as non-English or where detection fails) are counted and logged.

    Args:
        df: Input Spark DataFrame.  Must contain a ``text`` column of type
            StringType.

    Returns:
        A 2-tuple of:
        - **filtered_df**: Spark DataFrame containing only English rows.
        - **dropped_count**: Number of rows removed.

    Raises:
        pyspark.sql.utils.AnalysisException: If the ``text`` column is absent
            from *df*.
    """
    total = df.count()
    filtered_df = df.filter(_is_english_udf(F.col("text")))
    kept = filtered_df.count()
    dropped_count = total - kept
    logger.info(
        "Language filter: kept %d / %d rows (%d dropped as non-English)",
        kept,
        total,
        dropped_count,
    )
    return filtered_df, dropped_count


def preprocess(df: DataFrame) -> DataFrame:
    """Run the full preprocessing pipeline on a sentences DataFrame.

    Steps applied in order:

    1. ``filter_english`` — drop non-English rows.
    2. ``normalize_text`` — NFKD + ASCII normalisation applied to the
       ``text`` column; result stored in a new ``text_normalized`` column.

    Args:
        df: Spark DataFrame with at minimum a ``text`` column.

    Returns:
        A new Spark DataFrame with an added ``text_normalized`` column
        containing the cleaned text.  The original ``text`` column is
        preserved unchanged.
    """
    logger.info("Starting preprocessing pipeline")
    filtered_df, dropped = filter_english(df)
    result_df = filtered_df.withColumn("text_normalized", _normalize_udf(F.col("text")))
    logger.info("Preprocessing complete. %d rows retained.", result_df.count())
    return result_df
