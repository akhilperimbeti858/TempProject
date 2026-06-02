"""Ingestion module for the NER OFAC pipeline.

Reads raw sentence data from the ``ecata.sentences`` Hive/Spark table and
returns a Spark DataFrame for downstream preprocessing and inference.

Expected table columns
----------------------
a_number : string
    Unique identifier for the source document / docket.
source : string
    Origin system or dataset label.
line : int
    Line number within the source document.
section : string
    Section heading or identifier within the document.
text : string
    The raw sentence text to be analysed.
"""

import logging
from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


def load_sentences(spark: SparkSession) -> DataFrame:
    """Load all sentences from the ``ecata.sentences`` Spark table.

    Args:
        spark: An active SparkSession with access to the ``ecata`` database.

    Returns:
        A Spark DataFrame containing columns ``a_number``, ``source``,
        ``line``, ``section``, and ``text``.

    Raises:
        pyspark.sql.utils.AnalysisException: If the ``ecata.sentences`` table
            does not exist or is inaccessible.

    Example::

        spark = SparkSession.builder.appName("ner_ofac").getOrCreate()
        df = load_sentences(spark)
        df.show(5)
    """
    logger.info("Loading sentences from ecata.sentences")
    df = spark.table("ecata.sentences")
    row_count = df.count()
    logger.info("Loaded %d rows from ecata.sentences", row_count)
    return df
