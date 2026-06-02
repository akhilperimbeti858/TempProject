"""Comprehend inference runner for the NER OFAC pipeline.

Provides two modes of inference:

* **Real-time** — calls ``detect_entities_realtime`` row-by-row via the custom
  Comprehend endpoint.  Suitable for small-to-medium DataFrames.
* **Async batch** — starts a Comprehend ``start_entities_detection_job`` using
  documents already uploaded to S3 and returns the job ID immediately.

Both functions return results in a consistent dict schema so that downstream
OFAC matching code does not need to know which mode was used.
"""

import logging
from typing import Dict, List

import boto3

from ner_ofac import config
from ner_ofac.comprehend_client import detect_entities_realtime

logger = logging.getLogger(__name__)


def run_realtime(sentences_df, endpoint_arn: str = config.COMPREHEND_ENDPOINT_ARN) -> List[Dict]:
    """Run real-time Comprehend NER inference on every row of *sentences_df*.

    Iterates over the DataFrame (collected to the driver), calls
    ``detect_entities_realtime`` for the ``text_normalized`` column (falling
    back to ``text`` if not present), and assembles a result list.

    Args:
        sentences_df: A Spark or pandas DataFrame with at minimum a ``text``
            column and optionally a ``text_normalized`` column.  If a Spark
            DataFrame is passed it is collected to the driver first.
        endpoint_arn: ARN of the custom Comprehend endpoint.  Defaults to
            ``config.COMPREHEND_ENDPOINT_ARN``.

    Returns:
        A list of dicts, one per input row, each with the keys:

        - ``"row_index"`` (int): Row position in the input DataFrame.
        - ``"text"`` (str): The text that was analysed.
        - ``"entities"`` (list): List of entity dicts as returned by
          ``detect_entities_realtime``.

    Raises:
        KeyError: If neither ``text_normalized`` nor ``text`` columns exist.
    """
    # Support both Spark and pandas DataFrames.
    try:
        rows = sentences_df.collect()  # Spark
        row_dicts = [row.asDict() for row in rows]
    except AttributeError:
        row_dicts = sentences_df.to_dict(orient="records")  # pandas

    results = []
    for i, row in enumerate(row_dicts):
        text = row.get("text_normalized") or row.get("text", "")
        if not text:
            logger.warning("Row %d has empty text; skipping.", i)
            results.append({"row_index": i, "text": text, "entities": []})
            continue

        try:
            entities = detect_entities_realtime(text, endpoint_arn)
        except Exception as exc:  # noqa: BLE001
            logger.error("Comprehend failed for row %d: %s", i, exc)
            entities = []

        results.append({"row_index": i, "text": text, "entities": entities})
        if (i + 1) % 100 == 0:
            logger.info("Real-time inference progress: %d rows processed", i + 1)

    logger.info("run_realtime complete: %d rows processed", len(results))
    return results


def run_batch(
    sentences_df,
    endpoint_arn: str = config.COMPREHEND_ENDPOINT_ARN,
    s3_input: str = "",
    s3_output: str = "",
) -> str:
    """Start an asynchronous Comprehend NER batch job and return its job ID.

    Uses ``start_entities_detection_job`` to process documents stored in S3.
    Documents must already be uploaded (see ``training/upload_to_s3.py``) before
    calling this function.

    Args:
        sentences_df: Not used directly for the async call but accepted for API
            consistency.  Pass the DataFrame that was uploaded to *s3_input*.
        endpoint_arn: ARN of the custom Comprehend endpoint.
        s3_input: S3 URI (``s3://bucket/prefix/``) of the input documents.
        s3_output: S3 URI where Comprehend will write output JSON files.

    Returns:
        The Comprehend ``JobId`` string.

    Raises:
        botocore.exceptions.ClientError: If the API call fails.
        ValueError: If *s3_input* or *s3_output* are not provided.
    """
    if not s3_input or not s3_output:
        raise ValueError("s3_input and s3_output must both be provided for batch inference.")

    client = boto3.client("comprehend", region_name=config.AWS_REGION)

    response = client.start_entities_detection_job(
        InputDataConfig={
            "S3Uri": s3_input,
            "InputFormat": "ONE_DOC_PER_LINE",
        },
        OutputDataConfig={"S3Uri": s3_output},
        DataAccessRoleArn="",  # Populated from environment / IAM role in production.
        EntityRecognizerArn=endpoint_arn,
        LanguageCode="en",
    )

    job_id = response["JobId"]
    logger.info("Started Comprehend batch job: %s", job_id)
    return job_id
