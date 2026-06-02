"""S3 upload helpers for training documents and annotation CSVs.

Uploads the annotated training corpus to S3 so that it can be consumed by
Amazon Comprehend for custom NER training and Flywheel iteration updates.

Two upload paths are provided:
1. ``upload_documents`` — uploads a local directory of plain-text documents.
2. ``upload_annotations`` — serialises a pandas annotation DataFrame to a
   temporary CSV and uploads it to S3.
"""

import io
import logging
import os
from typing import Optional

import boto3
import pandas as pd

from ner_ofac import config

logger = logging.getLogger(__name__)


def _s3_client(region: str = config.AWS_REGION):
    """Return a boto3 S3 client.

    Args:
        region: AWS region string.

    Returns:
        A boto3 S3 client object.
    """
    return boto3.client("s3", region_name=region)


def upload_documents(
    local_path: str,
    s3_bucket: str = config.S3_BUCKET,
    s3_prefix: str = "training/documents",
) -> str:
    """Upload all files from *local_path* to an S3 prefix.

    Walks the directory tree rooted at *local_path* and uploads each file,
    preserving relative path structure under *s3_prefix*.

    Args:
        local_path: Local directory (or single file) to upload.
        s3_bucket: Target S3 bucket name.
        s3_prefix: S3 key prefix under which files are stored.
            A trailing ``/`` is added automatically if absent.

    Returns:
        The S3 URI of the uploaded prefix, e.g.
        ``"s3://my-bucket/training/documents/"``.

    Raises:
        FileNotFoundError: If *local_path* does not exist.
        botocore.exceptions.ClientError: On S3 upload failure.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"local_path does not exist: {local_path}")

    s3_prefix = s3_prefix.rstrip("/") + "/"
    client = _s3_client()
    uploaded = 0

    if os.path.isfile(local_path):
        key = s3_prefix + os.path.basename(local_path)
        client.upload_file(local_path, s3_bucket, key)
        logger.info("Uploaded %s → s3://%s/%s", local_path, s3_bucket, key)
        uploaded += 1
    else:
        for root, _, files in os.walk(local_path):
            for fname in files:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, local_path)
                key = s3_prefix + rel_path.replace(os.sep, "/")
                client.upload_file(full_path, s3_bucket, key)
                logger.debug("Uploaded %s → s3://%s/%s", full_path, s3_bucket, key)
                uploaded += 1

    s3_uri = f"s3://{s3_bucket}/{s3_prefix}"
    logger.info("upload_documents: %d files uploaded to %s", uploaded, s3_uri)
    return s3_uri


def upload_annotations(
    annotation_df: pd.DataFrame,
    s3_bucket: str = config.S3_BUCKET,
    s3_prefix: str = "training/annotations",
    filename: Optional[str] = None,
) -> str:
    """Serialise *annotation_df* to CSV and upload to S3.

    Args:
        annotation_df: Pandas DataFrame with Comprehend annotation columns
            (``File``, ``Line``, ``Begin Offset``, ``End Offset``, ``Type``).
        s3_bucket: Target S3 bucket name.
        s3_prefix: S3 key prefix under which the CSV is stored.
        filename: Optional object filename within the prefix.  Defaults to
            ``"annotations.csv"``.

    Returns:
        The full S3 URI of the uploaded annotation CSV, e.g.
        ``"s3://my-bucket/training/annotations/annotations.csv"``.

    Raises:
        botocore.exceptions.ClientError: On S3 upload failure.
        ValueError: If *annotation_df* is empty.
    """
    if annotation_df.empty:
        raise ValueError("annotation_df is empty; nothing to upload.")

    fname = filename or "annotations.csv"
    s3_prefix = s3_prefix.rstrip("/")
    key = f"{s3_prefix}/{fname}"

    csv_buffer = io.BytesIO()
    annotation_df.to_csv(csv_buffer, index=False, encoding="utf-8")
    csv_buffer.seek(0)

    client = _s3_client()
    client.put_object(
        Bucket=s3_bucket,
        Key=key,
        Body=csv_buffer.read(),
        ContentType="text/csv",
    )

    s3_uri = f"s3://{s3_bucket}/{key}"
    logger.info(
        "upload_annotations: uploaded %d rows to %s", len(annotation_df), s3_uri
    )
    return s3_uri
