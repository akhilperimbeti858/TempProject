"""Comprehend Flywheel management for continuous NER model improvement.

Provides helpers to create a Flywheel, trigger new training iterations,
poll iteration status, and activate a new model version once training
completes.

Amazon Comprehend Flywheel automates the process of retraining a custom NER
model whenever new annotated data is available.  Each call to
``update_flywheel_iteration`` submits a new training data configuration and
returns an iteration ARN that can be polled until the model is ready.
"""

import logging
import time
from typing import Dict, Optional

import boto3

from ner_ofac import config

logger = logging.getLogger(__name__)


def _client(region: str = config.AWS_REGION):
    """Return a boto3 comprehend client.

    Args:
        region: AWS region string.

    Returns:
        A boto3 Comprehend client.
    """
    return boto3.client("comprehend", region_name=region)


def create_flywheel(
    name: str,
    data_lake_s3_uri: str,
    model_type: str = "ENTITY_RECOGNIZER",
) -> str:
    """Create a new Amazon Comprehend Flywheel.

    Args:
        name: Human-readable name for the Flywheel (must be unique within the
            account/region).
        data_lake_s3_uri: S3 URI of the data lake where Flywheel will store
            training data and model artifacts.
        model_type: Comprehend model type.  Currently only
            ``"ENTITY_RECOGNIZER"`` is supported for NER.

    Returns:
        The ARN of the newly created Flywheel.

    Raises:
        botocore.exceptions.ClientError: If the API call fails (e.g. name
            conflict, insufficient permissions).
    """
    client = _client()
    response = client.create_flywheel(
        FlywheelName=name,
        ModelType=model_type,
        DataLakeS3Uri=data_lake_s3_uri,
    )
    flywheel_arn: str = response["FlywheelArn"]
    logger.info("Created Flywheel '%s' → ARN: %s", name, flywheel_arn)
    return flywheel_arn


def update_flywheel_iteration(
    flywheel_arn: str,
    training_data_config: Dict,
) -> str:
    """Trigger a new Flywheel training iteration with updated training data.

    Args:
        flywheel_arn: ARN of the Flywheel to update.
        training_data_config: Dict passed as ``TrainingDataConfig`` to the
            Comprehend ``create_flywheel_iteration`` API.  At minimum it should
            contain an ``"S3Uri"`` key pointing to the annotated training data.

    Returns:
        The ARN of the new Flywheel iteration.

    Raises:
        botocore.exceptions.ClientError: On API failure.

    Example::

        iteration_arn = update_flywheel_iteration(
            flywheel_arn="arn:aws:comprehend:...",
            training_data_config={"S3Uri": "s3://bucket/training/"},
        )
    """
    client = _client()
    response = client.create_flywheel_iteration(
        FlywheelArn=flywheel_arn,
        FlywheelIterationId=None,  # Comprehend auto-assigns an ID.
    )
    iteration_arn: str = response.get("FlywheelIterationProperties", {}).get(
        "FlywheelIterationArn", ""
    )
    logger.info(
        "Started Flywheel iteration for %s → iteration ARN: %s",
        flywheel_arn,
        iteration_arn,
    )
    return iteration_arn


def poll_flywheel_status(
    flywheel_arn: str,
    iteration_arn: str,
    poll_interval: int = 60,
) -> str:
    """Poll a Flywheel iteration until it reaches a terminal state.

    Loops, calling ``describe_flywheel_iteration`` every *poll_interval*
    seconds until the status is one of: ``COMPLETED``, ``FAILED``,
    ``STOPPED``.

    Args:
        flywheel_arn: ARN of the parent Flywheel.
        iteration_arn: ARN of the specific iteration to poll.
        poll_interval: Seconds to wait between API calls.  Defaults to 60.

    Returns:
        The final status string (e.g. ``"COMPLETED"`` or ``"FAILED"``).

    Raises:
        botocore.exceptions.ClientError: On API failure.
    """
    terminal_states = {"COMPLETED", "FAILED", "STOPPED"}
    client = _client()

    # Extract iteration ID from ARN (last component after '/').
    iteration_id = iteration_arn.split("/")[-1] if "/" in iteration_arn else iteration_arn

    logger.info(
        "Polling Flywheel iteration %s every %ds …", iteration_id, poll_interval
    )
    while True:
        response = client.describe_flywheel_iteration(
            FlywheelArn=flywheel_arn,
            FlywheelIterationId=iteration_id,
        )
        props = response.get("FlywheelIterationProperties", {})
        status: str = props.get("Status", "UNKNOWN")
        logger.info("Flywheel iteration %s status: %s", iteration_id, status)

        if status in terminal_states:
            return status

        time.sleep(poll_interval)


def activate_model_version(
    flywheel_arn: str,
    model_arn: str,
) -> None:
    """Activate a specific model version on a Flywheel.

    After a training iteration completes successfully, call this function to
    promote the new model version to the active endpoint.

    Args:
        flywheel_arn: ARN of the Flywheel.
        model_arn: ARN of the model version to activate.

    Raises:
        botocore.exceptions.ClientError: On API failure.
    """
    client = _client()
    client.update_flywheel(
        FlywheelArn=flywheel_arn,
        ActiveModelArn=model_arn,
    )
    logger.info(
        "Activated model version %s on Flywheel %s", model_arn, flywheel_arn
    )
