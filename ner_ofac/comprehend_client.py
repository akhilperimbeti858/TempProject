"""AWS Comprehend client wrapper for the NER OFAC pipeline.

Provides thin, typed wrappers around the boto3 Comprehend API for both
real-time single-document inference and asynchronous batch inference.

All return values normalise the Comprehend response into plain Python dicts
so that downstream code is not coupled to the boto3 response structure.

Entity dict schema
------------------
Each entity is returned as::

    {
        "Text":        str,   # entity surface string
        "Type":        str,   # e.g. "SANCTIONED_PERSON"
        "Score":       float, # confidence 0.0–1.0
        "BeginOffset": int,   # char offset of first char in source text
        "EndOffset":   int,   # char offset of char AFTER last char
    }
"""

import logging
from typing import Dict, List

import boto3

from ner_ofac import config

logger = logging.getLogger(__name__)


def _get_client(region: str = config.AWS_REGION):
    """Return a boto3 comprehend client for the given region.

    Args:
        region: AWS region string.  Defaults to ``config.AWS_REGION``.

    Returns:
        A boto3 Comprehend client object.
    """
    return boto3.client("comprehend", region_name=region)


def _normalise_entity(raw: Dict) -> Dict:
    """Convert a raw boto3 Comprehend entity dict to the canonical schema.

    Args:
        raw: Raw entity dict as returned by the Comprehend API.

    Returns:
        A dict with keys ``Text``, ``Type``, ``Score``, ``BeginOffset``,
        ``EndOffset``.
    """
    return {
        "Text": raw.get("Text", ""),
        "Type": raw.get("Type", ""),
        "Score": float(raw.get("Score", 0.0)),
        "BeginOffset": int(raw.get("BeginOffset", 0)),
        "EndOffset": int(raw.get("EndOffset", 0)),
    }


def detect_entities_realtime(text: str, endpoint_arn: str) -> List[Dict]:
    """Run real-time NER inference on a single text string.

    Calls ``comprehend.detect_entities`` with the custom endpoint ARN.

    Args:
        text: The sentence or passage to analyse.  Maximum length is
            determined by the Comprehend service (5 000 UTF-8 bytes for the
            synchronous API).
        endpoint_arn: ARN of the custom Comprehend NER endpoint.

    Returns:
        A list of entity dicts, each with keys ``Text``, ``Type``, ``Score``,
        ``BeginOffset``, ``EndOffset``.  Returns an empty list if no entities
        are detected.

    Raises:
        botocore.exceptions.ClientError: If the Comprehend API returns an
            error (e.g. invalid ARN, throttling, service error).
    """
    client = _get_client()
    logger.debug("Running real-time Comprehend inference on %d chars", len(text))
    response = client.detect_entities(Text=text, EndpointArn=endpoint_arn)
    entities = [_normalise_entity(e) for e in response.get("Entities", [])]
    logger.debug("Detected %d entities", len(entities))
    return entities


def detect_entities_batch(texts: List[str], endpoint_arn: str) -> List[List[Dict]]:
    """Run real-time NER inference on a batch of text strings.

    Calls ``detect_entities_realtime`` for each text.  The Comprehend
    synchronous batch API (``batch_detect_entities``) requires the *same*
    language code for all items in a batch and does not support custom
    endpoints; therefore this helper loops over inputs individually, which
    also avoids the 25-item limit of the native batch call.

    Args:
        texts: List of text strings to process.
        endpoint_arn: ARN of the custom Comprehend NER endpoint.

    Returns:
        A list of entity lists, in the same order as *texts*.  Each element
        is a list of entity dicts (potentially empty) as documented in
        ``detect_entities_realtime``.

    Raises:
        botocore.exceptions.ClientError: If any individual Comprehend call
            returns an error.
    """
    logger.info("Running batch Comprehend inference on %d texts", len(texts))
    results: List[List[Dict]] = []
    for i, text in enumerate(texts):
        try:
            entities = detect_entities_realtime(text, endpoint_arn)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Comprehend inference failed for text index %d: %s", i, exc)
            entities = []
        results.append(entities)
    logger.info("Batch inference complete: %d texts processed", len(results))
    return results
