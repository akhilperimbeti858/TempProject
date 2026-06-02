"""Feedback loop: converts analysis results into new Comprehend training data.

Implements the automated pipeline that turns pipeline output into labelled
training examples and triggers a new Flywheel iteration:

Auto-labelling rules
--------------------
* **Positive** (CONFIRMED entity): ``routing == "CONFIRMED"`` OR
  ``composite_score >= AUTO_LABEL_CONFIRMED_MIN`` → annotated with entity type.
* **Negative** (false positive suppression): Comprehend confidence
  ``>= AUTO_LABEL_NEGATIVE_MIN`` AND ``composite_score < REVIEW_THRESHOLD``
  → labelled as a non-entity / negative example.
* **REVIEW tier** (0.65–0.89): Entity text goes to a human review queue file,
  deduplicated by entity string so each unique surface form appears once.

Deduplication
-------------
Processed entity strings are hashed (SHA-256) and compared against
``seen_hashes.txt`` (one hex digest per line).  Only new hashes are added to
the training batch and then appended to ``seen_hashes.txt``.
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pandas as pd

from ner_ofac import config
from ner_ofac.training.annotation_prep import prepare_annotations
from ner_ofac.training.upload_to_s3 import upload_annotations
from ner_ofac.training.flywheel_manager import update_flywheel_iteration

logger = logging.getLogger(__name__)

# Path to the deduplication hash registry relative to this file.
_SEEN_HASHES_PATH = os.path.join(os.path.dirname(__file__), "seen_hashes.txt")


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


def _entity_hash(entity_text: str, entity_type: str) -> str:
    """Compute a SHA-256 hex digest for a (entity_text, entity_type) pair.

    Args:
        entity_text: Surface text of the entity.
        entity_type: Entity type label.

    Returns:
        Lowercase hex string of the SHA-256 digest.
    """
    key = f"{entity_type}::{entity_text.strip().lower()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _load_seen_hashes(path: str = _SEEN_HASHES_PATH) -> set:
    """Load previously processed entity hashes from *path*.

    Args:
        path: Path to the seen_hashes.txt file.

    Returns:
        A set of hex digest strings.
    """
    if not os.path.isfile(path):
        return set()
    with open(path, "r", encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}


def _save_seen_hashes(new_hashes: List[str], path: str = _SEEN_HASHES_PATH) -> None:
    """Append *new_hashes* to the seen_hashes.txt file.

    Args:
        new_hashes: List of hex digest strings to append.
        path: Path to the seen_hashes.txt file.
    """
    with open(path, "a", encoding="utf-8") as fh:
        for h in new_hashes:
            fh.write(h + "\n")


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------


def _write_human_review_queue(review_items: List[Dict], output_dir: str) -> str:
    """Write deduplicated REVIEW-tier items to a human review queue file.

    Args:
        review_items: List of result dicts with ``routing == "REVIEW"``.
        output_dir: Directory where the file is written.

    Returns:
        Path to the written file, or empty string if no items.
    """
    if not review_items:
        return ""

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(output_dir, f"human_review_{timestamp}.txt")

    # Deduplicate by entity_text.
    seen_texts: set = set()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# Human Review Queue — {timestamp}\n")
        fh.write("# entity_text | entity_type | matched_name | composite_score\n\n")
        for item in review_items:
            text = item.get("entity_text", "")
            if text in seen_texts:
                continue
            seen_texts.add(text)
            fh.write(
                "{} | {} | {} | {:.4f}\n".format(
                    text,
                    item.get("entity_type", ""),
                    item.get("matched_name", ""),
                    float(item.get("composite_score", 0.0)),
                )
            )

    logger.info("Wrote %d unique REVIEW items to %s", len(seen_texts), path)
    return path


# ---------------------------------------------------------------------------
# Core feedback function
# ---------------------------------------------------------------------------


def run_feedback(
    analysis_results: List[Dict],
    sentences_df,
    flywheel_arn: str = config.FLYWHEEL_ARN,
    s3_bucket: str = config.S3_BUCKET,
    output_dir: str = config.OUTPUT_DIR,
    seen_hashes_path: str = _SEEN_HASHES_PATH,
) -> int:
    """Convert analysis results into new training annotations and trigger retraining.

    Implements the full feedback pipeline:

    1. Separate results into CONFIRMED, REVIEW, and NO_MATCH groups.
    2. Apply auto-labelling rules to produce positive and negative examples.
    3. Deduplicate against ``seen_hashes.txt``.
    4. Write human review queue for REVIEW-tier items.
    5. If new annotations exist: upload to S3 and trigger a Flywheel iteration.
    6. Append new hashes to ``seen_hashes.txt``.

    Args:
        analysis_results: List of result dicts from the analysis pipeline.
            Expected keys per item: ``entity_text``, ``entity_type``,
            ``matched_name``, ``composite_score``, ``routing``,
            ``comprehend_score``, ``row_index``, ``BeginOffset``,
            ``EndOffset``, ``text``.
        sentences_df: The original sentences Spark or pandas DataFrame,
            used to look up full sentence text for annotation preparation.
        flywheel_arn: ARN of the Comprehend Flywheel.
        s3_bucket: S3 bucket name for annotation upload.
        output_dir: Directory for local output artefacts (review queue files).
        seen_hashes_path: Path to the seen_hashes.txt deduplication registry.

    Returns:
        The number of new annotation rows added to training.

    Raises:
        botocore.exceptions.ClientError: If the S3 upload or Flywheel API
            call fails.
    """
    seen_hashes = _load_seen_hashes(seen_hashes_path)
    new_hashes: List[str] = []
    positive_entities: List[Dict] = []
    negative_entities: List[Dict] = []
    review_items: List[Dict] = []

    # ------------------------------------------------------------------
    # Build sentence lookup from DataFrame.
    # ------------------------------------------------------------------
    try:
        rows = sentences_df.collect()
        sentence_lookup = {r["row_index"] if "row_index" in r else i: r.asDict() for i, r in enumerate(rows)}
    except AttributeError:
        records = sentences_df.to_dict(orient="records")
        sentence_lookup = {r.get("row_index", i): r for i, r in enumerate(records)}

    # ------------------------------------------------------------------
    # Classify each result.
    # ------------------------------------------------------------------
    for result in analysis_results:
        entity_text = result.get("entity_text", "")
        entity_type = result.get("entity_type", "")
        composite = float(result.get("composite_score", 0.0))
        routing = result.get("routing", "NO_MATCH")
        comprehend_score = float(result.get("comprehend_score", 0.0))

        h = _entity_hash(entity_text, entity_type)
        if h in seen_hashes:
            logger.debug("Skipping already-seen entity: '%s'", entity_text)
            continue

        if routing == "CONFIRMED" or composite >= config.AUTO_LABEL_CONFIRMED_MIN:
            positive_entities.append(result)
            new_hashes.append(h)
        elif (
            comprehend_score >= config.AUTO_LABEL_NEGATIVE_MIN
            and composite < config.REVIEW_THRESHOLD
        ):
            negative_entities.append(result)
            new_hashes.append(h)
        elif routing == "REVIEW":
            review_items.append(result)
            # REVIEW items go to human queue but are also hashed to avoid
            # repeatedly queuing the same entity.
            new_hashes.append(h)

    logger.info(
        "Feedback: %d positive, %d negative, %d review items (after dedup)",
        len(positive_entities),
        len(negative_entities),
        len(review_items),
    )

    # Write human review queue.
    _write_human_review_queue(review_items, output_dir)

    # ------------------------------------------------------------------
    # Prepare annotations for positive examples.
    # ------------------------------------------------------------------
    if not positive_entities:
        logger.info("No new positive entities to annotate; skipping Flywheel trigger.")
        _save_seen_hashes(new_hashes, seen_hashes_path)
        return 0

    # Convert positive results to the annotation format.
    # We need sentence-level context; build sentence list from lookup.
    sentences_for_annot = []
    entities_for_annot = []

    for result in positive_entities:
        row_index = result.get("row_index", 0)
        sentence_row = sentence_lookup.get(row_index, {})
        text = result.get("text") or sentence_row.get("text_normalized") or sentence_row.get("text", "")
        filename = sentence_row.get("filename") or f"doc_{row_index}.txt"

        sentences_for_annot.append(
            {"row_index": row_index, "text": text, "filename": filename}
        )
        entities_for_annot.append(
            {
                "row_index": row_index,
                "entity_text": result.get("entity_text", ""),
                "entity_type": result.get("entity_type", ""),
                "BeginOffset": result.get("BeginOffset", 0),
                "EndOffset": result.get("EndOffset", 0),
                "line": 0,
            }
        )

    annotation_df = prepare_annotations(sentences_for_annot, entities_for_annot)

    # ------------------------------------------------------------------
    # Upload to S3 and trigger Flywheel.
    # ------------------------------------------------------------------
    s3_uri = upload_annotations(
        annotation_df,
        s3_bucket=s3_bucket,
        s3_prefix="training/feedback_annotations",
    )

    training_data_config = {"S3Uri": s3_uri}
    update_flywheel_iteration(flywheel_arn, training_data_config)

    # ------------------------------------------------------------------
    # Persist new hashes.
    # ------------------------------------------------------------------
    _save_seen_hashes(new_hashes, seen_hashes_path)

    new_annotation_count = len(annotation_df)
    logger.info(
        "run_feedback complete: %d new annotation rows uploaded, Flywheel iteration triggered.",
        new_annotation_count,
    )
    return new_annotation_count
