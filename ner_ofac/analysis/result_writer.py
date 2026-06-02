"""Result writer for the NER OFAC pipeline.

Writes pipeline output to three artifact types:

* **NER results CSV** — full record of every detected entity with match details.
* **Confirmed entities CSV** — append-only log of CONFIRMED matches.
* **Review queue text file** — timestamped file listing entities that need
  human review, one per line.

All file paths are controlled by ``config.OUTPUT_DIR`` and
``config.CONFIRMED_ENTITIES_PATH``.
"""

import csv
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd

from ner_ofac import config

logger = logging.getLogger(__name__)

# Columns written to the NER results CSV.
_NER_COLUMNS = [
    "row_index",
    "text",
    "entity_text",
    "entity_type",
    "comprehend_score",
    "matched_name",
    "composite_score",
    "stage",
    "routing",
]

# Columns for the confirmed-entities CSV.
_CONFIRMED_COLUMNS = [
    "entity_text",
    "entity_type",
    "matched_name",
    "composite_score",
    "timestamp",
]


def _ensure_dir(path: str) -> None:
    """Create *path* and all intermediate directories if they do not exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def write_ner_results(
    results: List[Dict],
    output_dir: str = config.OUTPUT_DIR,
) -> str:
    """Write the full NER inference + OFAC match results to a CSV file.

    Each element of *results* should be a dict containing at minimum the keys
    in ``_NER_COLUMNS``.  Missing keys are written as empty strings.

    Args:
        results: List of result dicts, one per detected entity.
        output_dir: Directory in which to create the CSV file.

    Returns:
        The absolute path of the written CSV file.

    Raises:
        OSError: If the output directory cannot be created or the file cannot
            be written.
    """
    _ensure_dir(output_dir)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = os.path.join(output_dir, f"ner_results_{timestamp}.csv")

    with open(filename, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_NER_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({col: row.get(col, "") for col in _NER_COLUMNS})

    logger.info("Wrote %d NER result rows to %s", len(results), filename)
    return filename


def write_confirmed(
    results: List[Dict],
    path: str = config.CONFIRMED_ENTITIES_PATH,
) -> int:
    """Append CONFIRMED matches to the running confirmed-entities CSV.

    Creates the CSV with a header row if it does not already exist.  Only rows
    where ``routing == "CONFIRMED"`` are written.

    Args:
        results: List of result dicts.  Each dict should contain keys
            ``entity_text``, ``entity_type``, ``matched_name``,
            ``composite_score``, and optionally ``timestamp``.
        path: Path to the confirmed entities CSV file.

    Returns:
        The number of rows appended.

    Raises:
        OSError: If the file cannot be created or written.
    """
    confirmed = [r for r in results if r.get("routing") == "CONFIRMED"]
    if not confirmed:
        logger.info("No CONFIRMED matches to write.")
        return 0

    file_exists = os.path.isfile(path)
    timestamp = datetime.now(tz=timezone.utc).isoformat()

    _ensure_dir(os.path.dirname(os.path.abspath(path)))

    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CONFIRMED_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in confirmed:
            writer.writerow(
                {
                    "entity_text": row.get("entity_text", ""),
                    "entity_type": row.get("entity_type", ""),
                    "matched_name": row.get("matched_name", ""),
                    "composite_score": row.get("composite_score", ""),
                    "timestamp": row.get("timestamp", timestamp),
                }
            )

    logger.info("Appended %d CONFIRMED matches to %s", len(confirmed), path)
    return len(confirmed)


def write_review_queue(
    results: List[Dict],
    output_dir: str = config.OUTPUT_DIR,
) -> str:
    """Write REVIEW-tier entities to a timestamped human-review queue file.

    Creates a plain-text file (one entity per line) containing the entity text,
    matched OFAC name, and composite score for each REVIEW-routed result.
    The filename includes a UTC timestamp so successive runs do not overwrite
    each other.

    Args:
        results: List of result dicts.  Only rows where
            ``routing == "REVIEW"`` are included.
        output_dir: Directory in which to create the review queue file.

    Returns:
        The absolute path of the written file.  Returns an empty string if
        there are no REVIEW-tier results.

    Raises:
        OSError: If the output directory cannot be created or the file cannot
            be written.
    """
    review_items = [r for r in results if r.get("routing") == "REVIEW"]
    if not review_items:
        logger.info("No REVIEW-tier matches; review queue file not created.")
        return ""

    _ensure_dir(output_dir)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = os.path.join(output_dir, f"review_queue_{timestamp}.txt")

    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(f"# Review Queue — generated {timestamp}\n")
        fh.write("# Format: entity_text | matched_name | composite_score | entity_type\n\n")
        for item in review_items:
            fh.write(
                "{entity_text} | {matched_name} | {composite_score:.4f} | {entity_type}\n".format(
                    entity_text=item.get("entity_text", ""),
                    matched_name=item.get("matched_name", ""),
                    composite_score=float(item.get("composite_score", 0.0)),
                    entity_type=item.get("entity_type", ""),
                )
            )

    logger.info(
        "Wrote %d REVIEW items to review queue file: %s", len(review_items), filename
    )
    return filename
