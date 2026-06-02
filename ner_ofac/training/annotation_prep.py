"""Annotation preparation for Comprehend custom NER training.

Converts raw sentences and their detected/confirmed entity spans into the CSV
annotation format required by Amazon Comprehend for training a custom entity
recogniser.

Comprehend annotation CSV format (one row per entity annotation):
    File        — filename of the corresponding document
    Line        — 0-based line number within the document
    Begin Offset— character offset of the entity start (0-based)
    End Offset  — character offset of the character AFTER the entity end
    Type        — entity type label (e.g. ``SANCTIONED_PERSON``)
"""

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

# Required column names as expected by the Comprehend annotation CSV.
_ANNOTATION_COLUMNS = ["File", "Line", "Begin Offset", "End Offset", "Type"]


def prepare_annotations(
    sentences: List[Dict],
    entities: List[Dict],
) -> pd.DataFrame:
    """Build a Comprehend annotation DataFrame from sentences and entity spans.

    Each element of *entities* must reference a sentence in *sentences* via a
    ``row_index`` key.  The entity's character offsets (``BeginOffset`` and
    ``EndOffset``) are taken from the Comprehend detection result and must be
    valid for the corresponding sentence text.

    Args:
        sentences: List of sentence dicts.  Each dict must contain:
            - ``"row_index"`` (int): Position in the original DataFrame.
            - ``"text"`` (str): The sentence text (used to validate offsets).
            - ``"filename"`` (str, optional): Document filename.  Defaults to
              ``"doc_{row_index}.txt"`` if absent.
        entities: List of entity annotation dicts.  Each dict must contain:
            - ``"row_index"`` (int): Matches a ``row_index`` in *sentences*.
            - ``"entity_text"`` (str): Surface text of the entity.
            - ``"entity_type"`` (str): Label such as ``"SANCTIONED_PERSON"``.
            - ``"BeginOffset"`` (int): Character start offset.
            - ``"EndOffset"`` (int): Character end offset (exclusive).
            - ``"line"`` (int, optional): Line number within the document.
              Defaults to ``0``.

    Returns:
        A ``pandas.DataFrame`` with columns ``File``, ``Line``,
        ``Begin Offset``, ``End Offset``, ``Type``.  Rows are sorted by
        ``File`` then ``Line`` then ``Begin Offset``.

    Raises:
        ValueError: If *entities* is empty.

    Example::

        sentences = [{"row_index": 0, "text": "JOHN DOE was sanctioned.", "filename": "doc_0.txt"}]
        entities = [
            {"row_index": 0, "entity_text": "JOHN DOE", "entity_type": "SANCTIONED_PERSON",
             "BeginOffset": 0, "EndOffset": 8, "line": 0}
        ]
        df = prepare_annotations(sentences, entities)
    """
    if not entities:
        raise ValueError("entities list must not be empty.")

    # Build a lookup from row_index → sentence dict.
    sentence_lookup: Dict[int, Dict] = {s["row_index"]: s for s in sentences}

    rows = []
    for ent in entities:
        row_index = ent.get("row_index", 0)
        sentence = sentence_lookup.get(row_index, {})
        filename = sentence.get("filename") or f"doc_{row_index}.txt"
        line_num = int(ent.get("line", 0))
        begin = int(ent.get("BeginOffset", 0))
        end = int(ent.get("EndOffset", 0))
        ent_type = ent.get("entity_type", "")

        # Validate offsets against sentence text if available.
        text = sentence.get("text", "")
        if text and end > len(text):
            logger.warning(
                "Entity '%s' EndOffset %d exceeds sentence length %d for row %d; skipping.",
                ent.get("entity_text", ""),
                end,
                len(text),
                row_index,
            )
            continue

        rows.append(
            {
                "File": filename,
                "Line": line_num,
                "Begin Offset": begin,
                "End Offset": end,
                "Type": ent_type,
            }
        )

    annotation_df = pd.DataFrame(rows, columns=_ANNOTATION_COLUMNS)
    annotation_df = annotation_df.sort_values(
        by=["File", "Line", "Begin Offset"]
    ).reset_index(drop=True)

    logger.info("prepare_annotations: produced %d annotation rows", len(annotation_df))
    return annotation_df
