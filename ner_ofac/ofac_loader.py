"""OFAC SDN list loader for the NER OFAC pipeline.

Reads the OFAC Specially Designated Nationals (SDN) CSV file and returns
per-category name lists keyed by the entity types used in this pipeline
(``SANCTIONED_PERSON`` and ``SANCTIONED_ORG``).

OFAC SDN CSV column reference
-------------------------------
ent_num   : Unique entity identifier.
SDN_Name  : Primary name of the sanctioned party.
SDN_Type  : Classification — ``"individual"`` for persons; ``"entity"``,
            ``"vessel"``, or ``"aircraft"`` for organisations / assets.
Program   : Sanctions programme(s) applied.
Title     : Honorific or job title (persons only).
Call_Sign : Vessel call sign.
Vess_type : Vessel type description.
Tonnage   : Vessel tonnage.
GRT       : Gross registered tonnage.
Vess_flag : Flag state of the vessel.
Vess_owner: Vessel owner.
Remarks   : Free-text remarks.

Mapping applied
---------------
- ``SDN_Type == "individual"``  →  ``SANCTIONED_PERSON``
- All other SDN_Type values     →  ``SANCTIONED_ORG``

Only rows with a non-null, non-empty ``SDN_Name`` are included.
"""

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

# Column expected to hold the entity's primary name in the SDN CSV.
_NAME_COL = "SDN_Name"
# Column used to distinguish persons from organisations.
_TYPE_COL = "SDN_Type"


def load_ofac(path: str) -> Dict[str, List[str]]:
    """Load the OFAC SDN CSV and return per-category name lists.

    Reads the CSV at *path*, splits rows into ``SANCTIONED_PERSON`` (rows
    where ``SDN_Type`` is ``"individual"``) and ``SANCTIONED_ORG`` (all
    other non-null SDN_Type values), and returns a dict of deduplicated,
    stripped name strings.

    Args:
        path: File-system path to the OFAC SDN CSV file.

    Returns:
        A dict with exactly two keys:

        - ``"SANCTIONED_PERSON"`` — list of unique individual names.
        - ``"SANCTIONED_ORG"``    — list of unique org/vessel/aircraft names.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If the CSV is missing the ``SDN_Name`` or ``SDN_Type``
            columns.
        pd.errors.ParserError: If the CSV cannot be parsed.

    Example::

        names = load_ofac("./data/ofac_sdn.csv")
        print(len(names["SANCTIONED_PERSON"]), "individuals loaded")
        print(len(names["SANCTIONED_ORG"]),    "orgs/vessels/aircraft loaded")
    """
    logger.info("Loading OFAC SDN list from: %s", path)
    df = pd.read_csv(path, dtype=str, encoding="utf-8", on_bad_lines="warn")

    # Validate required columns exist.
    for col in (_NAME_COL, _TYPE_COL):
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' not found in OFAC CSV. "
                f"Available columns: {list(df.columns)}"
            )

    # Drop rows with missing names.
    original_len = len(df)
    df = df.dropna(subset=[_NAME_COL])
    df = df[df[_NAME_COL].str.strip() != ""]
    logger.info(
        "Dropped %d rows with null/empty SDN_Name; %d rows remain",
        original_len - len(df),
        len(df),
    )

    # Normalise the type column to lower-case stripped strings.
    df[_TYPE_COL] = df[_TYPE_COL].fillna("").str.strip().str.lower()

    person_mask = df[_TYPE_COL] == "individual"
    org_mask = ~person_mask

    persons = sorted(set(df.loc[person_mask, _NAME_COL].str.strip().tolist()))
    orgs = sorted(set(df.loc[org_mask, _NAME_COL].str.strip().tolist()))

    logger.info(
        "OFAC SDN loaded: %d SANCTIONED_PERSON names, %d SANCTIONED_ORG names",
        len(persons),
        len(orgs),
    )

    return {
        "SANCTIONED_PERSON": persons,
        "SANCTIONED_ORG": orgs,
    }
