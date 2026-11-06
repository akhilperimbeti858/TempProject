"""Tests for dq_merge_lean.py, using the real ORR_DATA / AFF_DATA column names.

Top priority: every ORR uac_alien_no that exists as an a_number in AFF_DATA must come out matched
(asylum_case_exists = True, matched_a_number = that a_number), whatever its dq_status.
"""
import os
import sys

import pytest
from pyspark.sql import SparkSession

import dq_merge_lean as lean

ORR_SCHEMA = ("uac_alien_no string, uac_first_name string, uac_last_name string, uac_sex string, "
              "uac_dob string, uac_cob string, uac_aka string, uac_status string, current_status string")
AFF_SCHEMA = ("a_number string, first_name string, middle_name string, last_name string, sex string, "
              "date_of_birth string, country_of_birth string, receipt_number string, case_id string")


def orr(key, first="Juan", last="Perez", sex="M", dob="2012-03-03", cob="Honduras", aka=None):
    return (key, first, last, sex, dob, cob, aka, "Active", "Released")


def aff(key, first="Juan", middle=None, last="Perez", sex="M", dob="2012-03-03", cob="Honduras"):
    return (key, first, middle, last, sex, dob, cob, f"IOE{abs(hash(key)) % 10**7}", f"C{abs(hash(key)) % 10**5}")


# The same a_number written differently in the two tables. Every pair must match.
KEY_VARIANTS = [  # (ORR uac_alien_no, AFF a_number)
    ("100000001", "100000001"),            # identical
    ("A100000002", "100000002"),           # A prefix
    ("a-100-000-003", "100000003"),        # lower case + hyphens
    (" 100000004 ", "100000004"),          # spaces
    ("100000005 ", "100000005"),      # non-breaking space (Excel / web exports)
    ("100000006", "​100000006"),      # zero-width space
    ("100000007.0", "100000007"),          # number saved as text with .0
    ("1.00000008E8", "100000008"),         # scientific notation saved as text
    ("0100000009", "100000009"),           # 10-character A-number with a leading zero
    ("A0100000010", "100000010"),          # A + 10 characters
    ("'100000011", "100000011"),           # leading apostrophe (Excel text marker)
    ("100000012\t", "100000012"),          # tab
    ("12345678", "012345678"),             # 8 digits vs zero-padded 9
]

ORR_ROWS = [orr(o, first=f"Kid{i}") for i, (o, _) in enumerate(KEY_VARIANTS)] + [
    orr("200000001"),                                                       # VERIFIED
    orr("200000002", first="Jon", last="Doe"),                              # NEEDS_REVIEW (typo)
    orr("200000003", first="Ann", last="Lee", sex="F", dob="2009-01-01"),   # CONFLICT (names differ)
    orr("200000004", dob="2009-02-02"),                                     # CONFLICT (DOB differs)
    orr("200000005", dob="2011-05-05", aka="Correct dob: 01/02/2011"),      # NEEDS_REVIEW via notes
    orr("300000001", first="Ana", last="Rios"),                             # NO_MATCH
    orr("300000002", first="Maria", last="Garcia", sex="F", dob="2008-08-08"),  # POSSIBLE_KEY_ERROR
]
AFF_ROWS = [aff(a, first=f"Kid{i}") for i, (_, a) in enumerate(KEY_VARIANTS)] + [
    aff("200000001"),
    aff("200000002", first="John", last="Doe"),
    aff("200000003", first="Deanna", last="Liu", sex="F", dob="2009-01-01"),
    aff("200000004", dob="2009-12-25"),
    aff("200000005", dob="2011-01-02"),
    aff("555555555", first="Maria", last="Garcia", sex="F", dob="2008-08-08"),
]


@pytest.fixture(scope="session")
def spark():
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    s = (SparkSession.builder.master("local[2]").config("spark.sql.shuffle.partitions", "4")
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    yield s
    s.stop()


@pytest.fixture(scope="session")
def res(spark):
    return lean.run_dq_merge(spark.createDataFrame(ORR_ROWS, ORR_SCHEMA), spark.createDataFrame(AFF_ROWS, AFF_SCHEMA))


def by_key(res):
    return {r["uac_alien_no"]: r for r in res["merged"].collect()}


@pytest.mark.parametrize("orr_key, aff_key", KEY_VARIANTS)
def test_every_a_number_match_is_kept(res, orr_key, aff_key):
    r = by_key(res)[orr_key]
    assert r["asylum_case_exists"] is True, f"{orr_key!r} vs {aff_key!r}: {r['dq_all_notes']}"
    assert r["matched_a_number"] == aff_key


def test_matches_are_not_cleared_by_conflict_or_review(res):
    m = by_key(res)
    for key, status in [("200000001", "VERIFIED"), ("200000002", "NEEDS_REVIEW"),
                        ("200000003", "CONFLICT"), ("200000004", "CONFLICT"), ("200000005", "NEEDS_REVIEW")]:
        r = m[key]
        assert r["dq_status"] == status, (key, r["dq_all_notes"])
        assert r["asylum_case_exists"] is True and r["matched_a_number"] == key, key
        assert r["matched_receipt_number"] is not None and r["matched_case_id"] is not None, key


def test_unmatched_rows(res):
    m = by_key(res)
    assert m["300000001"]["dq_status"] == "NO_MATCH" and m["300000001"]["matched_a_number"] == "300000001"
    r = m["300000002"]
    assert r["dq_status"] == "POSSIBLE_KEY_ERROR" and r["matched_a_number"] == "555555555"
    assert r["asylum_case_exists"] is False


def test_notes_resolve_dob(res):
    r = by_key(res)["200000005"]
    assert r["dq_resolved_by_notes"] == ["dob"] and "DOB backed" in r["dq_all_notes"]


def test_row_count_and_checks(res):
    assert res["merged"].count() == len(ORR_ROWS)
    assert res["checks"]["ROW_COUNT_PRESERVED"] is True
    assert res["checks"]["detail"]["matched"] == len(KEY_VARIANTS) + 5


def test_numeric_a_number_column(spark):
    aff_df = spark.createDataFrame([aff("100000001")], AFF_SCHEMA).selectExpr(
        "cast(a_number as bigint) as a_number", *[c for c in AFF_SCHEMA.split(", ")[1:] for c in [c.split()[0]]])
    out = lean.run_dq_merge(spark.createDataFrame([orr("A100000001")], ORR_SCHEMA), aff_df)
    r = out["merged"].first()
    assert r["asylum_case_exists"] is True and r["matched_a_number"] == "100000001"


def test_exact_key_safety_net_passes(res):
    assert res["checks"]["EXACT_KEY_MATCHES_KEPT"] is True
    assert res["checks"]["detail"]["exact_key_matches_lost"] == 0
    assert res["checks"]["detail"]["exact_key_matches"] >= 6     # Rows whose raw key text is identical
    assert res["passed"] is True


def test_exact_key_safety_net_catches_a_lost_match(spark, monkeypatch):
    # Break the key cleaning on purpose: identical keys then no longer join, and the check must fail
    monkeypatch.setattr(lean, "_key", lambda c, dtype: lean.F.concat(c.cast("string"), lean.F.rand().cast("string")))
    out = lean.run_dq_merge(spark.createDataFrame([orr("100000001")], ORR_SCHEMA),
                            spark.createDataFrame([aff("100000001")], AFF_SCHEMA))
    assert out["checks"]["EXACT_KEY_MATCHES_KEPT"] is False and out["passed"] is False
    assert out["checks"]["detail"]["exact_key_matches_lost"] == 1
