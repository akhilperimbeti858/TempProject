import os
import sys

import pytest
from pyspark.sql import SparkSession, functions as F

import dq_merge as full
import dq_merge_lean as lean
from test_dq_merge import ASY_ROWS, ASY_SCHEMA, NOTES_ASY_ROWS, NOTES_ROWS, ORR_ROWS, ORR_SCHEMA

NOTES_SCHEMA = ORR_SCHEMA + ", additional_notes string"

# Several possible keys for one ORR record, and source priority
MULTI_ORR = [
    # Same name + DOB under two other a_ids -> lowest a_id, flagged as multiple
    ("900000001", "Lucia", "Flores", "2011-11-11", "F", "Honduras", None),
    # 'fields' hit (900000020) beats a 'notes_name' hit with a lower a_id (900000011)
    ("900000010", "Rafael", "Soto", "2010-05-05", "M", "Mexico", "aka Miguel Angel Rios"),
    # Typo at the start of the last name in the note (KRUZ vs CRUZ): found via the first-name block
    # and the full-name score, as in dq_merge.py
    ("910000001", "Mary", "Smithers", "2012-12-12", "F", "Mexico", "aka Maria Kruz"),
]
MULTI_ASY = [
    ("900000003", "Lucia", "Flores", "2011-11-11", "F", "Honduras", None),
    ("900000002", "Lucia", "Flores", "2011-11-11", "F", "Honduras", None),
    ("900000020", "Rafael", "Soto", "2010-05-05", "M", "Mexico", None),
    ("900000011", "Miguel Angel", "Rios", "2010-05-05", "M", "Mexico", None),
    ("910000002", "Maria", "Cruz", "2012-12-12", "F", "Mexico", None),
]


@pytest.fixture(scope="session")
def spark():
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    return (SparkSession.builder.master("local[2]")
            .config("spark.sql.shuffle.partitions", "4").getOrCreate())


def asylum(spark, rows):
    """Asylum fixture rows plus receipt_number / case_id derived from the a_id."""
    return (spark.createDataFrame(rows, ASY_SCHEMA)
            .withColumn("receipt_number", F.concat(F.lit("IOE"), "a_id"))
            .withColumn("case_id", F.concat(F.lit("C"), "a_id")))


@pytest.fixture(scope="session")
def data(spark):
    base = (spark.createDataFrame(ORR_ROWS, ORR_SCHEMA), asylum(spark, ASY_ROWS))
    notes = (spark.createDataFrame([r + (None,) for r in ORR_ROWS] + NOTES_ROWS + MULTI_ORR, NOTES_SCHEMA),
             asylum(spark, ASY_ROWS + NOTES_ASY_ROWS + MULTI_ASY))
    return {"base": base, "notes": notes}


@pytest.fixture(scope="session")
def lean_res(data):
    return {k: lean.run_dq_merge(*v) for k, v in data.items()}


def by_key(df):
    return {r["a_id"]: r for r in df.collect()}


@pytest.mark.parametrize("case", ["base", "notes"])
def test_same_results_as_full_script(data, lean_res, case):
    expected = by_key(full.run_dq_merge(*data[case])["merged"])
    got = by_key(lean_res[case]["merged"])
    assert got.keys() == expected.keys()
    for k, e in expected.items():
        for col in ("dq_status", "dq_match_score", "dq_resolved_by_notes", "dq_possible_key_source",
                    "dq_possible_asylum_keys",
                    *lean.STATUS.values(), *lean.SIM_COLS.values()):
            assert got[k][col] == e[col], (k, col, got[k][col], e[col])


def test_matched_a_number(lean_res):
    m = by_key(lean_res["notes"]["merged"])
    assert m["A012345678"]["matched_a_number"] == "12345678"      # Matched: asylum's own format
    assert m["111111111"]["matched_a_number"] == "111111111"      # CONFLICT keeps its own case
    assert m["444444444"]["dq_status"] == "POSSIBLE_KEY_ERROR"
    assert m["444444444"]["matched_a_number"] == "555555555"
    assert m["700000005"]["matched_a_number"] == "800000005"      # Found through the notes
    assert m["333333333"]["matched_a_number"] == "333333333"      # NO_MATCH keeps the ORR a_id


def test_multiple_possible_keys(lean_res):
    m = by_key(lean_res["notes"]["merged"])
    r = m["900000001"]
    assert r["dq_possible_asylum_keys"] == ["900000002", "900000003"]
    assert r["matched_a_number"] == "900000002" and r["dq_multiple_possible_keys"] is True
    r = m["900000010"]
    assert r["dq_possible_key_source"] == ["fields", "notes_name"]
    assert r["matched_a_number"] == "900000020"                   # fields beats notes_name
    assert m["444444444"]["dq_multiple_possible_keys"] is False


def test_passed(spark, lean_res):
    assert lean_res["base"]["passed"] is True
    assert lean_res["base"]["checks"]["detail"]["orr_rows"] == len(ORR_ROWS)
    dup = asylum(spark, ASY_ROWS + [("111111111", "X", "Y", "2001-01-01", "M", "Mexico", None)])
    res = lean.run_dq_merge(spark.createDataFrame(ORR_ROWS, ORR_SCHEMA), dup)
    assert res["passed"] is False and res["checks"]["ASYLUM_KEYS_UNIQUE"] is False


def test_row_issues(lean_res):
    m = by_key(lean_res["notes"]["merged"])
    assert "MEDIUM:NOTES_DOB_DIFFERS" in m["700000001"]["dq_row_issues"]
    assert m["A012345678"]["dq_row_issues"] is None


def test_receipt_and_case_id(lean_res):
    m = by_key(lean_res["notes"]["merged"])
    r = m["444444444"]                                            # POSSIBLE_KEY_ERROR, one possible case
    assert r["dq_possible_receipt_numbers"] == ["IOE555555555"] and r["dq_possible_case_ids"] == ["C555555555"]
    assert (r["matched_receipt_number"], r["matched_case_id"]) == ("IOE555555555", "C555555555")
    r = m["900000001"]                                            # Two possible cases: arrays line up by position
    assert r["dq_possible_asylum_keys"] == ["900000002", "900000003"]
    assert r["dq_possible_receipt_numbers"] == ["IOE900000002", "IOE900000003"]
    assert r["dq_possible_case_ids"] == ["C900000002", "C900000003"]
    assert (r["matched_receipt_number"], r["matched_case_id"]) == ("IOE900000002", "C900000002")
    r = m["900000010"]                                            # Chosen by source, not lowest a_id
    assert (r["matched_a_number"], r["matched_case_id"]) == ("900000020", "C900000020")
    r = m["A012345678"]                                           # Matched: its own case
    assert r["matched_receipt_number"] == r["asy_receipt_number"] == "IOE12345678"
    r = m["333333333"]                                            # NO_MATCH
    assert r["matched_receipt_number"] is None and r["matched_case_id"] is None


def test_missing_receipt_column(spark, data):
    orr, asy = data["base"]
    with pytest.raises(ValueError, match="receipt_number"):
        lean.run_dq_merge(orr, asy.drop("receipt_number"))


def notes_of(lean_res, a_id):
    return by_key(lean_res["notes"]["merged"])[a_id]["dq_all_notes"]


def test_all_notes_conflict_reasons(lean_res):
    n = notes_of(lean_res, "700000004")
    assert n.startswith("STATUS CONFLICT: DOB disagrees outright (ORR 2009-02-02 vs asylum 2009-12-25)")
    assert "no date in the notes equals the asylum DOB 2009-12-25" in n
    n = notes_of(lean_res, "111111111")
    assert "first and last name both disagree" in n and "below the review threshold" in n
    assert "FIRST MISMATCH: ORR 'Ann' vs asylum 'Deanna'" in n


def test_all_notes_notes_evidence(lean_res):
    n = notes_of(lean_res, "700000001")
    assert "resolved by the notes: dob" in n and "DOB backed: the notes contain the asylum DOB 2011-01-02" in n
    assert "ISSUES MEDIUM:NOTES_DOB_DIFFERS" in n
    n = notes_of(lean_res, "700000003")
    assert "best name version 'PEREZ, JUAN' read as first=JUAN last=PEREZ [LAST, FIRST]" in n
    assert "-> backs first, last" in n
    n = notes_of(lean_res, "700000010")                       # Two versions: only JUAN LOPEZ is used
    assert "best name version 'JUAN LOPEZ'" in n and "-> backs first" in n


def test_all_notes_other_keys(lean_res):
    n = notes_of(lean_res, "444444444")
    assert "STATUS POSSIBLE_KEY_ERROR" in n and "CHOSEN 555555555 (receipt IOE555555555, case C555555555)" in n
    assert "555555555 via fields: same name + DOB 2008-08-08" in n
    n = notes_of(lean_res, "700000005")
    assert "800000005 via notes_name: note name 'JUAN CARLOS PEREZ'" in n and "DOB 2013-06-06" in n
    n = notes_of(lean_res, "910000001")                       # KRUZ in the note, CRUZ in the asylum table
    assert "CHOSEN 910000002" in n and "via notes_name: note name 'MARIA KRUZ'" in n
    assert "MULTIPLE 2 possible cases" in notes_of(lean_res, "900000001")


def test_all_notes_simple_rows(lean_res):
    assert notes_of(lean_res, "333333333").startswith("STATUS NO_MATCH: a_id 333333333 is not in the asylum table")
    n = notes_of(lean_res, "A012345678")
    assert n.startswith("STATUS VERIFIED: all fields agree") and "formats differ" in n
