import os
import sys

import pytest
from pyspark.sql import SparkSession

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
]
MULTI_ASY = [
    ("900000003", "Lucia", "Flores", "2011-11-11", "F", "Honduras", None),
    ("900000002", "Lucia", "Flores", "2011-11-11", "F", "Honduras", None),
    ("900000020", "Rafael", "Soto", "2010-05-05", "M", "Mexico", None),
    ("900000011", "Miguel Angel", "Rios", "2010-05-05", "M", "Mexico", None),
]


@pytest.fixture(scope="session")
def spark():
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    return (SparkSession.builder.master("local[2]")
            .config("spark.sql.shuffle.partitions", "4").getOrCreate())


@pytest.fixture(scope="session")
def data(spark):
    base = (spark.createDataFrame(ORR_ROWS, ORR_SCHEMA), spark.createDataFrame(ASY_ROWS, ASY_SCHEMA))
    notes = (spark.createDataFrame([r + (None,) for r in ORR_ROWS] + NOTES_ROWS + MULTI_ORR, NOTES_SCHEMA),
             spark.createDataFrame(ASY_ROWS + NOTES_ASY_ROWS + MULTI_ASY, ASY_SCHEMA))
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
    dup = spark.createDataFrame(ASY_ROWS + [("111111111", "X", "Y", "2001-01-01", "M", "Mexico", None)], ASY_SCHEMA)
    res = lean.run_dq_merge(spark.createDataFrame(ORR_ROWS, ORR_SCHEMA), dup)
    assert res["passed"] is False and res["checks"]["ASYLUM_KEYS_UNIQUE"] is False


def test_row_issues(lean_res):
    m = by_key(lean_res["notes"]["merged"])
    assert "MEDIUM:NOTES_DOB_DIFFERS" in m["700000001"]["dq_row_issues"]
    assert m["A012345678"]["dq_row_issues"] is None
