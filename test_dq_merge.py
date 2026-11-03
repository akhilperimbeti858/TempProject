import pytest
from pyspark.sql import SparkSession, functions as F

import dq_merge as dq

ORR_SCHEMA = "a_id string, first_name string, last_name string, dob string, sex string, country_of_birth string"
ASY_SCHEMA = ORR_SCHEMA + ", middle_name string"

ORR_ROWS = [
    # VERIFIED: 'A012345678' vs '12345678'; middle name typed into last name; different DOB/sex formats
    ("A012345678", "John", "James Smith", "2010-03-05", "male", "Honduras"),
    # CONFLICT: same a_id, different person (ANN/LEE vs DEANNA/LIU)
    ("111111111", "Ann", "Lee", "2009-01-01", "F", "Guatemala"),
    # NEEDS_REVIEW: typo in first name
    ("222222222", "Jon", "Doe", "2011-07-07", "M", "El Salvador"),
    # NO_MATCH
    ("333333333", "Ana", "Perez", "2012-02-02", "F", "Mexico"),
    # POSSIBLE_KEY_ERROR: same person exists under 555555555
    ("444444444", "María", "García", "2008-08-08", "F", "Honduras"),
    # NEEDS_REVIEW: first and last swapped
    ("666666666", "Smith", "Robert", "2007-06-06", "M", "Guatemala"),
]
ASY_ROWS = [
    ("12345678", "JOHN", "SMITH", "3/5/2010", "M", "HONDURAS", "James"),
    ("111111111", "Deanna", "Liu", "2009-01-01", "F", "Guatemala", None),
    ("222222222", "John", "Doe", "2011-07-07", "M", "El Salvador", "NMN"),
    ("555555555", "Maria", "Garcia", "2008-08-08", "F", "Honduras", None),
    ("666666666", "Robert", "Smith", "2007-06-06", "M", "Guatemala", None),
    ("888888888", "Other", "Person", "2000-01-01", "M", "Mexico", None),
]


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    s = (SparkSession.builder.master("local[2]")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.sql.warehouse.dir", str(tmp_path_factory.mktemp("wh")))
         .getOrCreate())
    yield s
    s.stop()


@pytest.fixture(scope="session")
def res(spark):
    return dq.run_dq_merge(spark.createDataFrame(ORR_ROWS, ORR_SCHEMA),
                           spark.createDataFrame(ASY_ROWS, ASY_SCHEMA))


def by_key(res):
    return {r["a_id"]: r for r in res["merged"].collect()}


def test_statuses(res):
    m = by_key(res)
    assert m["A012345678"]["dq_status"] == "VERIFIED"
    assert m["A012345678"]["dq_last_name_status"] == "MIDDLE_IN_LAST"
    assert m["111111111"]["dq_status"] == "CONFLICT"
    assert m["222222222"]["dq_status"] == "NEEDS_REVIEW"
    assert m["222222222"]["dq_first_name_status"] == "FUZZY"
    assert m["333333333"]["dq_status"] == "NO_MATCH"
    assert m["444444444"]["dq_status"] == "POSSIBLE_KEY_ERROR"
    assert m["444444444"][dq.OTHER_KEYS_COL] == ["555555555"]
    assert m["666666666"]["dq_status"] == "NEEDS_REVIEW"
    assert m["666666666"]["dq_first_name_status"] == "SWAPPED"
    assert m["666666666"]["dq_full_name_status"] == "REORDERED"


def test_match_flag_and_row_count(res):
    m = by_key(res)
    assert len(m) == len(ORR_ROWS)
    assert m["A012345678"][dq.MATCH_FLAG_COL] is True
    assert m["333333333"][dq.MATCH_FLAG_COL] is False
    assert m["A012345678"]["asy_a_id"] == "12345678"


def test_loose_names_are_mismatches(res):
    r = by_key(res)["111111111"]
    assert r["dq_first_name_status"] == "MISMATCH"   # ANN vs DEANNA (was PARTIAL)
    assert r["dq_last_name_status"] == "MISMATCH"    # LEE vs LIU (was FUZZY)


def test_gate_small_batch_does_not_block(res):
    # 1 conflict of 5 matches = 20%, but below MIN_MATCHES_FOR_RATE, so HIGH only
    g = {r["check"]: r for r in res["gate"].collect()}
    assert g["CONFLICT_RATE"]["severity"] == "HIGH" and not g["CONFLICT_RATE"]["passed"]
    assert res["passed"]


def test_duplicate_asylum_key_blocks_publish(spark):
    asy = spark.createDataFrame(ASY_ROWS + [("111111111", "X", "Y", "2001-01-01", "M", "Mexico", None)], ASY_SCHEMA)
    r = dq.run_dq_merge(spark.createDataFrame(ORR_ROWS, ORR_SCHEMA), asy)
    assert not r["passed"]
    with pytest.raises(RuntimeError, match="ASYLUM_KEYS_UNIQUE"):
        dq.publish(r, "should_not_exist")
    assert not spark.catalog.tableExists("should_not_exist")


def test_publish(spark, res):
    dq.publish(res, "orr_asylum_merged")
    assert spark.table("orr_asylum_merged").count() == len(ORR_ROWS)


def test_row_ids_are_deterministic(spark):
    df = spark.createDataFrame(ORR_ROWS + ORR_ROWS[:1], ORR_SCHEMA).repartition(3)  # includes an exact duplicate
    ids = lambda: sorted(r[0] for r in dq.normalize(df, dq.ORR_COLS, "o__").select("o__row_id").collect())
    first = ids()
    assert first == ids() and len(set(first)) == len(ORR_ROWS) + 1


def test_issues_reported(res):
    checks = {(r["table"], r["check"]) for r in res["issues"].collect()}
    assert (dq.ORR, "SAME_PERSON_MULTIPLE_KEYS") not in checks
    assert (dq.ORR, "KEY_FORMAT") not in checks  # 'A012345678' is a valid raw shape


def test_column_clash_rejected(spark):
    orr = spark.createDataFrame(ORR_ROWS, ORR_SCHEMA).withColumn("dq_status", F.lit("x"))
    with pytest.raises(ValueError, match="collide"):
        dq.run_dq_merge(orr, spark.createDataFrame(ASY_ROWS, ASY_SCHEMA))
