import os
import sys

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
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)  # Python UDF workers need rapidfuzz/pandas too
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


# ---------------------------------------------------------------------------
# additional_notes
# ---------------------------------------------------------------------------
NOTES_ROWS = [
    # Wrong dob column, correct DOB labelled in the notes (M/D order)
    ("700000001", "Carlos", "Ruiz", "2011-05-05", "M", "Honduras", "Correct dob: 01/02/2011"),
    # Bare-ish date in a sentence, D/M order
    ("700000002", "Luis", "Mora", "2010-09-09", "M", "Honduras", "child states born 13/04/2010"),
    # Wrong name columns, right name in the notes as LAST, FIRST
    ("700000003", "Pedro", "Lopez", "2012-03-03", "M", "Guatemala", "correct name PEREZ, JUAN"),
    # Unrelated date in the notes: must stay CONFLICT
    ("700000004", "Rosa", "Diaz", "2009-02-02", "F", "Mexico", "arrived 03/04/2022"),
    # NO_MATCH, alias in the notes + ORR DOB matches a case under another a_id
    ("700000005", "Jhon", "Pirez", "2013-06-06", "M", "Honduras", "aka Juan Carlos Pérez"),
    # NO_MATCH, ORR name + DOB from the notes matches a case under another a_id
    ("700000006", "Elena", "Vega", "2014-01-01", "F", "Guatemala", "DOB 2014-07-15 per sponsor docs"),
    # Filler: CORRECT must not count as the asylum last name CORREA
    ("700000007", "Ana", "Mendez", "2011-03-03", "F", "Honduras", "as per nta: correct dob 2011-01-02"),
    # Filler around a real name still resolves it
    ("700000008", "Jose", "Ramos", "2012-08-08", "M", "Honduras", "as per NTA: PEREZ, JUAN"),
    # Name particles: asylum DE LA CRUZ matches on CRUZ
    ("700000009", "Maria", "Santos", "2010-10-10", "F", "Mexico", "aka Maria de la Cruz"),
    # Two name versions: words must not be mixed across them (JUAN from one, PEREZ from the other)
    ("700000010", "Jose", "Ramos", "2012-09-09", "M", "Honduras", "aka Juan Lopez / also Pedro Perez"),
    # Two versions of the same name: the LAST, FIRST one is the best match
    ("700000011", "Jose", "Ramos", "2012-10-10", "M", "Honduras", "aka Jon Peres; correct name PEREZ, JUAN"),
]
NOTES_ASY_ROWS = [
    ("700000001", "Carlos", "Ruiz", "2011-01-02", "M", "Honduras", None),
    ("700000002", "Luis", "Mora", "2010-04-13", "M", "Honduras", None),
    ("700000003", "Juan", "Perez", "2012-03-03", "M", "Guatemala", None),
    ("700000004", "Rosa", "Diaz", "2009-12-25", "F", "Mexico", None),
    ("800000005", "Juan Carlos", "Perez", "2013-06-06", "M", "Honduras", None),
    ("800000006", "Elena", "Vega", "2014-07-15", "F", "Guatemala", None),
    ("700000007", "Ana", "Correa", "2011-01-02", "F", "Honduras", None),
    ("700000008", "Juan", "Perez", "2012-08-08", "M", "Honduras", None),
    ("700000009", "Maria", "De La Cruz", "2010-10-10", "F", "Mexico", None),
    ("700000010", "Juan", "Perez", "2012-09-09", "M", "Honduras", None),
    ("700000011", "Juan", "Perez", "2012-10-10", "M", "Honduras", None),
]


@pytest.fixture(scope="session")
def notes_res(spark):
    orr = spark.createDataFrame([r + (None,) for r in ORR_ROWS] + NOTES_ROWS,
                                ORR_SCHEMA + ", additional_notes string")
    asy = spark.createDataFrame(ASY_ROWS + NOTES_ASY_ROWS, ASY_SCHEMA)
    return dq.run_dq_merge(orr, asy)


def test_notes_resolve_dob(notes_res):
    m = by_key(notes_res)
    for k in ("700000001", "700000002"):
        assert m[k]["dq_status"] == "NEEDS_REVIEW", k
        assert m[k][dq.NOTES_RESOLVED_COL] == ["dob"], k
        assert m[k]["dq_dob_status"] == "MISMATCH", k  # The ORR column itself is still reported as wrong


def test_notes_resolve_name(notes_res):
    r = by_key(notes_res)["700000003"]
    assert r["dq_status"] == "NEEDS_REVIEW"
    assert r[dq.NOTES_RESOLVED_COL] == ["first", "last"]


def test_unrelated_note_date_stays_conflict(notes_res):
    r = by_key(notes_res)["700000004"]
    assert r["dq_status"] == "CONFLICT" and r[dq.NOTES_RESOLVED_COL] == []


def test_notes_find_other_keys(notes_res):
    m = by_key(notes_res)
    assert m["700000005"]["dq_status"] == "POSSIBLE_KEY_ERROR"
    assert m["700000005"][dq.OTHER_KEYS_COL] == ["800000005"]
    assert m["700000005"][dq.OTHER_KEYS_SOURCE_COL] == ["notes_name"]
    assert m["700000006"]["dq_status"] == "POSSIBLE_KEY_ERROR"
    assert m["700000006"][dq.OTHER_KEYS_SOURCE_COL] == ["notes_dob"]
    assert m["444444444"][dq.OTHER_KEYS_SOURCE_COL] == ["fields"]


def test_rows_without_notes_unchanged(res, notes_res):
    before, after = by_key(res), by_key(notes_res)
    for k in before:
        assert after[k]["dq_status"] == before[k]["dq_status"], k


def test_notes_dob_differs_check(notes_res):
    flagged = {r["a_id_raw"] for r in notes_res["issues"].where("check = 'NOTES_DOB_DIFFERS'").collect()}
    assert flagged == {"700000001", "700000002", "700000006", "700000007"}


def test_filler_words_do_not_match_names(notes_res):
    m = by_key(notes_res)
    r = m["700000007"]  # Only the DOB is backed by the note; the last name still conflicts
    assert r[dq.NOTES_RESOLVED_COL] == ["dob"] and r["dq_status"] == "CONFLICT"
    r = m["700000008"]
    assert r[dq.NOTES_RESOLVED_COL] == ["first", "last"] and r["dq_status"] == "NEEDS_REVIEW"
    r = m["700000009"]
    assert r[dq.NOTES_RESOLVED_COL] == ["last"] and r["dq_status"] == "NEEDS_REVIEW"


def test_note_name_versions_are_not_mixed(notes_res):
    r = by_key(notes_res)["700000010"]
    assert r[dq.NOTES_RESOLVED_COL] == ["first"]          # JUAN LOPEZ is the best version; PEREZ is not borrowed
    assert r[dq.NOTE_BEST_COL].startswith("JUAN LOPEZ")


def test_best_of_several_note_versions(notes_res):
    r = by_key(notes_res)["700000011"]
    assert r[dq.NOTES_RESOLVED_COL] == ["first", "last"]
    assert r[dq.NOTE_BEST_COL] == "PEREZ, JUAN -> first=JUAN last=PEREZ [LAST, FIRST] 100/100/100"


def test_notes_extract_report(notes_res):
    rep = {r["a_id"]: r for r in notes_res["notes_extract"].collect()}
    assert "333333333" not in rep                           # Only rows with notes are reported
    r = rep["700000003"]
    cands = {(c["first"], c["last"], c["layout"]) for c in r["note_name_candidates"]}
    assert ("JUAN", "PEREZ", "LAST, FIRST") in cands
    assert r["best_note_name"]["layout"] == "LAST, FIRST"
    assert r["best_note_name"]["first_score"] == 100.0
    assert [str(d) for d in rep["700000001"]["note_dates"]] == ["2011-01-02", "2011-02-01"]  # Both day orders


def test_similarity_columns(res):
    m = by_key(res)
    assert m["222222222"][dq.SIM_COLS["first"]] == pytest.approx(93.3, abs=0.1)   # JON vs JOHN
    assert m["A012345678"][dq.SIM_COLS["full"]] == 100.0
    assert m["333333333"][dq.SIM_COLS["first"]] is None                           # NO_MATCH


def test_no_notes_column_has_no_report(res):
    assert res["notes_extract"] is None
