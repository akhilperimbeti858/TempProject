"""Runs the dq_merge_job notebook locally, with a stand-in for Databricks' dbutils."""
import json
import os
import sys

import pytest
from pyspark.sql import SparkSession, functions as F

import dq_merge_lean as lean
from test_dq_merge import ASY_ROWS, ASY_SCHEMA, NOTES_ASY_ROWS, NOTES_ROWS, ORR_ROWS, ORR_SCHEMA

NOTEBOOK = os.path.join(os.path.dirname(__file__), "dq_merge_job.py")


class _Exit(Exception):
    pass


class FakeDbutils:
    """Just enough of dbutils for the notebook: widgets, library.restartPython, notebook.exit."""

    def __init__(self, params):
        self.params, self.exit_value = params, None
        outer = self

        class Widgets:
            def text(self, name, default, label=None):
                outer.params.setdefault(name, default)

            def dropdown(self, name, default, choices, label=None):
                outer.params.setdefault(name, default)

            def get(self, name):
                return outer.params[name]

        class Library:
            def restartPython(self):
                pass

        class Notebook:
            def exit(self, value):
                outer.exit_value = value
                raise _Exit()

        self.widgets, self.library, self.notebook = Widgets(), Library(), Notebook()


def run_notebook(spark, **params):
    """Execute every cell of the notebook source, as a job would. Returns the exit() JSON."""
    db = FakeDbutils({"install_rapidfuzz": "false", "output_format": "parquet", **params})
    saved = (lean.CACHE, lean.NOTES_COL)
    try:
        exec(compile(open(NOTEBOOK).read(), NOTEBOOK, "exec"), {"spark": spark, "dbutils": db})
    except _Exit:
        return json.loads(db.exit_value)
    finally:
        lean.CACHE, lean.NOTES_COL = saved  # The notebook sets these; keep other tests unaffected
    raise AssertionError("notebook did not call dbutils.notebook.exit")


def asylum_df(spark, rows):
    return (spark.createDataFrame(rows, ASY_SCHEMA)
            .withColumn("receipt_number", F.concat(F.lit("IOE"), "a_id"))
            .withColumn("case_id", F.concat(F.lit("C"), "a_id")))


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    s = (SparkSession.builder.master("local[2]").config("spark.sql.shuffle.partitions", "4")
         .config("spark.sql.warehouse.dir", str(tmp_path_factory.mktemp("wh"))).getOrCreate())
    orr = spark_orr = s.createDataFrame([r + (None,) for r in ORR_ROWS] + NOTES_ROWS,
                                        ORR_SCHEMA + ", additional_notes string")
    spark_orr.write.mode("overwrite").saveAsTable("job_orr")
    orr.withColumnRenamed("additional_notes", "Additional_Notes").write.mode("overwrite").saveAsTable("job_orr_cap")
    asylum_df(s, ASY_ROWS + NOTES_ASY_ROWS).write.mode("overwrite").saveAsTable("job_asylum")
    asylum_df(s, ASY_ROWS + NOTES_ASY_ROWS + [("111111111", "X", "Y", "2001-01-01", "M", "Mexico", None)]) \
        .write.mode("overwrite").saveAsTable("job_asylum_dup")
    return s


def test_job_writes_output_and_summary(spark):
    out = run_notebook(spark, orr_table="job_orr", asylum_table="job_asylum", output_table="job_out",
                       run_id="12345", run_date="2026-11-04")
    assert out["passed"] is True and out["run_id"] == "12345"
    assert out["status_counts"]["POSSIBLE_KEY_ERROR"] >= 1
    t = spark.table("job_out")
    assert t.count() == len(ORR_ROWS) + len(NOTES_ROWS)
    r = t.where("a_id = '444444444'").first()
    assert r["matched_a_number"] == "555555555" and r["dq_run_id"] == "12345"
    assert str(r["dq_run_date"]) == "2026-11-04" and r["dq_all_notes"].startswith("STATUS POSSIBLE_KEY_ERROR")


def test_job_filter_runs_a_subset(spark):
    out = run_notebook(spark, orr_table="job_orr", asylum_table="job_asylum", output_table="job_out_subset",
                       orr_filter="a_id LIKE '7%'")
    assert out["checks"]["detail"]["orr_rows"] == len(NOTES_ROWS)


def test_job_fails_on_gate_and_quarantines(spark):
    with pytest.raises(RuntimeError, match="ASYLUM_KEYS_UNIQUE"):
        run_notebook(spark, orr_table="job_orr", asylum_table="job_asylum_dup", output_table="job_out_bad",
                     quarantine_table="job_quarantine")
    assert not spark.catalog.tableExists("job_out_bad")          # A bad batch never reaches the output
    assert spark.table("job_quarantine").count() > 0


def test_job_rejects_missing_parameters(spark):
    with pytest.raises(ValueError, match="output_table"):
        run_notebook(spark, orr_table="job_orr", asylum_table="job_asylum", output_table="")


def test_job_notes_column_parameter(spark):
    out = run_notebook(spark, orr_table="job_orr_cap", asylum_table="job_asylum", output_table="job_out_cap",
                       notes_column="Additional_Notes")
    assert out["passed"] is True
    r = spark.table("job_out_cap").where("a_id = '700000001'").first()
    assert r["dq_resolved_by_notes"] == ["dob"]                   # Notes read under the custom column name
