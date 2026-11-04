# Databricks notebook source
# MAGIC %md
# MAGIC # HHS-ORR to affirmative asylum DQ merge (scheduled job)
# MAGIC
# MAGIC Runs `dq_merge_lean.run_dq_merge` on the ORR and asylum tables and writes `merged` to a Delta table.
# MAGIC
# MAGIC **Flow:** read the job parameters, read the two tables (optionally filtered to a subset), run the merge,
# MAGIC log the checks, then:
# MAGIC * **checks passed:** write the output table and end with a JSON summary
# MAGIC * **checks failed:** optionally write the batch to a quarantine table, then **fail the run**, so job alerts and
# MAGIC   retries fire and a bad batch never replaces the output table
# MAGIC
# MAGIC **Job setup**
# MAGIC * Source: this repo (Git source or Repos). Keep this notebook next to `dq_merge_lean.py`.
# MAGIC * Library: add `rapidfuzz` to the job cluster's libraries (preferred), or leave `install_rapidfuzz = true`.
# MAGIC * Parameters: see the widgets below. Pass `run_id` as `{{job.run_id}}` so each output row records its run.
# MAGIC * Serverless compute: set `cache = false`.

# COMMAND ----------

# MAGIC %md ## 1. Parameters

# COMMAND ----------

dbutils.widgets.text("orr_table", "", "ORR table (catalog.schema.table)")
dbutils.widgets.text("asylum_table", "", "Asylum table (catalog.schema.table)")
dbutils.widgets.text("output_table", "", "Output table for merged")
dbutils.widgets.text("quarantine_table", "", "Table for a failed batch (optional)")
dbutils.widgets.text("orr_filter", "", "SQL filter on ORR, for a subset (optional)")
dbutils.widgets.text("asylum_filter", "", "SQL filter on asylum, for a subset (optional)")
dbutils.widgets.text("notes_column", "additional_notes", "ORR free-text notes column")
dbutils.widgets.dropdown("write_mode", "overwrite", ["overwrite", "replace_run_date"], "Write mode")
dbutils.widgets.text("run_date", "", "Run date yyyy-MM-dd (default: today)")
dbutils.widgets.text("run_id", "", "Job run id ({{job.run_id}})")
dbutils.widgets.dropdown("fail_on_gate", "true", ["true", "false"], "Fail the run when checks fail")
dbutils.widgets.dropdown("cache", "true", ["true", "false"], "Cache intermediates (false on serverless)")
dbutils.widgets.dropdown("install_rapidfuzz", "true", ["true", "false"], "pip install rapidfuzz in this run")
dbutils.widgets.text("output_format", "delta", "Output format")

# COMMAND ----------

# MAGIC %md ## 2. Dependencies
# MAGIC Installs rapidfuzz for this run only when it is not already on the cluster. Restarting Python clears
# MAGIC variables but keeps widget values, so parameters are read again in the next cell.

# COMMAND ----------

import importlib.util

if dbutils.widgets.get("install_rapidfuzz") == "true" and importlib.util.find_spec("rapidfuzz") is None:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "rapidfuzz>=3.0"])
    dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## 3. Run

# COMMAND ----------

import datetime
import json
import logging

from pyspark.sql import functions as F

import dq_merge_lean as lean

log = logging.getLogger("dq_merge_job")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)


def get_params():
    """Job parameters, validated. Raises ValueError so a misconfigured job fails immediately."""
    p = {k: dbutils.widgets.get(k).strip() for k in [
        "orr_table", "asylum_table", "output_table", "quarantine_table", "orr_filter", "asylum_filter",
        "notes_column", "write_mode", "run_date", "run_id", "fail_on_gate", "cache", "output_format"]}
    missing = [k for k in ("orr_table", "asylum_table", "output_table") if not p[k]]
    if missing:
        raise ValueError(f"Missing required job parameters: {missing}")
    p["run_date"] = p["run_date"] or datetime.date.today().isoformat()
    datetime.date.fromisoformat(p["run_date"])  # Fails fast on a malformed date
    p["fail_on_gate"] = p["fail_on_gate"] == "true"
    p["cache"] = p["cache"] == "true"
    return p


def read_input(table, where, label):
    df = spark.table(table)
    if where:
        df = df.where(where)
    log.info("%s: %s%s", label, table, f" WHERE {where}" if where else "")
    return df


def with_run_columns(df, p):
    """Who/when columns, so every output row can be traced to its run."""
    return (df.withColumn("dq_run_date", F.lit(p["run_date"]).cast("date"))
              .withColumn("dq_run_ts", F.current_timestamp())
              .withColumn("dq_run_id", F.lit(p["run_id"] or None).cast("string")))


def write(df, table, p):
    """Re-runnable write.

    overwrite         replace the whole table (schema changes allowed)
    replace_run_date  replace only this run_date's rows; a re-run of the same date replaces, never duplicates
    """
    w = df.write.format(p["output_format"])
    if p["write_mode"] == "replace_run_date":
        w = (w.mode("overwrite").option("replaceWhere", f"dq_run_date = '{p['run_date']}'")
              .option("mergeSchema", "true").partitionBy("dq_run_date"))
    else:
        w = w.mode("overwrite").option("overwriteSchema", "true")
    w.saveAsTable(table)
    log.info("Wrote %s (%s)", table, p["write_mode"])


def main():
    p = get_params()
    lean.CACHE = p["cache"]
    lean.NOTES_COL = p["notes_column"]

    orr = read_input(p["orr_table"], p["orr_filter"], "ORR")
    asylum = read_input(p["asylum_table"], p["asylum_filter"], "Asylum")

    # Notes are skipped silently when the column name does not match exactly, so say so loudly
    if lean.NOTES_COL in orr.columns:
        log.info("Notes column: %s", lean.NOTES_COL)
    else:
        log.warning("Notes column '%s' NOT FOUND: notes evidence is disabled for this run. ORR columns "
                    "containing 'note': %s", lean.NOTES_COL, [c for c in orr.columns if "note" in c.lower()])

    res = lean.run_dq_merge(orr, asylum)
    merged, checks = res["merged"], res["checks"]
    try:
        detail = checks["detail"]
        log.info("Checks: %s", json.dumps({k: v for k, v in checks.items() if k != "detail"}))
        log.info("Detail: %s", json.dumps(detail))
        if detail["orr_rows"] == 0:
            raise RuntimeError("ORR input is empty: check orr_table / orr_filter")

        status_counts = {r["dq_status"]: r["count"] for r in merged.groupBy("dq_status").count().collect()}
        log.info("Status counts: %s", json.dumps(status_counts))
        summary = {"passed": res["passed"], "run_date": p["run_date"], "run_id": p["run_id"],
                   "output_table": p["output_table"], "checks": checks, "status_counts": status_counts}
        out = with_run_columns(merged, p)

        if not res["passed"]:
            failed = [k for k, v in checks.items() if k != "detail" and not v]
            if p["quarantine_table"]:
                write(out, p["quarantine_table"], p)
                summary["quarantine_table"] = p["quarantine_table"]
            if p["fail_on_gate"]:
                raise RuntimeError(f"DQ checks failed: {failed}; output table not written. Detail: {json.dumps(detail)}")
            log.warning("DQ checks failed (%s) but fail_on_gate=false: writing anyway", failed)

        write(out, p["output_table"], p)
        return summary
    finally:
        for df in (merged,):
            try:
                df.unpersist()
            except Exception:
                pass


summary = main()

# COMMAND ----------

# MAGIC %md ## 4. Result
# MAGIC The JSON summary is the task's output, visible in the run page and readable by downstream tasks.

# COMMAND ----------

dbutils.notebook.exit(json.dumps(summary, default=str))
