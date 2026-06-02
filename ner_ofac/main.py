"""CLI entry point for the NER OFAC matching and training pipeline.

Usage
-----
::

    python -m ner_ofac.main --mode analyze
    python -m ner_ofac.main --mode train
    python -m ner_ofac.main --mode feedback
    python -m ner_ofac.main --mode both

Modes
-----
train
    Prepare annotation data from ``confirmed_entities.csv`` and trigger a new
    Comprehend Flywheel training iteration.

analyze
    Load sentences → preprocess → run Comprehend NER → OFAC cascading match →
    context-expand REVIEW-tier results → write output artefacts.

feedback
    Convert the latest analysis results into new training annotations and
    trigger a Flywheel iteration.

both
    Run ``analyze`` followed immediately by ``feedback``.
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------


def _run_analyze():
    """Execute the full analysis pipeline and return results.

    Returns:
        A list of result dicts suitable for passing to ``_run_feedback``.
    """
    from pyspark.sql import SparkSession
    from sentence_transformers import SentenceTransformer

    from ner_ofac import config
    from ner_ofac.ingest import load_sentences
    from ner_ofac.preprocess import preprocess
    from ner_ofac.ofac_loader import load_ofac
    from ner_ofac.analysis.run_inference import run_realtime
    from ner_ofac.analysis.ofac_matcher import match_entity
    from ner_ofac.analysis.faiss_index import build_index
    from ner_ofac.analysis.context_expander import recheck_review
    from ner_ofac.analysis.result_writer import (
        write_ner_results,
        write_confirmed,
        write_review_queue,
    )

    spark = SparkSession.builder.appName("ner_ofac_analyze").getOrCreate()
    logger.info("Spark session started.")

    # 1. Ingest + preprocess.
    raw_df = load_sentences(spark)
    clean_df = preprocess(raw_df)

    # 2. Load OFAC name lists and build per-category FAISS indexes.
    ofac_names = load_ofac(config.OFAC_CSV_PATH)
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    indexes = {}
    for entity_type, names in ofac_names.items():
        if names:
            idx, nl = build_index(names, model)
            indexes[entity_type] = (idx, nl)
        else:
            logger.warning("No OFAC names for entity type %s; skipping index.", entity_type)

    # 3. Run Comprehend inference.
    inference_results = run_realtime(clean_df, config.COMPREHEND_ENDPOINT_ARN)

    # 4. OFAC cascading match for each detected entity.
    all_results = []
    for inf in inference_results:
        row_index = inf["row_index"]
        text = inf["text"]
        for ent in inf["entities"]:
            entity_text = ent["Text"]
            entity_type = ent["Type"]
            comprehend_score = ent["Score"]
            begin_offset = ent["BeginOffset"]
            end_offset = ent["EndOffset"]

            if entity_type not in indexes:
                logger.debug("No FAISS index for type %s; skipping.", entity_type)
                continue

            idx, nl = indexes[entity_type]
            names_list = ofac_names[entity_type]

            match = match_entity(entity_text, entity_type, names_list, idx, nl, model)

            # Context-expand REVIEW-tier matches.
            if match.routing == "REVIEW":
                match = recheck_review(
                    match, text, begin_offset, end_offset, names_list, idx, nl, model
                )

            all_results.append(
                {
                    "row_index": row_index,
                    "text": text,
                    "entity_text": entity_text,
                    "entity_type": entity_type,
                    "comprehend_score": comprehend_score,
                    "matched_name": match.matched_name or "",
                    "composite_score": match.composite_score,
                    "stage": match.stage,
                    "routing": match.routing,
                    "BeginOffset": begin_offset,
                    "EndOffset": end_offset,
                }
            )

    # 5. Write output artefacts.
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    write_ner_results(all_results)
    write_confirmed(all_results)
    write_review_queue(all_results)

    logger.info("Analysis complete. %d entities processed.", len(all_results))
    spark.stop()
    return all_results


def _run_train():
    """Prepare annotations from confirmed entities and trigger Flywheel training."""
    import pandas as pd

    from ner_ofac import config
    from ner_ofac.training.annotation_prep import prepare_annotations
    from ner_ofac.training.upload_to_s3 import upload_annotations
    from ner_ofac.training.flywheel_manager import update_flywheel_iteration

    if not os.path.isfile(config.CONFIRMED_ENTITIES_PATH):
        logger.error("Confirmed entities file not found: %s", config.CONFIRMED_ENTITIES_PATH)
        sys.exit(1)

    confirmed_df = pd.read_csv(config.CONFIRMED_ENTITIES_PATH, dtype=str)
    confirmed_df = confirmed_df.dropna(subset=["entity_text", "entity_type"])

    if confirmed_df.empty:
        logger.warning("No confirmed entities found; nothing to train on.")
        return

    # Build entity annotation dicts from confirmed_entities.csv.
    sentences = []
    entities = []
    for i, row in confirmed_df.iterrows():
        entity_text = str(row["entity_text"])
        entity_type = str(row["entity_type"])
        filename = f"confirmed_doc_{i}.txt"
        text = entity_text  # Use entity text as a minimal document.
        sentences.append({"row_index": i, "text": text, "filename": filename})
        entities.append(
            {
                "row_index": i,
                "entity_text": entity_text,
                "entity_type": entity_type,
                "BeginOffset": 0,
                "EndOffset": len(entity_text),
                "line": 0,
            }
        )

    annotation_df = prepare_annotations(sentences, entities)
    s3_uri = upload_annotations(annotation_df, s3_bucket=config.S3_BUCKET)
    training_data_config = {"S3Uri": s3_uri}
    iteration_arn = update_flywheel_iteration(config.FLYWHEEL_ARN, training_data_config)
    logger.info("Train mode complete. Flywheel iteration: %s", iteration_arn)


def _run_feedback(analysis_results):
    """Convert analysis results into training data and trigger retraining.

    Args:
        analysis_results: List of result dicts from ``_run_analyze``.
    """
    import pandas as pd

    from ner_ofac import config
    from ner_ofac.training.feedback_loop import run_feedback

    # Build a minimal sentences DataFrame from analysis results.
    row_lookup = {}
    for r in analysis_results:
        ri = r["row_index"]
        if ri not in row_lookup:
            row_lookup[ri] = {"row_index": ri, "text": r["text"], "filename": f"doc_{ri}.txt"}
    sentences_df = pd.DataFrame(list(row_lookup.values()))

    new_count = run_feedback(
        analysis_results,
        sentences_df,
        flywheel_arn=config.FLYWHEEL_ARN,
        s3_bucket=config.S3_BUCKET,
        output_dir=config.OUTPUT_DIR,
    )
    logger.info("Feedback mode complete. %d new annotation rows added.", new_count)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """Parse CLI arguments and dispatch to the appropriate pipeline mode.

    Raises:
        SystemExit: On invalid arguments.
    """
    parser = argparse.ArgumentParser(
        description="NER OFAC matching and Flywheel training pipeline."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "analyze", "feedback", "both"],
        required=True,
        help=(
            "Pipeline mode: "
            "'train' → prepare annotations + trigger Flywheel; "
            "'analyze' → inference + OFAC matching; "
            "'feedback' → analysis results → training data + retrain; "
            "'both' → analyze then feedback."
        ),
    )
    args = parser.parse_args()

    if args.mode == "train":
        _run_train()
    elif args.mode == "analyze":
        _run_analyze()
    elif args.mode == "feedback":
        logger.warning(
            "Feedback mode requires analysis results.  Run 'both' mode or pass results programmatically."
        )
        # In standalone feedback mode, attempt to load a recent NER results CSV.
        from ner_ofac import config
        import glob
        import csv

        pattern = os.path.join(config.OUTPUT_DIR, "ner_results_*.csv")
        files = sorted(glob.glob(pattern), reverse=True)
        if not files:
            logger.error("No NER results files found in %s. Run 'analyze' first.", config.OUTPUT_DIR)
            sys.exit(1)
        latest = files[0]
        logger.info("Loading analysis results from %s", latest)
        with open(latest, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            analysis_results = list(reader)
        _run_feedback(analysis_results)
    elif args.mode == "both":
        analysis_results = _run_analyze()
        _run_feedback(analysis_results)


if __name__ == "__main__":
    main()
