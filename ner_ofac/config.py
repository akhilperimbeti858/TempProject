"""Configuration module for the NER OFAC matching system.

This module centralizes all tunable parameters for the pipeline, including
AWS service configuration, OFAC matching thresholds, file paths, and the
auto-feedback loop settings. Modify these values to tune the system behavior
without changing any pipeline code.
"""

# ---------------------------------------------------------------------------
# AWS / Cloud configuration
# ---------------------------------------------------------------------------
AWS_REGION = "us-east-1"
"""AWS region where Comprehend, S3, and related services are deployed."""

S3_BUCKET = "<your-bucket>"
"""S3 bucket name used for storing training data, annotations, and results."""

COMPREHEND_ENDPOINT_ARN = "<arn>"
"""ARN of the custom Amazon Comprehend endpoint for NER inference."""

FLYWHEEL_ARN = "<arn>"
"""ARN of the Amazon Comprehend Flywheel used for continuous model training."""

# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------
ENTITY_TYPES = ["SANCTIONED_PERSON", "SANCTIONED_ORG"]
"""Entity types produced by the custom Comprehend model.

SANCTIONED_PERSON maps to individual rows in the OFAC SDN list.
SANCTIONED_ORG maps to entity/vessel/aircraft rows in the OFAC SDN list.
"""

# ---------------------------------------------------------------------------
# Matching thresholds
# ---------------------------------------------------------------------------
FUZZY_THRESHOLD = 85
"""Minimum rapidfuzz token_sort_ratio score (0–100) to pass Stage 2 fuzzy matching."""

FUZZY_WEIGHT = 0.4
"""Weight applied to the normalised fuzzy score in the composite scorer."""

SEMANTIC_WEIGHT = 0.4
"""Weight applied to the normalised semantic (FAISS) score in the composite scorer."""

# Note: exact_substring_bonus contributes the remaining 0.2 weight.

CONFIRMED_THRESHOLD = 0.90
"""Composite score at or above which a match is routed as CONFIRMED."""

REVIEW_THRESHOLD = 0.65
"""Composite score at or above which (but below CONFIRMED_THRESHOLD) a match is routed as REVIEW.
Matches below this threshold are routed as NO_MATCH."""

CONTEXT_WINDOW_WORDS = 5
"""Number of words to expand left and right around an entity span for context re-scoring."""

# ---------------------------------------------------------------------------
# Embedding / semantic model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
"""Sentence-transformers model used to encode entity names and query strings for FAISS search."""

# ---------------------------------------------------------------------------
# Auto-feedback loop thresholds
# ---------------------------------------------------------------------------
AUTO_FEEDBACK = False
"""Master flag for the automatic feedback loop.  Set True to enable auto-labelling and retraining."""

AUTO_LABEL_CONFIRMED_MIN = 0.90
"""Minimum composite score for a result to be auto-labelled as a positive training example."""

AUTO_LABEL_NEGATIVE_MIN = 0.85
"""Minimum Comprehend confidence required alongside composite < 0.65 to auto-label as negative."""

# ---------------------------------------------------------------------------
# File / directory paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = "./output"
"""Directory where NER result CSVs and review queue files are written."""

OFAC_CSV_PATH = "./data/ofac_sdn.csv"
"""Local path to the OFAC Specially Designated Nationals (SDN) CSV file."""

CONFIRMED_ENTITIES_PATH = "./confirmed_entities.csv"
"""Path to the running log of confirmed matched entities."""
