"""Training sub-package for the NER OFAC pipeline.

Contains modules for preparing annotation data, uploading to S3, managing
the Comprehend Flywheel continuous training loop, and running the automated
feedback pipeline that converts analysis results into new training examples.
"""
