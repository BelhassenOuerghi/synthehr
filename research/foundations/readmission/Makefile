# Reproducibility entry points for the 30-day ICU readmission replication.
# Uses the shared EngD replication virtualenv.
PY ?= python3

.PHONY: help cohort features run run-fresh clean

help:
	@echo "make cohort     - build cohort + labels only, print summary"
	@echo "make features   - build the full feature frame (scans chartevents/labevents)"
	@echo "make run        - full pipeline reusing the cached feature parquet if present"
	@echo "make run-fresh  - full pipeline, re-extracting features from MIMIC-IV"
	@echo "make clean      - remove derived data/ artefacts (keeps code, results, figures)"

cohort:
	$(PY) -m src.build_cohort

features:
	$(PY) -m src.features

run:
	$(PY) -m src.run_pipeline --use-cache

run-fresh:
	$(PY) -m src.run_pipeline

clean:
	rm -rf data/processed/*.parquet data/processed/*.json models/*
