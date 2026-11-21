PY ?= python
SHELL := /bin/bash

.PHONY: test pilot run run-seed aggregate plots verify-vendored

test:               ## unit tests on fabricated data (no MIMIC needed)
	$(PY) tests/test_tabddpm.py && $(PY) tests/test_pooling.py

verify-vendored:    ## the frozen Stage 0-2 harness must be byte-identical
	cd src && shasum -a 256 -c VENDORED.sha256

pilot:              ## the one hyperparameter search (seed 42 training rows only)
	cd src && $(PY) -u pilot.py

run:                ## every arm, every seed; resumable
	cd src && $(PY) -u run.py

run-seed:           ## one seed end to end, e.g. make run-seed SEED=42
	cd src && $(PY) -u run.py --seeds $(SEED)

aggregate:          ## CIs and the pre-registered tests -> results/
	cd src && $(PY) aggregate.py

plots:              ## figures -> results/plots/
	cd src && $(PY) visualize.py
