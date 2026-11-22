# The frozen shared modules (CTGAN, loaders, accountant, attacks) live once in
# research/common; scripts find them through PYTHONPATH.
export PYTHONPATH := $(CURDIR)/../common

PY ?= python3
SHELL := /bin/bash

.PHONY: all generate evaluate plots clean

all: generate evaluate plots

generate:            ## fit CTGAN on real training data, write synthetic sets
	set -o pipefail; $(PY) src/generate.py 2>&1 | tee results/train_log.txt

evaluate:            ## TRTR vs TSTR utility on the real test sets
	$(PY) src/evaluate_utility.py

plots:               ## fidelity + utility figures
	$(PY) src/visualize.py

clean:
	rm -f results/metrics.json results/results_table.md results/plots/*.png
