# The frozen shared modules (CTGAN, loaders, accountant, attacks) live once in
# research/common; scripts find them through PYTHONPATH.
export PYTHONPATH := $(CURDIR)/../common

PY ?= python3
SHELL := /bin/bash

.PHONY: all generate evaluate plots smoke clean

all: generate evaluate plots

generate:            ## fit DP-CTGAN per epsilon on real training data -> synthetic sets
	set -o pipefail; $(PY) src/generate.py 2>&1 | tee results/train_log.txt

evaluate:            ## TRTR vs TSTR(eps) + membership inference -> results/
	$(PY) src/evaluate_utility.py

plots:               ## privacy-utility frontier + MIA figures
	$(PY) src/visualize.py

smoke:               ## tiny single-core end-to-end check (safe to run anytime)
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PY) src/smoke_test.py

clean:
	rm -f results/metrics.json results/results_table.md results/train_log.txt results/plots/*.png
