# The frozen shared modules (CTGAN, loaders, accountant, attacks) live once in
# research/common; scripts find them through PYTHONPATH.
export PYTHONPATH := $(CURDIR)/../common

PY ?= python3
SHELL := /bin/bash

.PHONY: all run aggregate smoke clean

all: run noise aggregate

run:                 ## multi-seed re-run of stages 0/1/2 on patient-level splits (resumable)
	set -o pipefail; $(PY) src/multiseed.py 2>&1 | tee -a results/run_log.txt

aggregate:           ## mean +/- 95% CI + paired tests -> results/
	$(PY) src/aggregate.py

smoke:               ## tiny config end-to-end check before committing 20h of compute
	$(PY) src/multiseed.py --config config-smoke.yaml && $(PY) src/aggregate.py

clean:
	rm -f results/seeds/*.json results/metrics_ci.json results/results_table.md

noise:               ## negative control: independent DP replicates (same recipe, different model randomness)
	set -o pipefail; $(PY) src/noise_control.py 2>&1 | tee -a results/noise_log.txt

plots:               ## headline figures: utility with CIs, instability, noise floor
	$(PY) src/visualize.py
