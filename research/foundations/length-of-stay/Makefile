# Override PY to point at your environment, e.g.
#   make all PY=python3
PY ?= python

.PHONY: extract train eval all clean

all: extract train eval

extract:
	$(PY) src/extract.py

train:
	$(PY) src/train.py

eval:
	$(PY) src/evaluate.py
	$(PY) src/visualize.py

clean:
	rm -rf data/derived/* models/* \
	       results/metrics.json \
	       results/results_table.md \
	       results/slos_by_unit.csv \
	       results/plots/*
