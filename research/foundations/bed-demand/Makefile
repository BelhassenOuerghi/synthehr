# Reproduces the full experiment. Point PY at any Python >= 3.9 with the
# packages in requirements.txt (default: the shared EngD replication venv).
PY ?= python3

.PHONY: all series backtest plots clean

all: series backtest plots

series:
	$(PY) src/build_series.py

backtest:
	$(PY) src/backtest.py

plots:
	$(PY) src/make_plots.py

clean:
	rm -rf data/derived results/plots results/metrics.json results/results_table.md
