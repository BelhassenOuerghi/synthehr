PY ?= python3

.PHONY: install test demo demo-quick verify-research

install:          ## install the tool and test dependencies (editable)
	$(PY) -m pip install -e ".[dev]"

test:             ## the full suite on fabricated data (no MIMIC needed)
	$(PY) tests/run_tests.py

demo-quick:       ## end-to-end run on a fabricated cohort, about a minute
	PYTHONPATH=src $(PY) -m synthehr.cli demo --quick --out synthehr-demo

demo:             ## the same with realistic model sizes (minutes on a GPU)
	PYTHONPATH=src $(PY) -m synthehr.cli demo --out synthehr-demo

verify-research:  ## the frozen research modules are byte-identical to the runs
	cd research/common && shasum -a 256 -c FROZEN.sha256
