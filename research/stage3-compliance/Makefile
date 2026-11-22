PY ?= python3
SHELL := /bin/bash

.PHONY: all evidence gate test clean

all: evidence gate test

test:                ## negative controls: prove the gate can FAIL a candidate
	$(PY) src/test_gate.py

evidence:            ## compile DPIA / data card / model card / risk register / mapping
	$(PY) src/build_evidence.py

gate:                ## run the release gate; promote datasets on technical pass
	$(PY) src/release_gate.py

clean:
	rm -f doc/evidence/*.md results/*.json
