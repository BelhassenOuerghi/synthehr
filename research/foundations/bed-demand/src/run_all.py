"""
run_all.py
----------
End-to-end pipeline: derive series from MIMIC-IV -> rolling backtest -> figures.

    python src/run_all.py [--config config.yaml]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_series
import backtest
import make_plots
import fill_readme

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    build_series.build(args.config)
    backtest.main(args.config)
    make_plots.main(args.config)
    fill_readme.main(args.config)
