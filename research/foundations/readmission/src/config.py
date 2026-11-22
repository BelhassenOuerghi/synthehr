"""Configuration loading and path resolution."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# Repository root = parent of the src/ package.
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load config.yaml and resolve relative working paths against the repo root."""
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    # Resolve working directories relative to the repo root (mimic_root stays absolute).
    for key in ("work_dir", "results_dir", "figures_dir"):
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = ROOT / p
        cfg["paths"][key] = str(p)
        os.makedirs(p, exist_ok=True)

    return cfg


def mimic_path(cfg: dict, subdir: str, table: str) -> str:
    """Return the absolute path to a gzipped MIMIC-IV CSV, e.g. ('icu','icustays')."""
    return str(Path(cfg["paths"]["mimic_root"]) / subdir / f"{table}.csv.gz")
