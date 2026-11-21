"""Load config.yaml and resolve its paths (relative to the repository, or
overridden by ENGD_REPOS_ROOT / ENGD_SYNTH_ROOT)."""
from __future__ import annotations

import os

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path=None):
    path = path or os.environ.get("ENGD_CONFIG") or os.path.join(
        HERE, "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    for key, env in (("repos_root", "ENGD_REPOS_ROOT"),
                     ("synthetic_root", "ENGD_SYNTH_ROOT")):
        val = os.environ.get(env, cfg[key])
        cfg[key] = os.path.normpath(os.path.join(base, val))
    return cfg
