"""Load config.yaml and resolve its paths (relative to the repository, or
overridden by SYNTHEHR_REPOS_ROOT / SYNTHEHR_SYNTH_ROOT)."""
from __future__ import annotations

import os

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path=None):
    path = path or os.environ.get("SYNTHEHR_CONFIG") or os.path.join(
        HERE, "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    for key, env in (("repos_root", "SYNTHEHR_REPOS_ROOT"),
                     ("synthetic_root", "SYNTHEHR_SYNTH_ROOT")):
        val = os.environ.get(env, cfg[key])
        cfg[key] = os.path.normpath(os.path.join(base, val))
    return cfg
