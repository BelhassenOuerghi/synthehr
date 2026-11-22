"""Run the whole suite without pytest: ``python tests/run_tests.py``.
(With pytest installed, ``pytest tests`` runs the same functions.)"""
from __future__ import annotations

import glob
import importlib
import os
import sys
import time
import traceback
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
warnings.filterwarnings("ignore")


def main(patterns=None) -> int:
    files = sorted(glob.glob(os.path.join(HERE, "test_*.py")))
    if patterns:
        files = [f for f in files if any(p in f for p in patterns)]
    failures = 0
    for path in files:
        mod = importlib.import_module(os.path.basename(path)[:-3])
        for name in sorted(dir(mod)):
            fn = getattr(mod, name)
            if name.startswith("test_") and callable(fn):
                t = time.time()
                try:
                    fn()
                    print(f"PASS  {mod.__name__}.{name}  ({time.time()-t:.1f}s)",
                          flush=True)
                except Exception:
                    failures += 1
                    print(f"FAIL  {mod.__name__}.{name}", flush=True)
                    traceback.print_exc()
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
