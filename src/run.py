"""
run.py
------
Scheduler for the Pillar 4 experiment. Every (task, seed, arm) is a separate
subprocess (`jobs.py`), so a crash costs one arm, never the run, and the run is
resumable: an arm whose result file exists is skipped.

Two pools run side by side: CPU workers for the CTGAN family and for scoring
pooled or derived releases, and MPS workers for the diffusion models. Longest
jobs start first; a pooled or derived arm starts only when its components'
results exist.

    python run.py --seeds 42                 # one seed end to end
    python run.py --cpu 4 --threads 3 --mps 2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import jobs as J
import settings

SRC = os.path.dirname(os.path.abspath(__file__))
# rough single-process cost, used only to order the queue (longest first)
COST = {"ctgan": 30, "ddpm": 6, "pooled": 2, "TRTR": 2}


def _cost(task, arm):
    base = 1.6 if task == "readmission" else 1.0
    if arm in J.GEN:
        return base * COST[J.GEN[arm]["family"]]
    return base * COST["TRTR" if arm == "TRTR" else "pooled"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--tasks", nargs="*", default=["readmission", "los"])
    ap.add_argument("--arms", nargs="*", default=J.ALL_ARMS)
    ap.add_argument("--cpu", type=int, default=4)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--mps", type=int, default=2)
    a = ap.parse_args()
    cfg = settings.load()
    seeds = a.seeds or cfg["seeds"]
    logdir = os.path.join(cfg["synthetic_root"], "logs")
    os.makedirs(logdir, exist_ok=True)

    todo = [(t, s, arm) for s in seeds for t in a.tasks for arm in a.arms
            if not os.path.exists(J.result_path(t, s, arm))]
    todo.sort(key=lambda x: (-_cost(x[0], x[2]), x[1]))
    running, failed = {}, set()
    print(f"{len(todo)} jobs to run", flush=True)
    t0 = time.time()

    def ready(job):
        t, s, arm = job
        deps = J.deps_of(arm)
        if any((t, s, dpd) in failed for dpd in deps):
            return None                      # a component failed: skip
        return all(os.path.exists(J.result_path(t, s, dpd)) for dpd in deps)

    while todo or running:
        for job, (proc, pool, fh) in list(running.items()):
            if proc.poll() is not None:
                fh.close()
                del running[job]
                status = "ok" if proc.returncode == 0 else \
                    f"FAILED ({proc.returncode})"
                if proc.returncode != 0:
                    failed.add(job)
                print(f"[{(time.time()-t0)/60:6.1f} min] {status}: "
                      f"{job}  ({len(todo)} queued, {len(running)} running)",
                      flush=True)
        busy = {"cpu": sum(p == "cpu" for _, p, _ in running.values()),
                "mps": sum(p == "mps" for _, p, _ in running.values())}
        caps = {"cpu": a.cpu, "mps": a.mps}
        for job in list(todo):
            t, s, arm = job
            pool = J.device_of(arm)
            r = ready(job)
            if r is None:
                todo.remove(job); failed.add(job)
                print(f"skip {job}: a component failed", flush=True)
                continue
            if not r or busy[pool] >= caps[pool]:
                continue
            threads = a.threads if pool == "cpu" else 2
            env = dict(os.environ, OMP_NUM_THREADS=str(threads),
                       MKL_NUM_THREADS=str(threads))
            fh = open(os.path.join(logdir, f"{t}_{s}_{arm}.log"), "a")
            proc = subprocess.Popen(
                [sys.executable, os.path.join(SRC, "jobs.py"), "--task", t,
                 "--seed", str(s), "--arm", arm, "--device", pool,
                 "--threads", str(threads)],
                cwd=SRC, stdout=fh, stderr=subprocess.STDOUT, env=env)
            running[job] = (proc, pool, fh)
            busy[pool] += 1
            todo.remove(job)
        time.sleep(5)
    print(f"done in {(time.time()-t0)/60:.0f} min; "
          f"{len(failed)} failed: {sorted(failed)}", flush=True)


if __name__ == "__main__":
    main()
