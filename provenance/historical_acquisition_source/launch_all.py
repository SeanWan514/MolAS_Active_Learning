#!/usr/bin/env python3
"""Bounded-parallel, resumable launcher for all 90 runs."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from run_experiment import HERE, METHODS


def execute(job: tuple[int, int, str]) -> tuple[tuple[int, int, str], int]:
    fold, seed, method = job
    root = HERE / "outputs" / "runs" / method / f"fold{fold}" / f"seed{seed}"
    complete = root / "run_state" / "complete.json"
    if complete.exists():
        return job, 0
    (root / "run_state").mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(HERE / "run_experiment.py"),
        "--methods", method, "--folds", str(fold), "--seeds", str(seed),
    ]
    environment = os.environ.copy()
    environment["PYTHONWARNINGS"] = "ignore"
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = "1"
    with (root / "run_state" / "worker.log").open("a") as worker_log:
        result = subprocess.run(
            command, stdout=worker_log, stderr=subprocess.STDOUT,
            env=environment, check=False,
        )
    return job, result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    jobs = [(fold, seed, method) for seed in args.seeds for fold in range(5) for method in METHODS]
    status_log = HERE / "outputs" / "launcher_status.jsonl"
    status_log.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, job): job for job in jobs}
        for future in as_completed(futures):
            job, returncode = future.result()
            event = {"fold": job[0], "seed": job[1], "method": job[2], "returncode": returncode}
            with status_log.open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            if returncode:
                failures.append(event)
    if failures:
        raise SystemExit(f"Failed runs: {failures}")


if __name__ == "__main__":
    main()
