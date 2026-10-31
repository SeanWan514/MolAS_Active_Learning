#!/usr/bin/env python3
"""Explicit, bounded parallel launcher. Running this script starts experiments."""

import argparse
import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from release_support import add_path_arguments
from run_experiment import HERE, METHODS, atomic_json


def execute(job, args, launch_id):
    method, fold, seed = job
    command = [sys.executable, str(HERE / "run_experiment.py"),
               "--methods", method, "--folds", str(fold), "--seeds", str(seed)]
    for option, value in (("--data-dir", args.data_dir), ("--features", args.features),
                          ("--feature-ids", args.feature_ids), ("--descriptors", args.descriptors),
                          ("--fold-split", args.fold_split), ("--output-dir", args.output_dir)):
        command.extend([option, str(value.expanduser().resolve())])
    if args.smoke_test:
        command.append("--smoke-test")
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    log = args.output_dir / "launcher_logs" / launch_id / f"{method}_fold{fold}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=environment, check=False)
    return {"method": method, "fold": fold, "seed": seed, "returncode": result.returncode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_arguments(parser)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.output_dir = args.output_dir.expanduser().resolve()
    jobs = [(method, fold, seed) for seed in args.seeds for fold in args.folds for method in args.methods]
    if len(set(jobs)) != len(jobs):
        parser.error("Repeated methods, folds, or seeds would launch duplicate concurrent jobs")
    launch_id = uuid.uuid4().hex
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, job, args, launch_id) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    atomic_json(args.output_dir / "launches" / f"{launch_id}.json", {
        "methods": args.methods, "folds": args.folds, "seeds": args.seeds,
        "smoke_test": args.smoke_test, "workers": args.workers, "results": results,
    })
    if any(result["returncode"] for result in results):
        raise SystemExit("One or more jobs failed; inspect launcher_logs. Matching completed runs may be resumed safely.")


if __name__ == "__main__":
    main()
