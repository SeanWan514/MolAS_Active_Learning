#!/usr/bin/env python3
"""Audit existing exports. This does not run acquisition or regenerate results."""

import argparse
import csv
import json

from release_support import add_path_arguments
from run_experiment import METHODS


def audit_incremental(run_root, checkpoints=100):
    """Detect repeated prior blocks, including the historical resume-export bug."""
    seen = set()
    next_order = 1
    issues = []
    for percentage in range(1, checkpoints + 1):
        path = run_root / "queried_cells/newly_queried" / f"queried_{percentage:03d}pct.csv"
        if not path.is_file():
            issues.append(f"Missing incremental checkpoint {percentage}")
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                order, flat = int(row["query_order"]), int(row["flat_index"])
                if order != next_order:
                    issues.append(f"Checkpoint {percentage}: expected query order {next_order}, found {order}")
                if flat in seen:
                    issues.append(f"Checkpoint {percentage}: repeated flat index {flat}")
                seen.add(flat)
                next_order = order + 1
    return {"unique_cells": len(seen), "issues": issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_arguments(parser, inputs=False)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--check-incremental", action="store_true",
                        help="Read incremental logs to check contiguous query orders and uniqueness across checkpoints.")
    args = parser.parse_args()
    records = []
    for method in args.methods:
        for fold in args.folds:
            for seed in args.seeds:
                root = args.output_dir / ("smoke" if args.smoke_test else "runs") / method / f"fold{fold}" / f"seed{seed}"
                counts = {name: len(list((root / subpath).glob("*.csv"))) for name, subpath in {
                    "matrices": "strict_score_matrices", "cumulative_logs": "queried_cells/cumulative",
                    "incremental_logs": "queried_cells/newly_queried"}.items()}
                complete = (root / "run_state/complete.json").is_file()
                item = {"method": method, "fold": fold, "seed": seed, **counts, "complete_marker": complete,
                        "pass": complete and all(count == 100 for count in counts.values())}
                if args.check_incremental:
                    item["incremental_audit"] = audit_incremental(root)
                    item["pass"] = item["pass"] and not item["incremental_audit"]["issues"]
                records.append(item)
    report = {"scope": "file counts and incremental order/uniqueness" if args.check_incremental else "file counts only",
              "expected_runs": len(records), "passing_runs": sum(x["pass"] for x in records), "runs": records}
    print(json.dumps(report, indent=2))
    if not all(x["pass"] for x in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
