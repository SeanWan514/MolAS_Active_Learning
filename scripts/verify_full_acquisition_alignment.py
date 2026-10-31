#!/usr/bin/env python3
"""Cross-check final acquisition artifacts with existing inputs and fixed folds.

Checks existing data only: no docking, acquisition loop, completion, or fitting.
The original CSV exporter used %.8g; comparisons respect that serialization.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
import math
from pathlib import Path
import tarfile

ALGORITHMS = ("surfdock", "unimol", "diffdock", "diffdock_l", "gnina", "autodock", "smina")


def rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    root = args.root
    truth_rows = rows((root / "data/targets/full_strict_score_matrix.csv").read_bytes())
    cost_rows = rows((root / "data/raw/moad_runtime_trimmed.csv").read_bytes())
    truth = {r["PDB_CCD_ID"]: {a: float(r[a + "_strict_score"]) for a in ALGORITHMS} for r in truth_rows}
    costs = {r["PDB_CCD_ID"]: {a: float(r[a + "_runtime"]) for a in ALGORITHMS} for r in cost_rows}
    split = rows((root / "data/splits/moad_5fold_split_seed42.csv").read_bytes())
    folds = {r["instance_id"]: int(r["test_fold"]) for r in split}
    archives = sorted((root / "results/acquisition").glob("*/fold*_seed*.tar.xz"))

    def compatible(actual, expected):
        return math.isclose(float(actual), float("%.8g" % expected), rel_tol=1e-12, abs_tol=1e-14)

    def check(path):
        fold = int(path.name.split("_")[0][4:])
        seed = int(path.name.split("_")[1].split(".")[0][4:])
        expected = {x for x, f in folds.items() if f != fold}
        wanted = {"strict_score_matrices/strict_score_100pct.csv", "queried_cells/cumulative/queried_100pct.csv"}
        payload = {}
        with tarfile.open(path, "r:xz") as tf:
            for item in tf:
                if item.name in wanted:
                    payload[item.name] = tf.extractfile(item).read()
        matrix = rows(payload["strict_score_matrices/strict_score_100pct.csv"])
        ledger = rows(payload["queried_cells/cumulative/queried_100pct.csv"])
        ids = [r["PDB_CCD_ID"] for r in matrix]
        if len(ids) != len(expected) or set(ids) != expected:
            raise ValueError("Matrix training-fold membership mismatch: " + str(path))
        for r in matrix:
            for a in ALGORITHMS:
                if not compatible(r[a], truth[r["PDB_CCD_ID"]][a]):
                    raise ValueError("Full matrix differs from observed strict targets: " + str(path))
        n = len(expected) * len(ALGORITHMS)
        cells = [(r["row_label"], r["algorithm"]) for r in ledger]
        if len(cells) != n or len(set(cells)) != n or [int(r["query_order"]) for r in ledger] != list(range(1, n + 1)):
            raise ValueError("Final ledger coverage/order mismatch: " + str(path))
        warm = []
        for r in ledger:
            identity, a = r["row_label"], r["algorithm"]
            if identity not in expected or a not in ALGORITHMS:
                raise ValueError("Held-out or invalid cell in acquisition ledger")
            if not compatible(r["score"], truth[identity][a]) or not compatible(r["runtime"], costs[identity][a]):
                raise ValueError("Recorded outcome/runtime differs from raw inputs: " + str(path))
            if int(r["round"]) == 0:
                warm.append((identity, a))
        return {"archive": str(path.relative_to(root)), "fold": fold, "seed": seed,
                "training_complexes": len(ids), "observed_cells": n, "warm_start": warm}

    with ThreadPoolExecutor(max_workers=4) as ex:
        checked = list(ex.map(check, archives))
    signatures = {}
    for r in checked:
        key = (r["fold"], r["seed"])
        warm = r.pop("warm_start")
        r["warm_start_cells"] = len(warm)
        if key in signatures and signatures[key] != warm:
            raise ValueError("Warm starts differ between policies for fold/seed " + str(key))
        signatures[key] = warm
    if len(checked) != 90:
        raise ValueError("Expected 90 runs")
    report = {"status": "pass", "runs_checked": len(checked),
              "full_matrices_match_existing_strict_targets": True,
              "final_ledgers_cover_training_cells_exactly_once": True,
              "held_out_cells_in_acquisition_ledgers": 0,
              "recorded_scores_and_runtimes_match_inputs_at_export_precision": True,
              "warm_starts_match_across_six_policies_for_every_fold_seed": True,
              "new_experiments_executed": False, "details": checked}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
