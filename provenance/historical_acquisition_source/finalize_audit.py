#!/usr/bin/env python3
"""Completeness audit for all expected fold/seed/method artifacts."""

import json
from pathlib import Path

import pandas as pd

from run_experiment import HERE, METHODS


def main() -> None:
    records = []
    for method in METHODS:
        for fold in range(5):
            for seed in range(3):
                root = HERE / "outputs" / "runs" / method / f"fold{fold}" / f"seed{seed}"
                matrices = list((root / "strict_score_matrices").glob("*.csv"))
                cumulative = list((root / "queried_cells" / "cumulative").glob("*.csv"))
                newly = list((root / "queried_cells" / "newly_queried").glob("*.csv"))
                complete = root / "run_state" / "complete.json"
                records.append({
                    "method": method, "fold": fold, "seed": seed,
                    "matrices": len(matrices), "cumulative_logs": len(cumulative),
                    "new_logs": len(newly), "complete": complete.exists(),
                    "pass": complete.exists() and len(matrices) == len(cumulative) == len(newly) == 100,
                })
    frame = pd.DataFrame(records)
    output = HERE / "outputs" / "audits"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "completeness.csv", index=False)
    summary = {
        "expected_runs": 90,
        "complete_runs": int(frame["complete"].sum()),
        "passing_runs": int(frame["pass"].sum()),
        "status": "pass" if bool(frame["pass"].all()) else "incomplete",
    }
    (output / "completeness_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
