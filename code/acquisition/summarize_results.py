#!/usr/bin/env python3
"""Aggregate all fold/seed/checkpoint metrics for paper analysis."""

import json
import argparse
from pathlib import Path

import pandas as pd

from run_experiment import METHODS, atomic_json
from release_support import add_path_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize existing acquisition diagnostics; no MolAS training or figure reproduction.")
    add_path_arguments(parser, inputs=False)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if "plain_mean_ucb" not in args.methods:
        parser.error("The historical ratio/delta summaries require plain_mean_ucb in --methods")
    records = []
    shares = []
    for method in args.methods:
        for fold in args.folds:
            for seed in args.seeds:
                root = args.output_dir / ("smoke" if args.smoke_test else "runs") / method / f"fold{fold}" / f"seed{seed}"
                for percentage in range(1, 101):
                    label = f"{percentage:03d}pct"
                    reconstruction = json.loads((root / "reconstruction_metrics" / f"metrics_{label}.json").read_text())
                    runtime = json.loads((root / "runtime_prediction_metrics" / f"metrics_{label}.json").read_text())
                    records.append({"method": method, "fold": fold, "seed": seed, **reconstruction, **runtime})
                    if percentage in {10, 20, 30, 40}:
                        distribution = pd.read_csv(root / "algorithm_query_distributions" / f"distribution_{label}.csv")
                        shares.append({
                            "method": method, "fold": fold, "seed": seed,
                            "percentage": percentage,
                            "max_algorithm_share": distribution["cumulative_share"].max(),
                            "min_algorithm_share": distribution["cumulative_share"].min(),
                        })
    frame = pd.DataFrame(records)
    summary = frame.groupby(["method", "percentage"]).agg(
        runtime_mean=("cumulative_true_runtime", "mean"),
        runtime_std=("cumulative_true_runtime", "std"),
        unqueried_mae_mean=("unqueried_mae", "mean"),
        unqueried_mae_std=("unqueried_mae", "std"),
        ndcg_at_3_mean=("ndcg_at_3", "mean"),
        ndcg_at_3_std=("ndcg_at_3", "std"),
        top1_mean=("top1_agreement", "mean"),
        top1_std=("top1_agreement", "std"),
        vbs_regret_mean=("vbs_regret", "mean"),
        vbs_regret_std=("vbs_regret", "std"),
    ).reset_index()
    baseline = summary[summary.method == "plain_mean_ucb"].set_index("percentage")
    summary["runtime_ratio_to_plain"] = summary.apply(
        lambda row: row.runtime_mean / baseline.loc[row.percentage, "runtime_mean"], axis=1
    )
    summary["mae_delta_to_plain"] = summary.apply(
        lambda row: row.unqueried_mae_mean - baseline.loc[row.percentage, "unqueried_mae_mean"], axis=1
    )
    summary["ndcg_delta_to_plain"] = summary.apply(
        lambda row: row.ndcg_at_3_mean - baseline.loc[row.percentage, "ndcg_at_3_mean"], axis=1
    )
    output = args.output_dir / ("smoke_summary" if args.smoke_test else "summary")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "summary_scope.json", {"methods": args.methods, "folds": args.folds,
                "seeds": args.seeds, "smoke_test": args.smoke_test, "records": len(frame),
                "aggregation": "Descriptive means and standard deviations over the selected fold-seed records; not independent confidence intervals."})
    frame.to_csv(output / "all_checkpoint_metrics.csv", index=False)
    summary.to_csv(output / "method_checkpoint_summary.csv", index=False)
    pd.DataFrame(shares).to_csv(output / "algorithm_share_key_checkpoints.csv", index=False)
    print(summary[summary.percentage.isin([10, 20, 30, 40])].to_string(index=False))


if __name__ == "__main__":
    main()
