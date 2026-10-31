#!/usr/bin/env python3
"""Aggregate all fold/seed/checkpoint metrics for paper analysis."""

import json
from pathlib import Path

import pandas as pd

from run_experiment import HERE, METHODS


def main() -> None:
    records = []
    shares = []
    for method in METHODS:
        for fold in range(5):
            for seed in range(3):
                root = HERE / "outputs" / "runs" / method / f"fold{fold}" / f"seed{seed}"
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
    output = HERE / "outputs" / "summary"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "all_checkpoint_metrics.csv", index=False)
    summary.to_csv(output / "method_checkpoint_summary.csv", index=False)
    pd.DataFrame(shares).to_csv(output / "algorithm_share_key_checkpoints.csv", index=False)
    print(summary[summary.percentage.isin([10, 20, 30, 40])].to_string(index=False))


if __name__ == "__main__":
    main()
