from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


STABILITY_SIGNALS = (
    "top1_agreement",
    "top3_jaccard",
    "top_weighted_kendall",
    "score_margin_stability",
)


def pair_stability(previous: np.ndarray, current: np.ndarray) -> dict[str, float]:
    """Row-macro stability of two aligned score matrices; larger means more stable."""
    previous_order = np.argsort(-previous, axis=1, kind="stable")
    current_order = np.argsort(-current, axis=1, kind="stable")
    top1 = (previous_order[:, 0] == current_order[:, 0]).mean()

    overlap = (
        previous_order[:, :3, None] == current_order[:, None, :3]
    ).sum(axis=(1, 2))
    top3 = np.mean(overlap / (6 - overlap))

    pair_i, pair_j = np.triu_indices(previous.shape[1], 1)
    previous_difference = previous[:, pair_i] - previous[:, pair_j]
    current_difference = current[:, pair_i] - current[:, pair_j]
    ranks = np.arange(previous.shape[1])
    previous_rank, current_rank = np.empty_like(previous_order), np.empty_like(current_order)
    np.put_along_axis(previous_rank, previous_order, ranks, axis=1)
    np.put_along_axis(current_rank, current_order, ranks, axis=1)
    joint_rank = np.minimum(previous_rank, current_rank)
    item_weight = 1 / (joint_rank + 1)
    pair_weight = item_weight[:, pair_i] + item_weight[:, pair_j]
    previous_sign, current_sign = np.sign(previous_difference), np.sign(current_difference)
    numerator = (pair_weight * previous_sign * current_sign).sum(axis=1)
    denominator = np.sqrt(
        (pair_weight * previous_sign**2).sum(axis=1)
        * (pair_weight * current_sign**2).sum(axis=1)
    )
    tau = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    tau[(denominator == 0) & np.all(previous_sign == current_sign, axis=1)] = 1
    top_weighted_kendall = np.mean((tau + 1) / 2)

    scale = np.maximum(previous.max(axis=1), current.max(axis=1)) - np.minimum(
        previous.min(axis=1), current.min(axis=1)
    )
    scale = np.maximum(scale, np.finfo(float).eps)
    score_drift = np.mean(np.abs(current - previous), axis=1) / scale
    previous_margin = np.take_along_axis(previous, previous_order[:, :2], axis=1)
    current_margin = np.take_along_axis(current, current_order[:, :2], axis=1)
    previous_margin = (previous_margin[:, 0] - previous_margin[:, 1]) / scale
    current_margin = (current_margin[:, 0] - current_margin[:, 1]) / scale
    margin_drift = np.abs(current_margin - previous_margin)
    score_margin_stability = np.clip(1 - (score_drift + margin_drift) / 2, 0, 1).mean()

    return {
        "top1_agreement": float(top1),
        "top3_jaccard": float(top3),
        "top_weighted_kendall": float(top_weighted_kendall),
        "score_margin_stability": float(score_margin_stability),
        "score_drift": float(score_drift.mean()),
        "margin_drift": float(margin_drift.mean()),
    }


def masked_pair_stability(
    previous: np.ndarray,
    current: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, float]:
    """Row-macro stability among cells eligible in both matrices."""
    counts = eligible.sum(axis=1)
    previous_masked = np.where(eligible, previous, -np.inf)
    current_masked = np.where(eligible, current, -np.inf)
    previous_order = np.argsort(-previous_masked, axis=1, kind="stable")
    current_order = np.argsort(-current_masked, axis=1, kind="stable")

    valid_top1 = counts >= 2
    top1 = (
        np.mean(previous_order[valid_top1, 0] == current_order[valid_top1, 0])
        if valid_top1.any() else np.nan
    )
    valid_top3 = counts >= 4
    if valid_top3.any():
        overlap = (
            previous_order[valid_top3, :3, None]
            == current_order[valid_top3, None, :3]
        ).sum(axis=(1, 2))
        top3 = np.mean(overlap / (6 - overlap))
    else:
        top3 = np.nan

    pair_i, pair_j = np.triu_indices(previous.shape[1], 1)
    pair_eligible = eligible[:, pair_i] & eligible[:, pair_j]
    previous_sign = np.sign(previous[:, pair_i] - previous[:, pair_j])
    current_sign = np.sign(current[:, pair_i] - current[:, pair_j])
    ranks = np.arange(previous.shape[1])
    previous_rank, current_rank = np.empty_like(previous_order), np.empty_like(current_order)
    np.put_along_axis(previous_rank, previous_order, ranks, axis=1)
    np.put_along_axis(current_rank, current_order, ranks, axis=1)
    item_weight = 1 / (np.minimum(previous_rank, current_rank) + 1)
    pair_weight = (item_weight[:, pair_i] + item_weight[:, pair_j]) * pair_eligible
    numerator = (pair_weight * previous_sign * current_sign).sum(axis=1)
    denominator = np.sqrt(
        (pair_weight * previous_sign**2).sum(axis=1)
        * (pair_weight * current_sign**2).sum(axis=1)
    )
    tau = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    same_relations = np.all((previous_sign == current_sign) | ~pair_eligible, axis=1)
    tau[(denominator == 0) & same_relations] = 1
    kendall = np.mean((tau[valid_top1] + 1) / 2) if valid_top1.any() else np.nan

    if valid_top1.any():
        indices = np.flatnonzero(valid_top1)
        mask = eligible[indices]
        previous_valid, current_valid = previous[indices], current[indices]
        upper = np.maximum(
            np.where(mask, previous_valid, -np.inf).max(axis=1),
            np.where(mask, current_valid, -np.inf).max(axis=1),
        )
        lower = np.minimum(
            np.where(mask, previous_valid, np.inf).min(axis=1),
            np.where(mask, current_valid, np.inf).min(axis=1),
        )
        scale = np.maximum(upper - lower, np.finfo(float).eps)
        score_drift = (
            (np.abs(current_valid - previous_valid) * mask).sum(axis=1)
            / counts[indices] / scale
        )
        previous_top2 = np.take_along_axis(previous_valid, previous_order[indices, :2], axis=1)
        current_top2 = np.take_along_axis(current_valid, current_order[indices, :2], axis=1)
        previous_margin = (previous_top2[:, 0] - previous_top2[:, 1]) / scale
        current_margin = (current_top2[:, 0] - current_top2[:, 1]) / scale
        margin_drift = np.abs(current_margin - previous_margin)
        score_margin = np.clip(1 - (score_drift + margin_drift) / 2, 0, 1).mean()
        score_drift_mean, margin_drift_mean = score_drift.mean(), margin_drift.mean()
    else:
        score_margin = score_drift_mean = margin_drift_mean = np.nan

    return {
        "top1_agreement": float(top1),
        "top3_jaccard": float(top3),
        "top_weighted_kendall": float(kendall),
        "score_margin_stability": float(score_margin),
        "score_drift": float(score_drift_mean),
        "margin_drift": float(margin_drift_mean),
        "eligible_cells": int(eligible.sum()),
        "eligible_cell_fraction": float(eligible.mean()),
        "eligible_rows_top1": int(valid_top1.sum()),
        "eligible_rows_top3": int(valid_top3.sum()),
        "eligible_row_fraction_top1": float(valid_top1.mean()),
        "eligible_row_fraction_top3": float(valid_top3.mean()),
    }


def rolling_stability(consecutive: pd.DataFrame, windows=(2, 3, 5)) -> pd.DataFrame:
    """Average the most recent w-1 consecutive transitions from w matrices."""
    rows = []
    group_columns = ["method", "fold", "al_seed"]
    progress_columns = [column for column in ("cost_fraction",) if column in consecutive]
    value_columns = [*STABILITY_SIGNALS, "score_drift", "margin_drift"]
    for keys, trajectory in consecutive.groupby(group_columns, sort=False):
        trajectory = trajectory.sort_values("percentage").reset_index(drop=True)
        for window in windows:
            averaged = trajectory[value_columns].rolling(window - 1, min_periods=window - 1).mean()
            valid = averaged[list(STABILITY_SIGNALS)].notna().any(axis=1)
            part = trajectory.loc[valid, [
                "percentage", "query_cost_h", "query_count", "query_fraction",
                *progress_columns,
            ]].copy()
            part[value_columns] = averaged.loc[valid].to_numpy()
            part[["method", "fold", "al_seed"]] = keys
            part["window"] = window
            rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def downstream_gain_table(
    stability: pd.DataFrame,
    downstream: pd.DataFrame,
    downstream_metrics: tuple[str, ...],
) -> pd.DataFrame:
    progress_columns = [column for column in ("cost_fraction",) if column in stability]
    long = stability.melt(
        id_vars=[
            "method", "fold", "al_seed", "percentage", "query_cost_h",
            "query_count", "query_fraction", *progress_columns, "window",
        ],
        value_vars=STABILITY_SIGNALS,
        var_name="stability_metric",
        value_name="stability",
    )
    points = long.merge(
        downstream[["method", "fold", "percentage", *downstream_metrics]],
        on=["method", "fold", "percentage"],
        how="inner",
        validate="many_to_one",
    ).melt(
        id_vars=[
            "method", "fold", "al_seed", "percentage", "query_cost_h",
            "query_count", "query_fraction", *progress_columns, "window",
            "stability_metric", "stability",
        ],
        value_vars=downstream_metrics,
        var_name="downstream_metric",
        value_name="downstream_value",
    )
    result = []
    keys = ["method", "fold", "al_seed", "window", "stability_metric", "downstream_metric"]
    for _, trajectory in points.groupby(keys, sort=False):
        trajectory = trajectory.sort_values("query_cost_h").copy()
        values = trajectory["downstream_value"]
        trajectory["next_gain"] = values.shift(-1) - values
        trajectory["next_abs_change"] = trajectory["next_gain"].abs()
        trajectory["future_best_gain"] = values.iloc[::-1].cummax().iloc[::-1] - values
        trajectory["endpoint_delta"] = values.iloc[-1] - values
        trajectory["final_cost_h"] = trajectory["query_cost_h"].iloc[-1]
        result.append(trajectory)
    return pd.concat(result, ignore_index=True) if result else points


def proxy_associations(points: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "fold", "al_seed", "window", "stability_metric", "downstream_metric"]
    rows = []
    for values, group in points.groupby(keys, sort=False):
        row = dict(zip(keys, values))
        row["n_checkpoints"] = len(group)
        for target in ("next_gain", "next_abs_change", "future_best_gain"):
            row[f"rho_{target}"] = group["stability"].corr(group[target], method="spearman")
        rows.append(row)
    return pd.DataFrame(rows)


def progress_controlled_associations(
    points: pd.DataFrame,
    progress=("query_fraction", "cost_fraction"),
) -> pd.DataFrame:
    """Trajectory-level partial Spearman associations controlling one progress measure."""
    keys = [
        column for column in (
            "stability_scope", "method", "fold", "al_seed", "window",
            "stability_metric", "downstream_metric",
        ) if column in points
    ]
    rows = []
    for values, trajectory in points.groupby(keys, sort=False):
        base = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        for target in ("next_gain", "next_abs_change", "future_best_gain"):
            for control in progress:
                data = trajectory[["stability", target, control]].dropna().rank()
                unadjusted = data["stability"].corr(data[target])
                if len(data) >= 4:
                    design = np.column_stack([np.ones(len(data)), data[control]])
                    stability_residual = data["stability"] - design @ np.linalg.lstsq(
                        design, data["stability"], rcond=None
                    )[0]
                    target_residual = data[target] - design @ np.linalg.lstsq(
                        design, data[target], rcond=None
                    )[0]
                    controlled = stability_residual.corr(target_residual)
                else:
                    controlled = np.nan
                rows.append({
                    **base,
                    "target": target,
                    "progress_control": control,
                    "n_checkpoints": len(data),
                    "rho_unadjusted": unadjusted,
                    "rho_progress_controlled": controlled,
                    "rho_stability_progress": data["stability"].corr(data[control]),
                    "rho_target_progress": data[target].corr(data[control]),
                })
    return pd.DataFrame(rows)


def simulate_rules(
    points: pd.DataFrame,
    thresholds=(0.90, 0.95, 0.98, 0.99),
    patiences=(1, 2, 3),
) -> pd.DataFrame:
    keys = ["method", "fold", "al_seed", "window", "stability_metric", "downstream_metric"]
    rows = []
    for values, trajectory in points.groupby(keys, sort=False):
        trajectory = trajectory.sort_values("query_cost_h").reset_index(drop=True)
        final_cost = trajectory["query_cost_h"].iloc[-1]
        final_value = trajectory["downstream_value"].iloc[-1]
        future_best = trajectory["downstream_value"].max()
        for threshold, patience in product(thresholds, patiences):
            stable = trajectory["stability"].ge(threshold)
            run = stable.groupby((~stable).cumsum()).cumsum()
            candidates = np.flatnonzero(run.to_numpy() >= patience)
            triggered = bool(len(candidates))
            stop = int(candidates[0]) if triggered else len(trajectory) - 1
            stopped = trajectory.iloc[stop]
            rows.append({
                **dict(zip(keys, values)),
                "threshold": threshold,
                "patience": patience,
                "triggered": triggered,
                "stop_percentage": int(stopped["percentage"]),
                "stop_cost_h": float(stopped["query_cost_h"]),
                "final_cost_h": float(final_cost),
                "cost_saved_h": float(final_cost - stopped["query_cost_h"]),
                "cost_saved_fraction": float(1 - stopped["query_cost_h"] / final_cost),
                "stop_performance": float(stopped["downstream_value"]),
                "endpoint_performance": float(final_value),
                "future_best_performance": float(future_best),
                "endpoint_regret": float(final_value - stopped["downstream_value"]),
                "future_best_regret": float(future_best - stopped["downstream_value"]),
            })
    return pd.DataFrame(rows)


def summarise_rules(simulations: pd.DataFrame) -> pd.DataFrame:
    keys = ["stability_metric", "window", "threshold", "patience", "downstream_metric"]
    summary = simulations.groupby(keys, as_index=False).agg(
        n_trajectories=("fold", "size"),
        n_methods=("method", "nunique"),
        triggered_fraction=("triggered", "mean"),
        median_cost_saved_h=("cost_saved_h", "median"),
        median_cost_saved_fraction=("cost_saved_fraction", "median"),
        mean_future_best_regret=("future_best_regret", "mean"),
        p90_future_best_regret=("future_best_regret", lambda values: values.quantile(0.90)),
        max_future_best_regret=("future_best_regret", "max"),
        mean_endpoint_regret=("endpoint_regret", "mean"),
    )
    by_method = simulations.groupby([*keys, "method"], as_index=False).agg(
        method_mean_cost_saved_fraction=("cost_saved_fraction", "mean"),
        method_mean_future_best_regret=("future_best_regret", "mean"),
    )
    robustness = by_method.groupby(keys, as_index=False).agg(
        min_method_mean_cost_saved_fraction=("method_mean_cost_saved_fraction", "min"),
        worst_method_mean_future_best_regret=("method_mean_future_best_regret", "max"),
    )
    summary = summary.merge(robustness, on=keys, validate="one_to_one")
    summary["all_trajectories_triggered"] = summary["triggered_fraction"].eq(1)
    return summary
