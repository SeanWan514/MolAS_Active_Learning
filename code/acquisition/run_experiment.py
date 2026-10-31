#!/usr/bin/env python3
"""Portable release of the six-policy offline MolAS acquisition runner.

Historical source is preserved separately; see README.md for provenance and
release-only I/O, validation, metadata, and resume-log changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from release_support import (add_path_arguments, environment_versions, fingerprint,
                             invocation_arguments, previous_checkpoint_size, verify_existing_run)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data/raw"
FEATURE_CACHE = ROOT / "data/features/complex_features_ligand_protein_d256_seed0.npy"
FEATURE_IDS = ROOT / "data/features/complex_feature_ids.csv"
OUTPUT = ROOT / "outputs/acquisition"
DESCRIPTORS = ROOT / "data/descriptors/moad_physical_descriptors_filtered.csv"
FOLD_SPLIT = ROOT / "data/splits/moad_5fold_split_seed42.csv"


def configure_paths(args: argparse.Namespace) -> None:
    global DATA, FEATURE_CACHE, FEATURE_IDS, OUTPUT, DESCRIPTORS, FOLD_SPLIT
    DATA = args.data_dir.expanduser().resolve()
    FEATURE_CACHE = args.features.expanduser().resolve()
    FEATURE_IDS = args.feature_ids.expanduser().resolve()
    OUTPUT = args.output_dir.expanduser().resolve()
    DESCRIPTORS = args.descriptors.expanduser().resolve()
    FOLD_SPLIT = args.fold_split.expanduser().resolve()


ALGORITHMS = [
    "surfdock",
    "unimol",
    "diffdock",
    "diffdock_l",
    "gnina",
    "autodock",
    "smina",
]
SVD_RANKS = (1, 2, 3, 5)
LAMBDA = 3.0
METHODS = (
    "plain_mean_ucb",
    "direct_cost_uc",
    "runtime_uncertainty",
    "cost_tiers_5",
    "relaxed_cost_power_05",
    "log_cost",
)
COST_METHODS = set(METHODS) - {"plain_mean_ucb"}
ENSEMBLE_COST_METHODS = {"runtime_uncertainty"}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, allow_nan=True)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame, *, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        frame.to_csv(handle, index=index, float_format="%.8g")
        tmp = Path(handle.name)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_matrix(name: str) -> tuple[pd.Index, np.ndarray]:
    frame = pd.read_csv(DATA / name, index_col=0)
    if "rmsd" in name:
        suffix = "_rmsd"
    elif "pb_ratio" in name:
        suffix = "_pb_ratio"
    elif "runtime" in name:
        suffix = "_runtime"
    else:
        raise ValueError(f"Unknown matrix schema: {name}")
    source_columns = [f"{algorithm}{suffix}" for algorithm in ALGORITHMS]
    missing = [column for column in source_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing algorithm columns: {missing}")
    return frame.index, frame.loc[:, source_columns].to_numpy(dtype=np.float64)


def _identifier_column(frame: pd.DataFrame, expected: pd.Index, source: Path) -> str:
    expected_values = set(expected.astype(str))
    matches = []
    for column in frame.columns:
        values = frame[column].astype(str)
        if values.is_unique and set(values) == expected_values:
            matches.append(column)
    if len(matches) != 1:
        raise ValueError(
            f"{source.name}: expected exactly one identifier column matching all matrix rows; "
            f"found {matches}"
        )
    return matches[0]


def aligned_auxiliary_frame(path: Path, expected: pd.Index) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input: {path}. See {HERE / 'README.md'}"
        )
    frame = pd.read_csv(path)
    identifier = _identifier_column(frame, expected, path)
    frame[identifier] = frame[identifier].astype(str)
    aligned = frame.set_index(identifier).loc[expected.astype(str)]
    if aligned.index.duplicated().any():
        raise ValueError(f"{path.name}: duplicate complex identifiers")
    return aligned


def load_descriptors(expected: pd.Index) -> np.ndarray:
    frame = aligned_auxiliary_frame(DESCRIPTORS, expected)
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] != 4:
        raise ValueError(
            f"{DESCRIPTORS.name}: expected exactly four numeric physical descriptors, "
            f"found {list(numeric.columns)}"
        )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{DESCRIPTORS.name}: physical descriptors contain missing/non-finite values")
    return numeric.to_numpy(dtype=np.float32)


def load_folds(expected: pd.Index) -> np.ndarray:
    frame = aligned_auxiliary_frame(FOLD_SPLIT, expected)
    candidate_columns = [
        column for column in frame.columns
        if "fold" in column.lower() and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if len(candidate_columns) != 1:
        raise ValueError(
            f"{FOLD_SPLIT.name}: expected one numeric fold column, found {candidate_columns}"
        )
    values = frame[candidate_columns[0]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{FOLD_SPLIT.name}: fold labels must be finite integers")
    folds = values.astype(np.int64)
    unique = sorted(np.unique(folds).tolist())
    if unique == [1, 2, 3, 4, 5]:
        folds = folds - 1
        unique = sorted(np.unique(folds).tolist())
    if unique != [0, 1, 2, 3, 4]:
        raise ValueError(f"{FOLD_SPLIT.name}: expected five folds encoded 0-4 or 1-5; found {unique}")
    return folds


def strict_score(rmsd: np.ndarray, pb_ratio: np.ndarray) -> np.ndarray:
    """Established MolAS strict-score definition used by Design 2."""
    with np.errstate(over="ignore", invalid="ignore"):
        rmsd_score = (1.0 + np.exp(-2.0 * LAMBDA)) / (
            1.0 + np.exp(LAMBDA * (rmsd - 2.0))
        )
    return np.clip(np.nan_to_num(rmsd_score) * (pb_ratio == 1.0), 0.0, 1.0)


def svd_complete(
    truth: np.ndarray,
    observed: np.ndarray,
    rank: int,
    iterations: int = 6,
) -> np.ndarray:
    if observed.all():
        return truth.copy()
    if not observed.any():
        raise ValueError("SVD completion requires an observed warm start")
    global_mean = float(truth[observed].mean())
    filled = np.empty_like(truth)
    for column in range(observed.shape[1]):
        mask = observed[:, column]
        filled[:, column] = float(truth[mask, column].mean()) if mask.any() else global_mean
    filled[observed] = truth[observed]
    for _ in range(iterations):
        column_mean = filled.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(filled - column_mean, full_matrices=False)
        use_rank = min(rank, len(s))
        filled = (u[:, :use_rank] * s[:use_rank]) @ vt[:use_rank] + column_mean
        filled = np.clip(filled, 0.0, 1.0)
        filled[observed] = truth[observed]
    return filled


def score_ensemble(truth: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack([svd_complete(truth, observed, rank, iterations=4) for rank in SVD_RANKS])
    return stack.mean(axis=0), stack.std(axis=0)


def warm_start(shape: tuple[int, int], seed: int, fraction: float = 0.01) -> np.ndarray:
    """Runtime-blind warm start with nearly equal representation by algorithm."""
    rows, columns = shape
    count = math.ceil(rows * columns * fraction)
    rng = np.random.default_rng(731_000 + seed)
    chosen: list[int] = []
    per_algorithm = count // columns
    remainder = count % columns
    for column in range(columns):
        number = per_algorithm + int(column < remainder)
        selected_rows = rng.choice(rows, size=number, replace=False)
        chosen.extend((selected_rows * columns + column).tolist())
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=np.int64)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {key: math.nan for key in ("min", "q05", "q25", "median", "q75", "q95", "max")}
    qs = np.quantile(values, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
    return dict(zip(("min", "q05", "q25", "median", "q75", "q95", "max"), map(float, qs)))


@dataclass
class RuntimePrediction:
    mean: np.ndarray
    sd: np.ndarray
    conservative: np.ndarray
    lower: float
    upper: float


class RuntimeSurrogate:
    """Observed-only runtime predictor with positive, round-local bounds."""

    def __init__(self, features: np.ndarray, algorithms: int = 7):
        one_hot = np.eye(algorithms, dtype=np.float32)
        expanded = np.repeat(features, algorithms, axis=0)
        algorithm_features = np.tile(one_hot, (features.shape[0], 1))
        self.x = np.concatenate([expanded, algorithm_features], axis=1)

    def predict(
        self,
        observed_flat: np.ndarray,
        observed_runtime: np.ndarray,
        candidates: np.ndarray,
        seed: int,
        ensemble_size: int,
        legacy: bool = False,
    ) -> RuntimePrediction:
        y = np.log1p(observed_runtime)
        q05, q95 = np.quantile(observed_runtime, [0.05, 0.95])
        lower = max(float(q05), np.finfo(float).eps)
        upper = max(float(q95), lower * (1.0 + 1e-6))
        predictions = []
        for member in range(ensemble_size):
            member_seed = seed * 1009 + member * 97
            model = make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=min(128, len(observed_flat)),
                    learning_rate_init=1e-3,
                    max_iter=10,
                    early_stopping=len(observed_flat) >= 80,
                    validation_fraction=0.15,
                    n_iter_no_change=5,
                    random_state=member_seed,
                ),
            )
            model.fit(self.x[observed_flat], y)
            predicted_log = model.predict(self.x[candidates])
            raw = np.expm1(predicted_log)
            if legacy:
                raw = np.clip(raw, 1e-6, float(np.median(observed_runtime) * 100.0))
            else:
                raw = np.clip(raw, lower, upper)
            predictions.append(raw)
        stack = np.stack(predictions)
        mean = stack.mean(axis=0)
        sd = stack.std(axis=0)
        conservative = mean + sd
        if not legacy:
            conservative = np.clip(conservative, lower, upper)
        return RuntimePrediction(mean, sd, conservative, lower, upper)


class HiddenOracle:
    """The only object permitted to hold unrevealed scores and runtimes."""

    def __init__(self, score: np.ndarray, runtime: np.ndarray):
        self._score = score
        self._runtime = runtime
        self.shape = score.shape
        self.access_log: list[dict[str, int]] = []

    def reveal(self, flat: np.ndarray, round_id: int) -> tuple[np.ndarray, np.ndarray]:
        rows, columns = np.unravel_index(flat, self.shape)
        self.access_log.extend(
            {"flat_index": int(index), "round": int(round_id)} for index in flat
        )
        return self._score[rows, columns], self._runtime[rows, columns]

    def evaluate(self, prediction: np.ndarray, observed: np.ndarray) -> dict[str, float]:
        """Evaluate without exposing truth back to the acquisition pipeline."""
        truth = self._score
        missing = ~observed
        result = {
            "full_mae": float(mean_absolute_error(truth.ravel(), prediction.ravel())),
            "full_rmse": float(mean_squared_error(truth.ravel(), prediction.ravel()) ** 0.5),
        }
        if missing.any():
            result["unqueried_mae"] = float(mean_absolute_error(truth[missing], prediction[missing]))
            result["unqueried_rmse"] = float(mean_squared_error(truth[missing], prediction[missing]) ** 0.5)
        else:
            result["unqueried_mae"] = 0.0
            result["unqueried_rmse"] = 0.0
        true_top = truth.argmax(axis=1)
        pred_top = prediction.argmax(axis=1)
        result["top1_agreement"] = float(np.mean(true_top == pred_top))
        true_top3 = np.argsort(-truth, axis=1)[:, :3]
        pred_top3 = np.argsort(-prediction, axis=1)[:, :3]
        result["top3_overlap"] = float(
            np.mean([len(set(a).intersection(b)) / 3.0 for a, b in zip(true_top3, pred_top3)])
        )
        discounts = 1.0 / np.log2(np.arange(2, 5))
        ndcgs = []
        for row in range(truth.shape[0]):
            gain = truth[row, pred_top3[row]]
            ideal = truth[row, true_top3[row]]
            denominator = float(np.sum(ideal * discounts))
            ndcgs.append(float(np.sum(gain * discounts) / denominator) if denominator else 1.0)
        result["ndcg_at_3"] = float(np.mean(ndcgs))
        pair_correct: list[np.ndarray] = []
        for left in range(truth.shape[1]):
            for right in range(left + 1, truth.shape[1]):
                true_sign = np.sign(truth[:, left] - truth[:, right])
                pred_sign = np.sign(prediction[:, left] - prediction[:, right])
                non_tie = true_sign != 0
                if non_tie.any():
                    pair_correct.append((true_sign[non_tie] == pred_sign[non_tie]).astype(float))
        result["pairwise_ranking_accuracy"] = float(np.concatenate(pair_correct).mean())
        result["vbs_regret"] = float(np.mean(truth.max(axis=1) - truth[np.arange(truth.shape[0]), pred_top]))
        return result


def continuous_cost_scale(cost: np.ndarray) -> np.ndarray:
    if np.allclose(cost.min(), cost.max()):
        return np.ones_like(cost)
    return 1.0 + 4.0 * (cost - cost.min()) / (cost.max() - cost.min())


def cost_tiers(cost: np.ndarray, number: int) -> np.ndarray:
    edges = np.quantile(cost, np.arange(1, number) / number)
    return np.digitize(cost, edges, right=True).astype(float) + 1.0


def acquisition_values(
    method: str,
    utility_mean: np.ndarray,
    utility_sd: np.ndarray,
    runtime: RuntimePrediction | None,
    candidate_algorithms: np.ndarray,
    cumulative_algorithm_counts: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    utility = utility_mean + utility_sd
    if method == "plain_mean_ucb":
        return utility, utility.copy(), np.full(utility.size, np.nan)
    assert runtime is not None
    if method == "direct_cost_uc":
        return utility, utility / (runtime.mean + 1e-6), runtime.mean

    normalized_utility = np.maximum(percentile_rank(utility), 1e-3)
    selected_cost = runtime.conservative if method in ENSEMBLE_COST_METHODS else runtime.mean
    scaled_cost = continuous_cost_scale(selected_cost)
    if method == "relaxed_cost_power_05":
        acquisition = normalized_utility / np.sqrt(scaled_cost)
    elif method == "cost_tiers_5":
        acquisition = normalized_utility / cost_tiers(selected_cost, 5)
    elif method == "runtime_uncertainty":
        acquisition = normalized_utility / np.sqrt(scaled_cost)
    elif method == "log_cost":
        acquisition = normalized_utility / np.log1p(scaled_cost)
    else:
        raise ValueError(f"Unknown method: {method}")
    return utility, acquisition, selected_cost


def choose_batch(
    candidates: np.ndarray,
    acquisition: np.ndarray,
    batch_size: int,
    columns: int,
    hard_cap: bool,
) -> np.ndarray:
    order = np.argsort(-acquisition, kind="stable")
    if not hard_cap:
        return candidates[order[:batch_size]]
    cap = max(1, math.floor(0.35 * batch_size))
    counts = np.zeros(columns, dtype=int)
    selected: list[int] = []
    deferred: list[int] = []
    for position in order:
        flat = int(candidates[position])
        algorithm = flat % columns
        if counts[algorithm] < cap:
            selected.append(flat)
            counts[algorithm] += 1
            if len(selected) == batch_size:
                break
        else:
            deferred.append(flat)
    if len(selected) < batch_size:
        selected.extend(deferred[: batch_size - len(selected)])
    return np.asarray(selected, dtype=np.int64)


def audit_summary(
    utility: np.ndarray,
    acquisition: np.ndarray,
    runtime: RuntimePrediction | None,
    selected_positions: np.ndarray,
) -> dict[str, object]:
    finite = np.isfinite(acquisition)
    safe_acquisition = acquisition[finite]
    overlap_count = min(50, utility.size)
    utility_top = set(np.argsort(-utility)[:overlap_count].tolist())
    acquisition_top = set(np.argsort(-acquisition)[:overlap_count].tolist())
    summary: dict[str, object] = {
        "utility_quantiles": quantiles(utility),
        "acquisition_quantiles": quantiles(safe_acquisition),
        "utility_acquisition_spearman": float(pd.Series(utility[finite]).corr(pd.Series(acquisition[finite]), method="spearman")),
        "top50_overlap": len(utility_top.intersection(acquisition_top)) / overlap_count,
        "selected_utility_mean": float(utility[selected_positions].mean()),
        "selected_acquisition_mean": float(acquisition[selected_positions].mean()),
    }
    if runtime is not None:
        summary.update(
            runtime_mean_quantiles=quantiles(runtime.mean),
            runtime_sd_quantiles=quantiles(runtime.sd),
            selected_predicted_runtime_mean=float(runtime.mean[selected_positions].mean()),
            runtime_lower_bound=runtime.lower,
            runtime_upper_bound=runtime.upper,
            predictions_at_lower_bound=int(np.isclose(runtime.mean, runtime.lower).sum()),
            predictions_at_upper_bound=int(np.isclose(runtime.mean, runtime.upper).sum()),
        )
    return summary


def checkpoint_paths(run_dir: Path, percentage: int) -> dict[str, Path]:
    label = f"{percentage:03d}pct"
    return {
        "matrix": run_dir / "strict_score_matrices" / f"strict_score_{label}.csv",
        "cumulative": run_dir / "queried_cells" / "cumulative" / f"queried_{label}.csv",
        "new": run_dir / "queried_cells" / "newly_queried" / f"queried_{label}.csv",
        "candidate": run_dir / "candidate_acquisition_audits" / f"audit_{label}.json",
        "distribution": run_dir / "algorithm_query_distributions" / f"distribution_{label}.csv",
        "reconstruction": run_dir / "reconstruction_metrics" / f"metrics_{label}.json",
        "runtime": run_dir / "runtime_prediction_metrics" / f"metrics_{label}.json",
    }


def query_frame(records: list[dict[str, object]], indices: list[int], columns: int) -> pd.DataFrame:
    selected = [records[index] for index in indices]
    return pd.DataFrame(selected)


def export_checkpoint(
    run_dir: Path,
    percentage: int,
    prediction: np.ndarray,
    row_index: pd.Index,
    records: list[dict[str, object]],
    previous_count: int,
    oracle: HiddenOracle,
    observed: np.ndarray,
) -> dict[str, float]:
    paths = checkpoint_paths(run_dir, percentage)
    atomic_csv(paths["matrix"], pd.DataFrame(prediction, index=row_index, columns=ALGORITHMS))
    cumulative_indices = list(range(len(records)))
    new_indices = list(range(previous_count, len(records)))
    atomic_csv(paths["cumulative"], query_frame(records, cumulative_indices, len(ALGORITHMS)), index=False)
    atomic_csv(paths["new"], query_frame(records, new_indices, len(ALGORITHMS)), index=False)
    counts = pd.Series([record["algorithm"] for record in records]).value_counts().reindex(ALGORITHMS, fill_value=0)
    distribution = pd.DataFrame({"algorithm": ALGORITHMS, "cumulative_count": counts.to_numpy()})
    distribution["cumulative_share"] = distribution["cumulative_count"] / len(records)
    atomic_csv(paths["distribution"], distribution, index=False)
    metrics = oracle.evaluate(prediction, observed)
    metrics.update(percentage=percentage, queried_cells=len(records))
    atomic_json(paths["reconstruction"], metrics)
    return metrics


def load_inputs() -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rmsd_index, rmsd = read_matrix("moad_rmsd_trimmed.csv")
    pb_index, pb = read_matrix("moad_pb_ratio_trimmed.csv")
    runtime_index, runtime = read_matrix("moad_runtime_trimmed.csv")
    if not rmsd_index.equals(pb_index) or not rmsd_index.equals(runtime_index):
        raise ValueError("Input matrix row labels are not aligned")
    if not np.isfinite(runtime).all() or (runtime < 0).any():
        raise ValueError("Runtime matrix must contain finite, nonnegative seconds")
    if not rmsd_index.is_unique or rmsd_index.hasnans:
        raise ValueError("Raw matrices require unique, non-missing complex identifiers")
    features = np.load(FEATURE_CACHE, allow_pickle=False).astype(np.float32)
    if features.ndim != 2 or features.shape != (len(rmsd_index), 256):
        raise ValueError("Expected a finite molecular feature matrix with one row per complex and 256 columns")
    if not np.isfinite(features).all():
        raise ValueError("Molecular features contain non-finite values")
    if not FEATURE_IDS.is_file():
        raise FileNotFoundError(
            f"Missing producer-verified feature-row identifiers: {FEATURE_IDS}. "
            "Shape alone cannot establish cache alignment; do not create this file by guessing raw-matrix order."
        )
    feature_ids = pd.read_csv(FEATURE_IDS, dtype=str, keep_default_na=False)
    if feature_ids.shape[1] != 1 or feature_ids.iloc[:, 0].tolist() != rmsd_index.astype(str).tolist():
        raise ValueError("Feature-row identifier CSV must have one column matching matrix identifiers in exactly the same order")
    descriptors = load_descriptors(rmsd_index)
    folds = load_folds(rmsd_index)
    combined_features = np.concatenate([features, descriptors], axis=1)
    return rmsd_index, strict_score(rmsd, pb), runtime, combined_features, folds


def run_one(
    method: str,
    seed: int,
    fold: int,
    *,
    smoke_test: bool = False,
    resume: bool = True,
    context: dict[str, object] | None = None,
    prepared_inputs: tuple | None = None,
) -> None:
    if method not in METHODS:
        raise ValueError(method)
    row_index, score_truth, runtime_truth, features, fold_labels = (
        load_inputs() if prepared_inputs is None else prepared_inputs
    )
    training = fold_labels != fold
    row_index = row_index[training]
    score_truth = score_truth[training]
    runtime_truth = runtime_truth[training]
    features = features[training]
    if smoke_test:
        score_truth = score_truth[:120]
        runtime_truth = runtime_truth[:120]
        features = features[:120]
        row_index = row_index[:120]
    rows, columns = score_truth.shape
    total = rows * columns
    run_root = (
        OUTPUT / ("smoke" if smoke_test else "runs")
        / method / f"fold{fold}" / f"seed{seed}"
    )
    state_path = run_root / "run_state" / "state.json"
    log_path = run_root / "run_state" / "events.jsonl"
    specification = {
        "method": method, "fold": fold, "seed": seed, "smoke_test": smoke_test,
        "context": context if context is not None else build_run_context(),
    }
    configuration_fingerprint = fingerprint(specification)
    if verify_existing_run(run_root, configuration_fingerprint, resume=resume):
        print(f"Already complete with matching provenance: {run_root}", flush=True)
        return
    atomic_json(run_root / "run_state/run_manifest.json", {
        "schema_version": 1, "configuration_fingerprint": configuration_fingerprint,
        "run": specification,
    })
    oracle = HiddenOracle(score_truth, runtime_truth)
    runtime_model = RuntimeSurrogate(features, columns)
    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        records = state["records"]
        last_checkpoint = int(state["last_checkpoint"])
        round_id = int(state["round_id"])
        pending_round_audits = state.get("pending_round_audits", [])
        if (state.get("method"), state.get("fold"), state.get("seed")) != (method, fold, seed):
            raise RuntimeError("Resume-state job identity does not match the requested run")
        observed_flat = np.asarray([record["flat_index"] for record in records], dtype=np.int64)
        revealed_score, revealed_runtime = oracle.reveal(observed_flat, -1)
        for record, score, runtime in zip(records, revealed_score, revealed_runtime):
            if not np.isclose(float(record["score"]), score) or not np.isclose(float(record["runtime"]), runtime):
                raise RuntimeError("Resume-state oracle values do not match current inputs")
    else:
        observed_flat = warm_start(score_truth.shape, seed)
        revealed_score, revealed_runtime = oracle.reveal(observed_flat, 0)
        records = []
        for flat, score, runtime in zip(observed_flat, revealed_score, revealed_runtime):
            row, column = divmod(int(flat), columns)
            records.append(
                {
                    "query_order": len(records) + 1,
                    "round": 0,
                    "flat_index": int(flat),
                    "row_index": row,
                    "row_label": str(row_index[row]),
                    "algorithm_index": column,
                    "algorithm": ALGORITHMS[column],
                    "utility": math.nan,
                    "acquisition": math.nan,
                    "predicted_runtime_mean": math.nan,
                    "predicted_runtime_sd": math.nan,
                    "score": float(score),
                    "runtime": float(runtime),
                }
            )
        last_checkpoint = 0
        round_id = 0
        pending_round_audits = []

    observed = np.zeros(total, dtype=bool)
    observed[observed_flat] = True
    checkpoint_targets = {percentage: math.ceil(total * percentage / 100.0) for percentage in range(1, 101)}
    checkpoint_targets[100] = total
    previous_checkpoint_count = previous_checkpoint_size(last_checkpoint, checkpoint_targets)
    rng = np.random.default_rng(seed * 100_003 + METHODS.index(method) * 997)
    started = time.time()

    for percentage in range(max(1, last_checkpoint + 1), 101):
        target = checkpoint_targets[percentage]
        round_audits: list[dict[str, object]] = list(pending_round_audits)
        pending_round_audits = []
        while observed.sum() < target:
            round_id += 1
            score_mean, score_sd = score_ensemble(score_truth, observed.reshape(score_truth.shape))
            candidates = np.flatnonzero(~observed)
            candidate_rows, candidate_algorithms = np.unravel_index(candidates, score_truth.shape)
            utility_mean = score_mean[candidate_rows, candidate_algorithms]
            utility_sd = score_sd[candidate_rows, candidate_algorithms]
            prediction = None
            if method in COST_METHODS:
                queried = np.flatnonzero(observed)
                runtime_ledger = {
                    int(record["flat_index"]): float(record["runtime"])
                    for record in records
                }
                observed_runtime = np.asarray(
                    [runtime_ledger[int(index)] for index in queried],
                    dtype=np.float64,
                )
                prediction = runtime_model.predict(
                    queried,
                    observed_runtime,
                    candidates,
                    seed + round_id * 31,
                    3 if method in ENSEMBLE_COST_METHODS else 1,
                    legacy=method == "legacy_direct_cost_mean_ucb",
                )
            cumulative_counts = np.bincount(np.flatnonzero(observed) % columns, minlength=columns)
            utility, acquisition, selected_cost = acquisition_values(
                method,
                utility_mean,
                utility_sd,
                prediction,
                candidate_algorithms,
                cumulative_counts,
                rng,
            )
            batch_size = min(50, target - int(observed.sum()))
            selected_flat = choose_batch(
                candidates,
                acquisition,
                batch_size,
                columns,
                hard_cap=method == "hard_cap_35",
            )
            candidate_lookup = {int(flat): position for position, flat in enumerate(candidates)}
            positions = np.asarray([candidate_lookup[int(flat)] for flat in selected_flat], dtype=int)
            round_audit = audit_summary(utility, acquisition, prediction, positions)
            round_audit.update(round=round_id, selected_cells=batch_size)
            round_audits.append(round_audit)
            revealed_score, revealed_runtime = oracle.reveal(selected_flat, round_id)
            for flat, score, runtime, position in zip(selected_flat, revealed_score, revealed_runtime, positions):
                row, column = divmod(int(flat), columns)
                records.append(
                    {
                        "query_order": len(records) + 1,
                        "round": round_id,
                        "flat_index": int(flat),
                        "row_index": row,
                        "row_label": str(row_index[row]),
                        "algorithm_index": column,
                        "algorithm": ALGORITHMS[column],
                        "utility": float(utility[position]),
                        "acquisition": float(acquisition[position]),
                        "predicted_runtime_mean": float(prediction.mean[position]) if prediction is not None else math.nan,
                        "predicted_runtime_sd": float(prediction.sd[position]) if prediction is not None else math.nan,
                        "score": float(score),
                        "runtime": float(runtime),
                    }
                )
            observed[selected_flat] = True
            state = {
                "version": 1,
                "method": method,
                "fold": fold,
                "seed": seed,
                "round_id": round_id,
                "last_checkpoint": percentage - 1,
                "records": records,
                "pending_round_audits": round_audits,
            }
            atomic_json(state_path, state)
            with log_path.open("a") as handle:
                handle.write(json.dumps({"event": "microbatch", "method": method, "fold": fold, "seed": seed, "round": round_id, "observed": int(observed.sum()), "target_percentage": percentage}) + "\n")

        final_prediction = svd_complete(score_truth, observed.reshape(score_truth.shape), rank=3)
        metrics = export_checkpoint(
            run_root,
            percentage,
            final_prediction,
            row_index,
            records,
            previous_checkpoint_count,
            oracle,
            observed.reshape(score_truth.shape),
        )
        paths = checkpoint_paths(run_root, percentage)
        atomic_json(paths["candidate"], {"percentage": percentage, "rounds": round_audits})
        queried_runtime = np.asarray([record["runtime"] for record in records], dtype=float)
        runtime_metrics = {
            "percentage": percentage,
            "queried_cells": len(records),
            "cumulative_true_runtime": float(queried_runtime.sum()),
            "mean_true_runtime": float(queried_runtime.mean()),
        }
        predicted = np.asarray([record["predicted_runtime_mean"] for record in records], dtype=float)
        valid = np.isfinite(predicted)
        if valid.any():
            runtime_metrics.update(
                prediction_mae=float(mean_absolute_error(queried_runtime[valid], predicted[valid])),
                prediction_rmse=float(mean_squared_error(queried_runtime[valid], predicted[valid]) ** 0.5),
                prediction_spearman=float(pd.Series(queried_runtime[valid]).corr(pd.Series(predicted[valid]), method="spearman")),
            )
        atomic_json(paths["runtime"], runtime_metrics)
        previous_checkpoint_count = target
        state = {
            "version": 1,
            "method": method,
            "fold": fold,
            "seed": seed,
            "round_id": round_id,
            "last_checkpoint": percentage,
            "records": records,
            "pending_round_audits": [],
            "last_metrics": metrics,
        }
        atomic_json(state_path, state)
        with log_path.open("a") as handle:
            handle.write(json.dumps({"event": "checkpoint", "method": method, "fold": fold, "seed": seed, "percentage": percentage, "observed": len(records), "elapsed_seconds": time.time() - started}) + "\n")

    atomic_json(
        run_root / "run_state" / "complete.json",
        {
            "method": method,
            "fold": fold,
            "seed": seed,
            "checkpoints": 100,
            "configuration_fingerprint": configuration_fingerprint,
            "queried_cells": len(records),
            "oracle_accesses": len(oracle.access_log),
            "elapsed_seconds": time.time() - started,
        },
    )


def input_files() -> dict[str, Path]:
    return {
        "rmsd": DATA / "moad_rmsd_trimmed.csv",
        "pb_ratio": DATA / "moad_pb_ratio_trimmed.csv",
        "runtime": DATA / "moad_runtime_trimmed.csv",
        "features": FEATURE_CACHE, "feature_ids": FEATURE_IDS,
        "descriptors": DESCRIPTORS, "fold_split": FOLD_SPLIT,
    }


def build_run_context() -> dict[str, object]:
    return {
        "implementation": "publication release; historical source snapshot is separate",
        "input_files": {role: {"filename": path.name, "bytes": path.stat().st_size,
                               "sha256": sha256(path)} for role, path in input_files().items()},
        "code_sha256": {name: sha256(HERE / name) for name in ("run_experiment.py", "release_support.py")},
        "environment": environment_versions(),
        "configuration": {
            "algorithms": ALGORITHMS, "score_lambda": LAMBDA,
            "acquisition_svd_ranks": list(SVD_RANKS), "acquisition_svd_iterations": 4,
            "export_svd_rank": 3, "export_svd_iterations": 6,
            "beta": 1.0, "microbatch_maximum": 50,
            "checkpoints": list(range(1, 101)), "warm_start_fraction": 0.01,
            "runtime_surrogate": {"family": "MLPRegressor", "hidden_layers": [32, 16],
                                  "max_iterations_per_refit": 10, "target": "log1p(runtime)",
                                  "observed_runtime_clip_quantiles": [0.05, 0.95]},
            "feature_order": "producer-verified ID file required; exact ordered-ID match validated",
        },
    }


def write_manifest(args: argparse.Namespace) -> dict[str, object]:
    context = build_run_context()
    invocation_id = uuid.uuid4().hex
    atomic_json(OUTPUT / "invocations" / f"{invocation_id}.json", {
        "schema_version": 1, "invocation_id": invocation_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": invocation_arguments(args), "context": context,
    })
    return context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=[0, 1, 2, 3, 4])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh; refuses an existing run directory.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without fitting models or acquiring cells.")
    add_path_arguments(parser)
    args = parser.parse_args()
    configure_paths(args)
    prepared_inputs = load_inputs()
    if args.validate_only:
        print(json.dumps({"status": "pass", "complexes": len(prepared_inputs[0]), "algorithms": len(ALGORITHMS)}))
        return
    context = write_manifest(args)
    for fold in args.folds:
        for seed in args.seeds:
            for method in args.methods:
                run_one(
                    method,
                    seed,
                    fold,
                    smoke_test=args.smoke_test,
                    resume=not args.no_resume,
                    context=context,
                    prepared_inputs=prepared_inputs,
                )


if __name__ == "__main__":
    main()
