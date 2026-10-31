#!/usr/bin/env python3
"""Five-fold runtime prediction ablation for learned, physical, and combined inputs."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_experiment import (
    ALGORITHMS, FEATURE_CACHE, HERE, load_descriptors, load_folds, read_matrix,
)


def expanded(features: np.ndarray) -> np.ndarray:
    algorithms = len(ALGORITHMS)
    return np.concatenate([
        np.repeat(features, algorithms, axis=0),
        np.tile(np.eye(algorithms, dtype=np.float32), (features.shape[0], 1)),
    ], axis=1)


def model(seed: int) -> object:
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(32, 16), activation="relu", max_iter=100,
            early_stopping=True, validation_fraction=0.15,
            random_state=seed,
        ),
    )


def main() -> None:
    row_index, runtime = read_matrix("moad_runtime_trimmed.csv")
    learned = np.load(FEATURE_CACHE).astype(np.float32)
    physical = load_descriptors(row_index)
    folds = load_folds(row_index)
    feature_sets = {
        "learned_plus_algorithm": learned,
        "physical_plus_algorithm": physical,
        "combined_plus_algorithm": np.concatenate([learned, physical], axis=1),
    }
    records = []
    for name, features in feature_sets.items():
        x = expanded(features)
        y = np.log1p(runtime.ravel())
        cell_folds = np.repeat(folds, len(ALGORITHMS))
        for fold in range(5):
            train = cell_folds != fold
            test = ~train
            estimator = model(42 + fold)
            estimator.fit(x[train], y[train])
            predicted_log = estimator.predict(x[test])
            predicted = np.maximum(np.expm1(predicted_log), 0.0)
            truth = np.expm1(y[test])
            records.append({
                "feature_set": name,
                "fold": fold,
                "log_mae": mean_absolute_error(y[test], predicted_log),
                "log_rmse": mean_squared_error(y[test], predicted_log) ** 0.5,
                "runtime_mae": mean_absolute_error(truth, predicted),
                "runtime_rmse": mean_squared_error(truth, predicted) ** 0.5,
                "log_r2": r2_score(y[test], predicted_log),
                "spearman": pd.Series(truth).corr(pd.Series(predicted), method="spearman"),
            })
    frame = pd.DataFrame(records)
    output = HERE / "outputs" / "runtime_feature_ablation"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "fold_metrics.csv", index=False)
    summary = frame.groupby("feature_set").agg(["mean", "std"])
    summary.to_csv(output / "summary.csv")
    (output / "complete.json").write_text(json.dumps({"status": "complete", "rows": len(frame)}, indent=2))
    print(summary)


if __name__ == "__main__":
    main()

