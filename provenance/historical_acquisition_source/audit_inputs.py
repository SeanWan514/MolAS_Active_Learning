#!/usr/bin/env python3
"""Fail-fast alignment and fold-isolation audit for the paper experiment."""

import json
from pathlib import Path

import numpy as np

from run_experiment import ALGORITHMS, HERE, load_inputs, sha256, DATA, DESCRIPTORS, FOLD_SPLIT, FEATURE_CACHE


def main() -> None:
    output = HERE / "outputs" / "audits" / "input_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        rows, scores, runtimes, features, folds = load_inputs()
    except Exception as error:
        output.write_text(json.dumps({
            "status": "blocked",
            "error_type": type(error).__name__,
            "error": str(error),
        }, indent=2))
        raise
    report = {
        "status": "pass",
        "complexes": len(rows),
        "algorithms": ALGORITHMS,
        "score_shape": list(scores.shape),
        "runtime_shape": list(runtimes.shape),
        "combined_feature_shape": list(features.shape),
        "fold_counts": {str(fold): int(np.sum(folds == fold)) for fold in range(5)},
        "training_counts": {str(fold): int(np.sum(folds != fold)) for fold in range(5)},
        "finite": {
            "scores": bool(np.isfinite(scores).all()),
            "runtimes": bool(np.isfinite(runtimes).all()),
            "features": bool(np.isfinite(features).all()),
        },
        "positive_runtime": bool((runtimes > 0).all()),
        "sha256": {
            str(path): sha256(path)
            for path in [
                DATA / "moad_rmsd_trimmed.csv",
                DATA / "moad_pb_ratio_trimmed.csv",
                DATA / "moad_runtime_trimmed.csv",
                FEATURE_CACHE,
                DESCRIPTORS,
                FOLD_SPLIT,
            ]
        },
    }
    if not all(report["finite"].values()) or not report["positive_runtime"]:
        raise ValueError(f"Input audit failed: {report}")
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
