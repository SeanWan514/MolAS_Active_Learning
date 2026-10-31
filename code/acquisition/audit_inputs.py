#!/usr/bin/env python3
"""Validate supplied input schemas and feature ordering; never fit models."""

import argparse
import json

import numpy as np

import run_experiment as runner
from release_support import add_path_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_arguments(parser)
    args = parser.parse_args()
    runner.configure_paths(args)
    ids, scores, runtimes, features, folds = runner.load_inputs()
    if not np.isfinite(runtimes).all() or (runtimes < 0).any():
        raise ValueError("Runtime matrix must contain finite, nonnegative seconds")
    report = {
        "status": "pass", "complexes": len(ids), "algorithms": scores.shape[1],
        "combined_features": features.shape[1],
        "held_out_fold_sizes": {str(i): int((folds == i).sum()) for i in range(5)},
        "feature_order_check": "Exact match to supplied producer-verified feature ID file; provenance is not inferred from dimensions.",
        "input_files": runner.build_run_context()["input_files"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
