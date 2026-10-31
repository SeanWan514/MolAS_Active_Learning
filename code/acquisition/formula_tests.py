#!/usr/bin/env python3
"""Unit checks for the four agreed acquisition formulations."""

import numpy as np

from run_experiment import RuntimePrediction, acquisition_values


def main() -> None:
    mean = np.array([0.2, 0.4, 0.6, 0.8])
    score_sd = np.array([0.1, 0.1, 0.1, 0.1])
    runtime = RuntimePrediction(
        mean=np.array([1.0, 2.0, 3.0, 4.0]),
        sd=np.array([0.4, 0.3, 0.2, 0.1]),
        conservative=np.array([1.4, 2.3, 3.2, 4.1]),
        lower=1.0,
        upper=5.0,
    )
    rng = np.random.default_rng(0)
    for method in ["direct_cost_uc", "runtime_uncertainty", "cost_tiers_5", "relaxed_cost_power_05", "log_cost"]:
        utility, acquisition, selected_cost = acquisition_values(
            method, mean, score_sd, runtime, np.arange(4), np.ones(7), rng,
        )
        assert np.isfinite(utility).all()
        assert np.isfinite(acquisition).all()
        assert (acquisition > 0).all()
        if method == "runtime_uncertainty":
            assert np.array_equal(selected_cost, runtime.conservative)
        else:
            assert np.array_equal(selected_cost, runtime.mean)
    print("formula tests: PASS")


if __name__ == "__main__":
    main()
