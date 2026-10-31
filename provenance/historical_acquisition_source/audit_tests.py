#!/usr/bin/env python3
"""Structural and hidden-runtime invariance tests."""

import numpy as np

from run_experiment import METHODS, RuntimeSurrogate, acquisition_values, warm_start


def main() -> None:
    first = warm_start((2543, 7), 0)
    second = warm_start((2543, 7), 0)
    assert np.array_equal(first, second)
    counts = np.bincount(first % 7, minlength=7)
    assert counts.max() - counts.min() <= 1

    rng = np.random.default_rng(12)
    features = rng.normal(size=(30, 12)).astype(np.float32)
    observed = np.arange(21, dtype=np.int64)
    observed_runtime = rng.lognormal(size=21)
    candidates = np.arange(21, 210, dtype=np.int64)
    surrogate = RuntimeSurrogate(features)
    prediction_a = surrogate.predict(observed, observed_runtime, candidates, 3, 3)
    prediction_b = surrogate.predict(observed, observed_runtime, candidates, 3, 3)
    assert np.allclose(prediction_a.mean, prediction_b.mean)

    mean = rng.normal(size=len(candidates))
    sd = np.abs(rng.normal(size=len(candidates)))
    algorithms = candidates % 7
    counts = np.full(7, 3)
    for method in METHODS:
        utility_a, acquisition_a, _ = acquisition_values(
            method, mean, sd, None if method == "plain_mean_ucb" else prediction_a,
            algorithms, counts, np.random.default_rng(9),
        )
        utility_b, acquisition_b, _ = acquisition_values(
            method, mean, sd, None if method == "plain_mean_ucb" else prediction_b,
            algorithms, counts, np.random.default_rng(9),
        )
        assert np.allclose(utility_a, utility_b)
        assert np.allclose(acquisition_a, acquisition_b)
        assert np.isfinite(acquisition_a).all()
    print("audit tests: PASS")


if __name__ == "__main__":
    main()
