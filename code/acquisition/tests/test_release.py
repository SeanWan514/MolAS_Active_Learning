"""Fabricated-data checks only: no docking, AL loop, SVD fit, or MLP fit."""

import argparse
import ast
import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

import run_experiment as runner
from finalize_audit import audit_incremental
from release_support import fingerprint, previous_checkpoint_size, verify_existing_run


class NumericalContractTests(unittest.TestCase):
    def test_scientific_source_is_unchanged(self):
        original = HERE.parents[1] / "provenance/historical_acquisition_source/run_experiment.py"
        def definitions(path):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(path.read_text()).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        before, after = definitions(original), definitions(HERE / "run_experiment.py")
        names = ["strict_score", "svd_complete", "score_ensemble", "warm_start", "percentile_rank",
                 "quantiles", "RuntimePrediction", "RuntimeSurrogate", "HiddenOracle",
                 "continuous_cost_scale", "cost_tiers", "acquisition_values", "choose_batch",
                 "audit_summary", "query_frame", "export_checkpoint"]
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(before[name], after[name])

    def test_all_six_acquisition_rules_and_raw_versus_ranked_utility(self):
        mean = np.array([.1, .4, .6, .9])
        costs = np.array([1., 2., 3., 4.])
        conservative = np.array([1.4, 2.3, 3.2, 4.1])
        runtime = runner.RuntimePrediction(costs, conservative-costs, conservative, 1., 5.)
        ranks = np.array([.25, .5, .75, 1.])
        scaled = np.array([1., 7/3, 11/3, 5.])
        expected = {
            "plain_mean_ucb": mean,
            "direct_cost_uc": mean/(costs+1e-6),
            "relaxed_cost_power_05": ranks/np.sqrt(scaled),
            "log_cost": ranks/np.log1p(scaled),
            "cost_tiers_5": ranks/np.array([1., 2., 4., 5.]),
            "runtime_uncertainty": ranks/np.sqrt(scaled),
        }
        for method, wanted in expected.items():
            _, actual, _ = runner.acquisition_values(method, mean, np.zeros(4), runtime,
                                                     np.arange(4), np.ones(7), np.random.default_rng(0))
            np.testing.assert_allclose(actual, wanted, rtol=1e-12, atol=1e-12)

    def test_ties_degenerate_scaling_and_tier_boundaries(self):
        np.testing.assert_array_equal(runner.percentile_rank(np.array([1., 2., 2., 4.])), [.25, .625, .625, 1.])
        np.testing.assert_array_equal(runner.continuous_cost_scale(np.array([2., 2., 2.])), [1., 1., 1.])
        np.testing.assert_array_equal(runner.cost_tiers(np.array([1., 1., 2., 2., 3.]), 5), [1., 1., 3., 3., 5.])

    def test_strict_score_uses_full_pb_validity(self):
        result = runner.strict_score(np.array([0., 0., 2., np.nan]), np.array([1., .999, 1., 1.]))
        np.testing.assert_allclose(result, [1., 0., (1+math.exp(-6))/2, 0.])

    def test_runtime_members_then_sum_are_clipped_without_fitting(self):
        predictions = iter([np.array([0., 5., 30.]), np.array([3., 8., 8.]), np.array([4., 9., 20.])])
        fit_targets = []
        class FakePipeline:
            def __init__(self):
                self.raw = next(predictions)
            def fit(self, x, y):
                fit_targets.append(y.copy())
                return self
            def predict(self, x):
                return np.log1p(self.raw)
        with patch.object(runner, "make_pipeline", side_effect=lambda *args: FakePipeline()):
            model = runner.RuntimeSurrogate(np.zeros((3, 2), dtype=np.float32), algorithms=2)
            result = model.predict(np.array([0, 1]), np.array([2., 10.]), np.array([2, 3, 4]), 0, 3)
        members = np.array([[2.4, 5., 9.6], [3., 8., 8.], [4., 9., 9.6]])
        np.testing.assert_allclose(result.mean, members.mean(axis=0))
        np.testing.assert_allclose(result.sd, members.std(axis=0))
        self.assertGreater(result.mean[-1]+result.sd[-1], 9.6)
        self.assertAlmostEqual(result.conservative[-1], 9.6)
        for y in fit_targets:
            np.testing.assert_array_equal(y, np.log1p([2., 10.]))


class ResumeExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.records = [{"query_order": n+1, "flat_index": n, "algorithm": runner.ALGORITHMS[n % 7]}
                        for n in range(30)]
        self.targets = {1: 10, 2: 20, 3: 30}

    def tearDown(self):
        self.tmp.cleanup()

    def export(self, root, percentage, previous_count):
        class NoModelOracle:
            def evaluate(self, prediction, observed):
                return {}
        runner.export_checkpoint(root, percentage, np.zeros((5, 7)), pd.Index(range(5)),
                                 self.records[:self.targets[percentage]], previous_count,
                                 NoModelOracle(), np.zeros((5, 7), dtype=bool))

    def test_resuming_completed_and_partial_checkpoints_has_no_repeated_block(self):
        for completed in (1, 2):
            root = self.root / str(completed)
            for pct in range(1, 4):
                previous = previous_checkpoint_size(pct-1, self.targets)
                self.export(root, pct, previous)
            self.assertEqual(previous_checkpoint_size(completed, self.targets), self.targets[completed])
            result = audit_incremental(root, checkpoints=3)
            self.assertEqual(result, {"unique_cells": 30, "issues": []})
        # A partial third checkpoint stores last_checkpoint=2 even if 25 rows
        # have already been acquired; its final export starts after row 20.
        self.assertEqual(previous_checkpoint_size(2, self.targets), 20)

    def test_audit_detects_historical_off_by_one_slice(self):
        self.export(self.root, 1, 0)
        self.export(self.root, 2, 10)
        self.export(self.root, 3, 10)  # historical error: should be 20
        result = audit_incremental(self.root, checkpoints=3)
        self.assertTrue(any("repeated flat index" in issue for issue in result["issues"]))
        self.assertEqual(result["unique_cells"], 30)


class ProvenanceTests(unittest.TestCase):
    def test_actual_invocation_scope_and_unique_manifests(self):
        args = argparse.Namespace(methods=["log_cost"], folds=[2], seeds=[7], smoke_test=True, no_resume=False)
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "OUTPUT", Path(temp)), \
                patch.object(runner, "build_run_context", return_value={"fabricated": True}):
            runner.write_manifest(args)
            runner.write_manifest(args)
            files = list((Path(temp)/"invocations").glob("*.json"))
            self.assertEqual(len(files), 2)
            item = json.loads(files[0].read_text())["invocation"]
            self.assertEqual(item["methods"], ["log_cost"])
            self.assertEqual(item["folds"], [2])
            self.assertEqual(item["seeds"], [7])
            self.assertTrue(item["smoke_test"])
            self.assertFalse((Path(temp)/"experiment_manifest.json").exists())

    def test_resume_refuses_legacy_or_changed_inputs_and_fresh_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"run"
            root.mkdir()
            (root/"legacy.csv").write_text("untouched\n")
            with self.assertRaises(RuntimeError):
                verify_existing_run(root, "wanted", resume=True)
            (root/"run_state").mkdir()
            (root/"run_state/run_manifest.json").write_text(json.dumps({"configuration_fingerprint": "old"}))
            with self.assertRaises(RuntimeError):
                verify_existing_run(root, "changed", resume=True)
            with self.assertRaises(FileExistsError):
                verify_existing_run(root, "old", resume=False)
            self.assertFalse(verify_existing_run(root, "old", resume=True))
            self.assertEqual((root/"legacy.csv").read_text(), "untouched\n")

    def test_completion_marker_cannot_skip_missing_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"run_state").mkdir()
            for name in ["run_manifest", "complete"]:
                (root/f"run_state/{name}.json").write_text(json.dumps({"configuration_fingerprint": "same", "checkpoints": 100}))
            with self.assertRaises(RuntimeError):
                verify_existing_run(root, "same", resume=True)

    def test_configuration_fingerprint_changes_with_input_hash(self):
        self.assertNotEqual(fingerprint({"input_sha256": "a"}), fingerprint({"input_sha256": "b"}))


class InputOrderingTests(unittest.TestCase):
    def test_producer_id_order_is_required_and_fractional_folds_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            ids = [f"complex_{n}" for n in range(5)]
            for kind in ["rmsd", "pb_ratio", "runtime"]:
                frame = pd.DataFrame(np.ones((5, 7)), index=ids,
                                     columns=[f"{a}_{kind}" for a in runner.ALGORITHMS])
                frame.to_csv(p/f"moad_{kind}_trimmed.csv")
            np.save(p/"features.npy", np.zeros((5, 256), dtype=np.float32))
            pd.DataFrame({"complex_id": ids, "d1": 1., "d2": 2., "d3": 3., "d4": 4.}).to_csv(p/"descriptors.csv", index=False)
            pd.DataFrame({"complex_id": ids, "fold": range(5)}).to_csv(p/"folds.csv", index=False)
            with patch.multiple(runner, DATA=p, FEATURE_CACHE=p/"features.npy", FEATURE_IDS=p/"feature_ids.csv",
                                DESCRIPTORS=p/"descriptors.csv", FOLD_SPLIT=p/"folds.csv"):
                with self.assertRaises(FileNotFoundError):
                    runner.load_inputs()
                pd.DataFrame({"complex_id": ids[::-1]}).to_csv(p/"feature_ids.csv", index=False)
                with self.assertRaises(ValueError):
                    runner.load_inputs()
                pd.DataFrame({"complex_id": ids}).to_csv(p/"feature_ids.csv", index=False)
                loaded = runner.load_inputs()
                self.assertEqual(loaded[3].shape, (5, 260))
                pd.DataFrame({"complex_id": ids, "fold": [0, 1, 2, 3, 3.9]}).to_csv(p/"folds.csv", index=False)
                with self.assertRaises(ValueError):
                    runner.load_inputs()


if __name__ == "__main__":
    unittest.main()
