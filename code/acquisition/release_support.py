"""Portable paths and provenance safeguards; no scientific computations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]


def add_path_arguments(parser: argparse.ArgumentParser, *, inputs: bool = True) -> None:
    if inputs:
        parser.add_argument("--data-dir", type=Path, default=REPOSITORY / "data/raw")
        parser.add_argument("--features", type=Path, default=REPOSITORY / "data/features/complex_features_ligand_protein_d256_seed0.npy")
        parser.add_argument("--feature-ids", type=Path, default=REPOSITORY / "data/features/complex_feature_ids.csv",
                            help="Required one-column CSV of producer-verified IDs in feature-row order.")
        parser.add_argument("--descriptors", type=Path, default=REPOSITORY / "data/descriptors/moad_physical_descriptors_filtered.csv")
        parser.add_argument("--fold-split", "--fold-file", dest="fold_split", type=Path, default=REPOSITORY / "data/splits/moad_5fold_split_seed42.csv")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "outputs/acquisition")


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def environment_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for distribution in ("numpy", "pandas", "scikit-learn"):
        versions[distribution] = importlib.metadata.version(distribution)
    return versions


def previous_checkpoint_size(last_checkpoint: int, checkpoint_targets: dict[int, int]) -> int:
    """Start the next incremental export after the last COMPLETED checkpoint."""
    if last_checkpoint == 0:
        return 0
    if last_checkpoint not in checkpoint_targets:
        raise ValueError(f"Unknown completed checkpoint: {last_checkpoint}")
    return checkpoint_targets[last_checkpoint]


def verify_existing_run(run_root: Path, expected_fingerprint: str, *, resume: bool) -> bool:
    """Check identity before reading state, skipping completion, or writing outputs.

    Return True only for a completed run with matching release provenance.
    Historical outputs without this metadata are deliberately not modified.
    """
    manifest_path = run_root / "run_state/run_manifest.json"
    if not run_root.exists() or not any(run_root.iterdir()):
        return False
    if not resume:
        raise FileExistsError(f"Output run already exists: {run_root}. Use a fresh --output-dir.")
    if not manifest_path.exists():
        raise RuntimeError(f"Existing run lacks release provenance: {run_root}. Keep historical outputs unchanged and use a fresh --output-dir.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("configuration_fingerprint") != expected_fingerprint:
        raise RuntimeError(f"Inputs, code, environment, or run configuration changed: {run_root}. Use a fresh --output-dir.")
    marker = run_root / "run_state/complete.json"
    if not marker.exists():
        return False
    complete = json.loads(marker.read_text())
    if complete.get("configuration_fingerprint") != expected_fingerprint or complete.get("checkpoints") != 100:
        raise RuntimeError(f"Invalid completion marker: {marker}")
    final_paths = (
        "strict_score_matrices/strict_score_100pct.csv",
        "queried_cells/cumulative/queried_100pct.csv",
        "queried_cells/newly_queried/queried_100pct.csv",
    )
    if not all((run_root / path).is_file() for path in final_paths):
        raise RuntimeError(f"Completion marker exists but final exports are missing: {run_root}")
    return True


def invocation_arguments(args: argparse.Namespace) -> dict[str, object]:
    """Record selected jobs; paths and environment variables are not serialized."""
    return {
        "methods": list(args.methods), "folds": list(args.folds), "seeds": list(args.seeds),
        "smoke_test": bool(getattr(args, "smoke_test", False)),
        "resume": not bool(getattr(args, "no_resume", False)),
    }
