#!/usr/bin/env python3
"""Package saved CSV outputs without training, evaluating models, or plotting.

This is a release-preparation utility written on 2026-10-31. It is NOT the
historical MolAS trainer/evaluator or the paper's plotting implementation.
Only locally resident regular files are opened. macOS iCloud placeholders
remain explicit inventory gaps and can be included by rerunning after download.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import zipfile
from collections import Counter, defaultdict

POLICIES = (
    "plain_mean_ucb", "direct_cost_uc", "relaxed_cost_power_05",
    "log_cost", "cost_tiers_5", "runtime_uncertainty",
)
SF_DATALESS = 0x40000000


def file_kind(name: str) -> str:
    if name == "overall_summary_metrics.csv":
        return "overall_endpoint_summary"
    if name.endswith("_predicted_scores.csv"):
        return "aggregate_predictions" if name.startswith("overall_") else "fold_predictions"
    if name.endswith("_descriptive.csv"):
        return "per_complex_outcomes"
    if name.endswith("_full.csv"):
        return "all_algorithm_endpoint_metrics"
    if name.endswith("_algorithm_selection_counts.csv"):
        return "algorithm_selection_counts"
    if name.startswith("best-"):
        return "selected_endpoint_metrics"
    return "other_csv"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="The original port_as_results_260829 directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="Repository results/downstream directory")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if (not source.is_dir() or source == output or source in output.parents
            or output in source.parents):
        raise SystemExit("Source must exist and output must be outside the source tree")
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    coverage: dict = defaultdict(lambda: {
        "present_fold_directories": set(), "file_count": 0, "packaged_files": 0,
        "missing_files": 0, "source_bytes": 0, "packaged_bytes": 0,
        "packaged_fold_metrics": set(), "packaged_fold_predictions": set(),
        "packaged_fold_descriptive": set(),
    })
    policy_summaries = []
    excluded = sorted(p.name for p in source.iterdir() if p.is_dir() and p.name not in POLICIES)

    for policy in POLICIES:
        root = source / policy
        if not root.is_dir():
            raise SystemExit(f"Missing policy directory: {policy}")
        archive_name = f"{policy}_seed0.zip"
        temporary = output / f".{archive_name}.tmp"
        counts: Counter = Counter()
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for base, dirs, filenames in os.walk(root, followlinks=False):
                dirs[:] = sorted(d for d in dirs if not (Path(base) / d).is_symlink())
                for filename in sorted(filenames):
                    if not filename.endswith(".csv"):
                        continue
                    path = Path(base) / filename
                    rel = path.relative_to(source)
                    parts = rel.parts
                    if len(parts) < 6 or parts[1] != "seed0":
                        raise SystemExit(f"Unexpected saved-output path; do not infer seed: {rel}")
                    checkpoint = parts[2]
                    fold = next((p.removeprefix("fold_") for p in parts if p.startswith("fold_")), "")
                    metadata = path.lstat()
                    flags = getattr(metadata, "st_flags", 0)
                    row = {
                        "policy": policy, "acquisition_seed_label": "0",
                        "checkpoint_label": checkpoint, "fold": fold,
                        "kind": file_kind(filename),
                        "source_workspace_relative_path": f"port_as_results_260829/{rel.as_posix()}",
                        "source_bytes": metadata.st_size, "source_flags": flags,
                        "archive": archive_name, "archive_member": rel.as_posix(),
                        "status": "", "sha256": "", "csv_rows": "",
                        "csv_columns": "", "summary_mirror": "",
                    }
                    item = coverage[(policy, checkpoint)]
                    item["file_count"] += 1
                    item["source_bytes"] += metadata.st_size
                    if fold:
                        item["present_fold_directories"].add(fold)
                    counts["source_files"] += 1
                    counts["source_bytes"] += metadata.st_size
                    if not stat.S_ISREG(metadata.st_mode):
                        row["status"] = "not_regular_file"
                    elif flags & SF_DATALESS:
                        row["status"] = "icloud_dataless_not_opened"
                    elif metadata.st_size > 2_000_000:
                        row["status"] = "exceeds_bounded_csv_read_limit"
                    else:
                        # Metadata was checked before opening: this utility never
                        # deliberately opens an iCloud placeholder or large binary.
                        payload = path.read_bytes()
                        if len(payload) != metadata.st_size:
                            raise RuntimeError(f"Source changed while reading: {rel}")
                        row["sha256"] = hashlib.sha256(payload).hexdigest()
                        try:
                            reader = csv.reader(io.StringIO(payload.decode("utf-8-sig")))
                            columns = next(reader)
                            row["csv_columns"] = json.dumps(columns, ensure_ascii=False)
                            row["csv_rows"] = sum(1 for _ in reader)
                        except (UnicodeError, StopIteration, csv.Error) as error:
                            raise RuntimeError(f"Cannot inspect saved CSV {rel}: {error}") from error
                        info = zipfile.ZipInfo(rel.as_posix(), date_time=(2026, 10, 31, 0, 0, 0))
                        info.compress_type = zipfile.ZIP_DEFLATED
                        info.external_attr = 0o100644 << 16
                        archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED,
                                         compresslevel=6)
                        row["status"] = "packaged_byte_for_byte"
                        counts["packaged_files"] += 1
                        counts["packaged_bytes"] += len(payload)
                        item["packaged_files"] += 1
                        item["packaged_bytes"] += len(payload)
                        if fold and row["kind"] == "selected_endpoint_metrics":
                            item["packaged_fold_metrics"].add(fold)
                        if fold and row["kind"] == "fold_predictions":
                            item["packaged_fold_predictions"].add(fold)
                        if fold and row["kind"] == "per_complex_outcomes":
                            item["packaged_fold_descriptive"].add(fold)
                        if row["kind"] in {"overall_endpoint_summary", "selected_endpoint_metrics",
                                           "all_algorithm_endpoint_metrics"}:
                            mirror = Path("summary_tables") / rel
                            destination = output / mirror
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            destination.write_bytes(payload)
                            row["summary_mirror"] = mirror.as_posix()
                    if row["status"] != "packaged_byte_for_byte":
                        item["missing_files"] += 1
                        counts["missing_files"] += 1
                        counts["missing_bytes"] += metadata.st_size
                    entries.append(row)
        destination = output / archive_name
        temporary.replace(destination)
        counts["archive_bytes"] = destination.stat().st_size
        for name in ("source_files", "source_bytes", "packaged_files", "packaged_bytes", "missing_files", "missing_bytes"):
            counts.setdefault(name, 0)
        archive_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
        policy_summaries.append({"policy": policy, **dict(counts),
                                 "archive": archive_name, "archive_sha256": archive_hash})
        print(json.dumps(policy_summaries[-1]), flush=True)

    fields = list(entries[0])
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(entries)
    with (output / "file_manifest.csv.gz").open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(csv_buffer.getvalue().encode("utf-8"))

    coverage_rows = []
    for (policy, checkpoint), values in sorted(coverage.items()):
        row = {"policy": policy, "acquisition_seed_label": 0, "checkpoint_label": checkpoint}
        for key, value in values.items():
            row[key] = ",".join(sorted(value)) if isinstance(value, set) else value
        row["all_saved_files_packaged"] = values["missing_files"] == 0
        coverage_rows.append(row)
    write_csv(output / "checkpoint_coverage.csv", coverage_rows, list(coverage_rows[0]))
    # A later packaging pass can see a different set of resident iCloud files.
    # Remove stale generated mirrors so this directory agrees with the manifest.
    current_mirrors = {row["summary_mirror"] for row in entries if row["summary_mirror"]}
    for mirror in (output / "summary_tables").rglob("*.csv"):
        if mirror.relative_to(output).as_posix() not in current_mirrors:
            mirror.unlink()
    summary = {
        "packaging_date": "2026-10-31", "operation": "catalog_and_copy_existing_outputs_only",
        "artifact_type": "historical saved-output CSV archives",
        "archives_contain_training_or_plotting_source": False,
        "source_directory_name": source.name,
        "acquisition_seed_labels": [0],
        "selector_training_seed": "not established by these files",
        "fold_directory_labels": [0, 1, 2, 3, 4],
        "checkpoint_unit": "percentage of acquired cells; not realized hours",
        "policies": policy_summaries,
        "excluded_nonpaper_directories": excluded,
        "all_saved_files_packaged": all(p.get("missing_files", 0) == 0 for p in policy_summaries),
        "cloud_policy": "Dataless placeholders are inventoried but not read; rerun after hydration",
        "scientific_results_modified": False,
        "new_scientific_experiments": False,
    }
    (output / "archive_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"complete": summary["all_saved_files_packaged"],
                      "inventory_files": len(entries), "checkpoints": len(coverage_rows)}))


if __name__ == "__main__":
    main()
