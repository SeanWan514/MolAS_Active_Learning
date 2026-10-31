#!/usr/bin/env python3
"""Copy readable historical acquisition artifacts without running experiments.

Cloud placeholders are inventoried, never mistaken for readable files. Original
CSVs remain byte-for-byte unchanged inside deterministic per-run ZIP archives.
This utility is for assembling the release from the authors' historical workspace;
it is not required to inspect archives already shipped in the repository.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

METHODS = ("plain_mean_ucb", "direct_cost_uc", "relaxed_cost_power_05",
           "log_cost", "runtime_uncertainty", "cost_tiers_5")
DATALESS = 0x40000000


def cloud_placeholder(path: Path) -> bool:
    return bool(getattr(path.stat(), "st_flags", 0) & DATALESS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    base = args.source_root / "MolAS_AL_restart/AL_MolAS/mean_ucb_cost_5fold_paper"
    dest = args.destination.resolve()
    out = dest / "results/acquisition"
    out.mkdir(parents=True, exist_ok=True)
    (dest / "provenance").mkdir(parents=True, exist_ok=True)
    records = []
    totals = collections.Counter()

    def archive(paths, target, prefix):
        target.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as z:
            for p in sorted(paths):
                if not p.is_file() or p.name == ".DS_Store":
                    continue
                rel = p.relative_to(base).as_posix()
                rec = {"source": str(p.relative_to(args.source_root)),
                       "archive": str(target.relative_to(dest)),
                       "member": p.relative_to(prefix).as_posix(),
                       "bytes": p.stat().st_size}
                if p.is_symlink():
                    rec["status"] = "excluded_symlink"
                elif cloud_placeholder(p):
                    rec["status"] = "missing_cloud_contents"
                else:
                    try:
                        raw = p.read_bytes()
                        rec["sha256"] = hashlib.sha256(raw).hexdigest()
                        zi = zipfile.ZipInfo(rec["member"], (2026, 10, 31, 0, 0, 0))
                        zi.compress_type = zipfile.ZIP_DEFLATED
                        zi.external_attr = 0o644 << 16
                        z.writestr(zi, raw, compresslevel=6)
                        rec["status"] = "included_unchanged"
                        count += 1
                        totals["included_uncompressed_bytes"] += len(raw)
                    except OSError as exc:
                        rec["status"] = "unreadable"
                        rec["error"] = type(exc).__name__
                totals[rec["status"]] += 1
                records.append(rec)
        if count == 0:
            target.unlink()
        else:
            totals["archive_bytes"] += target.stat().st_size
            totals["archives"] += 1

    for method in METHODS:
        for fold in range(5):
            for seed in range(3):
                run = base / "outputs/runs" / method / f"fold{fold}" / f"seed{seed}"
                # State binaries are not needed to inspect published observations
                # or exported matrices. Keep small JSON metadata and event logs.
                paths = [p for p in run.rglob("*") if p.is_file()
                         and p.suffix in (".csv", ".json", ".jsonl", ".txt")]
                archive(paths, out / method / f"fold{fold}_seed{seed}.zip", run)
        print(f"Packaged available artifacts: {method}", flush=True)
    auxiliary = []
    for name in ("audits", "summary", "runtime_feature_ablation"):
        auxiliary.extend((base / "outputs" / name).rglob("*"))
    auxiliary.extend(p for p in (base / "outputs").iterdir() if p.is_file())
    archive(auxiliary, out / "historical_summaries.zip", base / "outputs")
    manifest = dest / "provenance/acquisition_artifact_inventory.jsonl.gz"
    with manifest.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as f:
            for rec in records:
                f.write((json.dumps(rec, sort_keys=True) + "\n").encode())
    summary = dict(totals)
    summary.update({"methods": list(METHODS), "folds": list(range(5)),
                    "acquisition_seeds": [0, 1, 2],
                    "complete": not any(totals[k] for k in
                        ("missing_cloud_contents", "unreadable", "excluded_symlink")),
                    "note": "Historical files only; no experiments or results regenerated. "
                            "Downstream paper analyses use acquisition seed 0. "
                            "Resume-state binaries and smoke/pilot runs are excluded."})
    (dest / "provenance/acquisition_archive_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
