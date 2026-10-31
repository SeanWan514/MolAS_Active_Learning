#!/usr/bin/env python3
"""Extract an original acquisition CSV, checking its pinned Git content identity.

Checkpoint cumulative logs are losslessly represented by prefixes of the final
ledger. Every supported prefix has been checked against its original Git blob;
this is storage deduplication, not a repaired or regenerated experiment.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile

METHODS = ("plain_mean_ucb", "direct_cost_uc", "relaxed_cost_power_05", "log_cost",
           "runtime_uncertainty", "cost_tiers_5")


def git_blob_id(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--seed", type=int, choices=range(3), default=0)
    p.add_argument("--checkpoint", type=int, choices=range(1, 101), required=True)
    p.add_argument("--kind", choices=("matrix", "cumulative", "newly"), required=True)
    p.add_argument("--output", type=Path, help="CSV destination; omitted means stdout.")
    args = p.parse_args()
    base = f"{args.method}/fold{args.fold}/seed{args.seed}"
    pct = f"{args.checkpoint:03d}pct.csv"
    member = {"matrix": "strict_score_matrices/strict_score_" + pct,
              "cumulative": "queried_cells/cumulative/queried_" + pct,
              "newly": "queried_cells/newly_queried/queried_" + pct}[args.kind]
    logical = base + "/" + member
    match = None
    with gzip.open(args.root / "provenance/pinned_acquisition_files.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line)
            if row["path"] == logical:
                match = row
                break
    if match is None or match["status"] != "verified":
        raise RuntimeError(f"No verified archived artifact: {logical}")
    with tarfile.open(args.root / match["archive"], "r:xz") as archive:
        source = archive.extractfile(match["member"])
        if source is None:
            raise RuntimeError("Archive member is not a regular file")
        raw = source.read()
    if match.get("encoding") == "final_ledger_prefix":
        raw = b"".join(raw.splitlines(keepends=True)[:match["data_rows"] + 1])
    if git_blob_id(raw) != match["git_blob"] or hashlib.sha256(raw).hexdigest() != match["sha256"]:
        raise RuntimeError(f"Content verification failed: {logical}")
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(raw)
    else:
        sys.stdout.buffer.write(raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
