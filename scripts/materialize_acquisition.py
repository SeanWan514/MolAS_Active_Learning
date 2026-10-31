#!/usr/bin/env python3
"""Materialize original CSVs for the recovered analysis notebooks; no experiments.

Default: estimate only. --write requires a new or empty destination. Seed 0
is the paper's downstream archive; all three acquisition seeds are available.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from read_acquisition_artifact import METHODS, git_blob_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    p.add_argument("--seeds", nargs="+", type=int, choices=range(3), default=[0])
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    groups = collections.defaultdict(list)
    with gzip.open(args.root / "provenance/pinned_acquisition_files.jsonl.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            method, fold, seed = rec["path"].split("/")[:3]
            if method in args.methods and int(fold[4:]) in args.folds and int(seed[4:]) in args.seeds:
                groups[rec["archive"]].append(rec)
    rows = [r for group in groups.values() for r in group]
    print(json.dumps({"csv_files": len(rows), "expanded_bytes": sum(r["bytes"] for r in rows),
                      "write_requested": args.write, "acquisition_seeds": args.seeds}, indent=2))
    if not args.write:
        return
    dest = args.destination.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("Use a new or empty destination; existing files will not be overwritten.")
    dest.mkdir(parents=True, exist_ok=True)
    for archive, group in sorted(groups.items()):
        needed = {r["member"] for r in group}
        contents = {}
        with tarfile.open(args.root / archive, "r:xz") as tf:
            for item in tf:
                if item.name in needed:
                    f = tf.extractfile(item)
                    if f is None:
                        raise ValueError("Expected a regular CSV member")
                    contents[item.name] = f.read()
        for rec in group:
            raw = contents[rec["member"]]
            if rec.get("encoding") == "final_ledger_prefix":
                raw = b"".join(raw.splitlines(keepends=True)[:rec["data_rows"] + 1])
            if git_blob_id(raw) != rec["git_blob"] or hashlib.sha256(raw).hexdigest() != rec["sha256"]:
                raise ValueError("Archive integrity mismatch: " + rec["path"])
            target = dest / rec["path"]
            if not target.resolve().is_relative_to(dest):
                raise ValueError("Unsafe artifact path")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as f:
                f.write(raw)
        print("Materialized " + archive, flush=True)


if __name__ == "__main__":
    main()
