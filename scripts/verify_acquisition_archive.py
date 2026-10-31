#!/usr/bin/env python3
"""Verify every stored member and all 27,000 logical original CSV identities."""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from read_acquisition_artifact import git_blob_id


def read_rows(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    stored = collections.defaultdict(dict)
    logical = collections.defaultdict(list)
    for r in read_rows(args.root / "provenance/acquisition_stored_members.jsonl.gz"):
        stored[r["archive"]][r["member"]] = r
    for r in read_rows(args.root / "provenance/pinned_acquisition_files.jsonl.gz"):
        logical[r["archive"]].append(r)

    def check(archive):
        expected = stored[archive]
        by_member = collections.defaultdict(list)
        for r in logical[archive]:
            by_member[r["member"]].append(r)
        seen = set()
        checked = 0
        with tarfile.open(args.root / archive, "r:xz") as tf:
            for member in tf:
                if member.name in seen or member.name not in expected or not member.isfile():
                    raise ValueError("Unexpected or duplicate archive member: " + member.name)
                seen.add(member.name)
                f = tf.extractfile(member)
                raw = f.read()
                rec = expected[member.name]
                if len(raw) != rec["bytes"] or hashlib.sha256(raw).hexdigest() != rec["sha256"]:
                    raise ValueError("Stored content mismatch: " + member.name)
                lines = None
                for r in by_member[member.name]:
                    value = raw
                    if r.get("encoding") == "final_ledger_prefix":
                        if lines is None:
                            lines = raw.splitlines(keepends=True)
                        value = b"".join(lines[:r["data_rows"] + 1])
                    if (len(value) != r["bytes"] or git_blob_id(value) != r["git_blob"]
                            or hashlib.sha256(value).hexdigest() != r["sha256"]):
                        raise ValueError("Original CSV identity mismatch: " + r["path"])
                    checked += 1
        if seen != set(expected):
            raise ValueError("Missing archive members: " + archive)
        return {"archive": archive, "stored_members_verified": len(seen),
                "original_csv_identities_verified": checked}

    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as ex:
        checks = list(ex.map(check, sorted(stored)))
    report = {"status": "pass", "archives": len(checks),
              "stored_members_verified": sum(r["stored_members_verified"] for r in checks),
              "original_csv_identities_verified": sum(r["original_csv_identities_verified"] for r in checks),
              "new_experiments_executed": False, "details": checks}
    if report["original_csv_identities_verified"] != 27000 or report["archives"] != 90:
        raise ValueError("Unexpected archive coverage")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
