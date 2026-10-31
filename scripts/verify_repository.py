#!/usr/bin/env python3
"""Verify release-file hashes and report integrity separately from completeness."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--include-large-data", action="store_true",
                        help="Also hash the approximately 15 GB local structural files.")
    parser.add_argument("--strict-release", action="store_true",
                        help="Return 2 if this GitHub artifact release is not ready for publication.")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "provenance/repository_files.json").read_text())
    expected = list(manifest["files"])
    if args.include_large_data:
        expected.extend(json.loads((root / "provenance/external_graph_checksums.json").read_text())["files"])
    errors = []
    for rec in expected:
        path = root / rec["path"]
        if not path.resolve().is_relative_to(root):
            errors.append({"path": rec["path"], "error": "unsafe manifest path"})
        elif not path.is_file() or path.is_symlink():
            errors.append({"path": rec["path"], "error": "missing file or symlink"})
        elif path.stat().st_size != rec["bytes"] or digest(path) != rec["sha256"]:
            errors.append({"path": rec["path"], "error": "content differs from release manifest"})
    status = json.loads((root / "provenance/release_status.json").read_text())
    report = {"checked_files": len(expected), "integrity_passed": not errors,
              "errors": errors, "publication_ready": status["publication_ready"],
              "full_reproduction_ready": status["full_reproduction_ready"],
              "remaining_items": status["remaining_items"]}
    print(json.dumps(report, indent=2))
    if errors:
        return 1
    if args.strict_release and not status["publication_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
