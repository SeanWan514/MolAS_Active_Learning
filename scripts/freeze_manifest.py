#!/usr/bin/env python3
"""Record the release file identity after an intentional, reviewed update.

This does not validate scientific correctness or change component provenance.
Run component validators first. Large structural files have their own manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true", help="Explicitly replace repository_files.json")
    args = parser.parse_args()
    root = args.root.resolve()
    listed = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    ).decode().split("\0")
    excluded = {"provenance/repository_files.json", "provenance/final_verification.json"}
    records = []
    for name in sorted(set(listed) - excluded - {""}):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Unexpected non-file in release: {name}")
        if getattr(path.stat(), "st_flags", 0) & 0x40000000:
            raise ValueError(f"Cloud placeholder in release: {name}")
        with path.open("rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": sha})
    result = {"format_version": 1,
              "scope": "Git-included files; large local structural files have a separate manifest",
              "excluded_self_generated_files": sorted(excluded),
              "total_files": len(records), "total_bytes": sum(r["bytes"] for r in records),
              "files": records}
    if args.write:
        (root / "provenance/repository_files.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
