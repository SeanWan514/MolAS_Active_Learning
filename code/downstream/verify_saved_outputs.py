#!/usr/bin/env python3
"""Check saved-output archive integrity without running scientific analysis."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import zipfile


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root: Path, original_archive_manifest: Path | None = None) -> dict:
    root = root.resolve()

    def local_path(name: str) -> Path:
        candidate = (root / name).resolve()
        require(candidate.is_relative_to(root), f"Manifest path escapes archive root: {name}")
        return candidate

    summary = json.loads((root / "archive_summary.json").read_text())
    with gzip.open(root / "file_manifest.csv.gz", "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    packaged = [row for row in rows if row["status"] == "packaged_byte_for_byte"]
    verified = 0
    expected_archive_names = {policy["archive"] for policy in summary["policies"]}
    require({path.name for path in root.glob("*.zip")} == expected_archive_names,
            "ZIP files differ from the archive summary")

    for policy in summary["policies"]:
        archive_name = policy["archive"]
        archive_path = local_path(archive_name)
        matching = [row for row in packaged if row["archive"] == archive_name]
        expected = {row["archive_member"]: row for row in matching}
        require(len(expected) == len(matching), f"Duplicate manifest members: {archive_name}")
        require(len(expected) == policy["packaged_files"], f"Incorrect packaged count: {archive_name}")
        require(archive_path.stat().st_size == policy["archive_bytes"], f"Archive size mismatch: {archive_name}")
        require(sha256(archive_path) == policy["archive_sha256"], f"Archive hash mismatch: {archive_name}")
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), f"Duplicate ZIP members: {archive_name}")
            require(set(names) == set(expected), f"ZIP member set mismatch: {archive_name}")
            for name in names:
                # Reading the entire member also invokes ZipFile's CRC check.
                payload = archive.read(name)
                row = expected[name]
                require(len(payload) == int(row["source_bytes"]), f"Payload size mismatch: {name}")
                require(hashlib.sha256(payload).hexdigest() == row["sha256"], f"Payload hash mismatch: {name}")
                verified += 1

    require(verified == len(packaged), "Some packaged manifest rows were not verified")
    mirrors = [row for row in packaged if row["summary_mirror"]]
    expected_mirrors = {row["summary_mirror"] for row in mirrors}
    require(len(expected_mirrors) == len(mirrors), "Duplicate mirror paths in manifest")
    actual_mirrors = {path.relative_to(root).as_posix() for path in (root / "summary_tables").rglob("*.csv")}
    require(actual_mirrors == expected_mirrors, "Missing or stale generated summary mirrors")
    for row in mirrors:
        require(sha256(local_path(row["summary_mirror"])) == row["sha256"],
                f"Summary mirror hash mismatch: {row['summary_mirror']}")
    require(sum(policy["source_files"] for policy in summary["policies"]) == len(rows),
            "Original file count differs between summary and manifest")
    missing = len(rows) - verified
    require(sum(policy["missing_files"] for policy in summary["policies"]) == missing,
            "Missing file count differs between summary and manifest")
    require(summary["all_saved_files_packaged"] == (missing == 0), "Incorrect completeness flag")
    original_members_verified = None
    if original_archive_manifest is not None:
        original = json.loads(original_archive_manifest.read_text())
        source_members = {item["member"]: item for item in original["files"]}
        require(len(source_members) == len(original["files"]), "Duplicate original archive members")
        prefix = summary["source_directory_name"] + "/"
        require({prefix + row["archive_member"] for row in rows} == set(source_members),
                "Original archive and packaged inventory describe different source CSVs")
        for row in packaged:
            member = prefix + row["archive_member"]
            source = source_members[member]
            require(row["sha256"] == source["sha256"] and int(row["source_bytes"]) == source["bytes"],
                    f"Original archive member identity mismatch: {member}")
        original_members_verified = len(packaged)
    return {
        "scope": "release_archive_integrity_only_not_scientific_reproduction",
        "original_csv_files_in_manifest": len(rows),
        "verified_zip_members": verified,
        "verified_summary_mirrors": len(mirrors),
        "zip_crc_ok": True,
        "all_packaged_member_sha256_matches": True,
        "archive_sha256_matches": True,
        "stale_summary_mirrors": 0,
        "missing_original_csv_payloads": missing,
        "all_original_csv_payloads_packaged": missing == 0,
        "verified_original_archive_member_hashes": original_members_verified,
        "new_scientific_experiments": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results/downstream")
    parser.add_argument("--write-report", action="store_true",
                        help="Write verification.json after successful integrity checks")
    parser.add_argument("--require-complete", action="store_true",
                        help="Return nonzero if any original CSV payload remains unavailable")
    parser.add_argument("--original-archive-manifest", type=Path,
                        help="Also compare member identities to the original archive recovery manifest")
    args = parser.parse_args()
    report = validate(args.output_dir, args.original_archive_manifest)
    if args.write_report:
        (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_complete and not report["all_original_csv_payloads_packaged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
