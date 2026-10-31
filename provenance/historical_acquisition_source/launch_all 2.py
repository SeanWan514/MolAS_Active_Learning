#!/usr/bin/env python3
"""Resumable launcher for all 90 fold/seed/method runs."""

import json
import subprocess
import sys
from pathlib import Path

from run_experiment import HERE, METHODS


def main() -> None:
    log = HERE / "outputs" / "launcher_status.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        for seed in range(3):
            for method in METHODS:
                complete = HERE / "outputs" / "runs" / method / f"fold{fold}" / f"seed{seed}" / "run_state" / "complete.json"
                if complete.exists():
                    continue
                command = [sys.executable, str(HERE / "run_experiment.py"), "--methods", method, "--folds", str(fold), "--seeds", str(seed)]
                event = {"fold": fold, "seed": seed, "method": method, "command": command}
                with log.open("a") as handle:
                    handle.write(json.dumps({**event, "event": "start"}) + "\n")
                result = subprocess.run(command, check=False)
                with log.open("a") as handle:
                    handle.write(json.dumps({**event, "event": "finish", "returncode": result.returncode}) + "\n")
                if result.returncode:
                    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
