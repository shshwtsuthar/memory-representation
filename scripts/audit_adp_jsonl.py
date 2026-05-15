#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import zstandard as zstd


def add_adp_to_path(adp_root: str | None) -> None:
    if adp_root:
        root = Path(adp_root).resolve()
        if not (root / "schema" / "trajectory.py").exists():
            raise FileNotFoundError(
                f"--adp-root does not look like the ADP repo root: {root}"
            )
        sys.path.insert(0, str(root))


def import_adp_schema():
    from schema.trajectory import Trajectory
    return Trajectory


def iter_lines(path: str):
    if path.endswith(".zst"):
        with open(path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                buffer = ""
                while True:
                    chunk = reader.read(1024 * 1024)
                    if not chunk:
                        break

                    buffer += chunk.decode("utf-8", errors="replace")

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.strip():
                            yield line

                if buffer.strip():
                    yield buffer
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line


def model_validate_json(Trajectory, line: str):
    if hasattr(Trajectory, "model_validate_json"):
        return Trajectory.model_validate_json(line)
    return Trajectory.parse_raw(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adp-root", default=None)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    add_adp_to_path(args.adp_root)
    Trajectory = import_adp_schema()

    expanded_paths = []
    for pattern in args.paths:
        matches = sorted(glob.glob(pattern))
        expanded_paths.extend(matches if matches else [pattern])

    rows = 0
    classes = Counter()
    api_functions = Counter()
    resolved = Counter()
    repos = Counter()
    validation_errors = 0

    for path in expanded_paths:
        for line in iter_lines(path):
            rows += 1

            try:
                traj = model_validate_json(Trajectory, line)
            except Exception as e:
                validation_errors += 1
                print(f"[ERROR] validation failed in {path}: {type(e).__name__}: {e}", file=sys.stderr)
                continue

            resolved[traj.details.get("resolved", "")] += 1
            repos[traj.details.get("repo", "")] += 1

            for item in traj.content:
                cls = getattr(item, "class_", "MISSING_CLASS")
                classes[cls] += 1

                if cls == "api_action":
                    api_functions[getattr(item, "function", "unknown")] += 1

    print("rows:", rows)
    print("validation_errors:", validation_errors)
    print("classes:", classes)
    print("api_functions:", api_functions.most_common(20))
    print("resolved:", resolved)
    print("repos:", len(repos))


if __name__ == "__main__":
    main()
