#!/usr/bin/env python3
"""
Validate canonical OpenHands raw -> ADP conversion.

Checks:
  - same number of rows
  - ADP id == raw trajectory_id
  - details are strings
  - raw assistant tool-call count == ADP ApiAction count
  - raw assistant-only message count == ADP MessageAction count
  - raw system/user/tool observation count == ADP TextObservation count
  - tool function multiset preserved
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, TextIO


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


def iter_jsonl_path(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e


def raw_stats(row: dict[str, Any]) -> dict[str, Any]:
    api_functions = Counter()
    assistant_only_messages = 0
    text_observations = 0
    empty_assistant_no_tool = 0

    for msg in row["trajectory"]:
        role = msg.get("role")

        if role in {"system", "user", "tool"}:
            text_observations += 1
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content") or ""

            if tool_calls:
                for tc in tool_calls:
                    fn_obj = tc.get("function") or {}
                    fn_name = fn_obj.get("name") or tc.get("name") or "unknown_tool"
                    api_functions[fn_name] += 1
            elif isinstance(content, str) and content.strip():
                assistant_only_messages += 1
            else:
                empty_assistant_no_tool += 1

    return {
        "api_functions": api_functions,
        "api_count": sum(api_functions.values()),
        "message_action_count": assistant_only_messages,
        "text_observation_count": text_observations,
        "empty_assistant_no_tool": empty_assistant_no_tool,
    }


def adp_stats(adp_obj: Any) -> dict[str, Any]:
    api_functions = Counter()
    message_action_count = 0
    text_observation_count = 0
    other_classes = Counter()

    for item in adp_obj.content:
        cls = getattr(item, "class_", None)

        if cls == "api_action":
            api_functions[getattr(item, "function")] += 1
        elif cls == "message_action":
            message_action_count += 1
        elif cls == "text_observation":
            text_observation_count += 1
        else:
            other_classes[cls] += 1

    return {
        "api_functions": api_functions,
        "api_count": sum(api_functions.values()),
        "message_action_count": message_action_count,
        "text_observation_count": text_observation_count,
        "other_classes": other_classes,
    }


def model_validate_json(Trajectory, line: str):
    if hasattr(Trajectory, "model_validate_json"):
        return Trajectory.model_validate_json(line)

    return Trajectory.parse_raw(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adp-root", default=None)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--adp", required=True)
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()

    add_adp_to_path(args.adp_root)
    Trajectory = import_adp_schema()

    raw_rows = list(iter_jsonl_path(args.raw))

    with open(args.adp, "r", encoding="utf-8") as f:
        adp_lines = [line for line in f if line.strip()]

    errors: list[str] = []

    if len(raw_rows) != len(adp_lines):
        errors.append(f"row count mismatch: raw={len(raw_rows)} adp={len(adp_lines)}")

    n = min(len(raw_rows), len(adp_lines))

    aggregate_raw_api = Counter()
    aggregate_adp_api = Counter()
    aggregate_classes = Counter()

    for i in range(n):
        raw = raw_rows[i]
        adp_line = adp_lines[i]

        try:
            adp = model_validate_json(Trajectory, adp_line)
        except Exception as e:
            errors.append(f"row {i}: ADP validation failed: {type(e).__name__}: {e}")
            if len(errors) >= args.max_errors:
                break
            continue

        raw_id = str(raw.get("trajectory_id"))
        if adp.id != raw_id:
            errors.append(f"row {i}: id mismatch: raw={raw_id!r} adp={adp.id!r}")

        for k, v in adp.details.items():
            if not isinstance(k, str) or not isinstance(v, str):
                errors.append(
                    f"row {i}: details must be dict[str,str], bad item {k!r}: {type(v).__name__}"
                )
                break

        rs = raw_stats(raw)
        ads = adp_stats(adp)

        aggregate_raw_api.update(rs["api_functions"])
        aggregate_adp_api.update(ads["api_functions"])

        for item in adp.content:
            aggregate_classes[getattr(item, "class_", "MISSING_CLASS")] += 1

        if rs["api_count"] != ads["api_count"]:
            errors.append(
                f"row {i}: api count mismatch: raw={rs['api_count']} adp={ads['api_count']}"
            )

        if rs["api_functions"] != ads["api_functions"]:
            errors.append(
                f"row {i}: api function multiset mismatch: raw={rs['api_functions']} adp={ads['api_functions']}"
            )

        if rs["message_action_count"] != ads["message_action_count"]:
            errors.append(
                f"row {i}: MessageAction count mismatch: raw={rs['message_action_count']} adp={ads['message_action_count']}"
            )

        if rs["text_observation_count"] != ads["text_observation_count"]:
            errors.append(
                f"row {i}: TextObservation count mismatch: raw={rs['text_observation_count']} adp={ads['text_observation_count']}"
            )

        if len(errors) >= args.max_errors:
            break

    print("validated rows:", n)
    print("ADP class counts:", aggregate_classes)
    print("raw api functions:", aggregate_raw_api)
    print("adp api functions:", aggregate_adp_api)

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)

    print("\nOK: raw -> ADP conversion validates.")


if __name__ == "__main__":
    main()
