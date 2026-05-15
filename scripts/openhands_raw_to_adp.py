#!/usr/bin/env python3
"""
Canonical OpenHands/Nebius raw trajectory -> ADP converter.

Input:
  JSONL where each line is one Nebius OpenHands row with:
    trajectory_id, instance_id, repo, trajectory, tools, model_patch,
    exit_status, resolved, gen_tests_correct, pred_passes_gen_tests, ...

Output:
  ADP JSONL where each line is one Trajectory.

Canonical mapping:
  system message                  -> TextObservation(source="environment", name="system")
  user message                    -> TextObservation(source="user", name="user")
  tool message                    -> TextObservation(source="environment", name=<tool name>)
  assistant message with toolcall -> ApiAction(function=<tool>, kwargs=<args>)
  assistant message only          -> MessageAction(content=<assistant content>)

This converter is intentionally loss-preserving. It keeps OpenHands tool calls as ApiAction.
It does NOT reinterpret execute_bash as CodeAction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, TextIO


def add_adp_to_path(adp_root: str | None) -> None:
    """
    Make ADP's `schema` package importable.

    Expected structure:
      <adp_root>/schema/trajectory.py
      <adp_root>/schema/action/api.py
      ...
    """
    if adp_root:
        root = Path(adp_root).resolve()
        if not (root / "schema" / "trajectory.py").exists():
            raise FileNotFoundError(
                f"--adp-root does not look like the ADP repo root: {root}"
            )
        sys.path.insert(0, str(root))


def import_adp_schema():
    """
    Import after sys.path is configured.
    """
    from schema.trajectory import Trajectory
    from schema.action.api import ApiAction
    from schema.action.message import MessageAction
    from schema.observation.text import TextObservation

    return Trajectory, ApiAction, MessageAction, TextObservation


def maybe_json_loads(value: Any) -> Any:
    """
    Nebius stores assistant tool-call function.arguments as strings in the source dataset.
    Your current JSONL may already have dicts because we deserialized earlier.
    This accepts both.
    """
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def ensure_dict(value: Any) -> dict[str, Any]:
    """
    ApiAction.kwargs must be dict[str, Any].
    If raw arguments are missing, use {}.
    If raw arguments are non-dict after deserialization, preserve them under "value".
    """
    value = maybe_json_loads(value)

    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    return {"value": value}


def stringify_detail(value: Any) -> str:
    """
    Your uploaded Trajectory schema says details: dict[str, str].
    Therefore every metadata value must become a string.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_details(row: dict[str, Any]) -> dict[str, str]:
    """
    Preserve all top-level metadata except the trajectory itself.

    This includes large fields such as tools and model_patch. That is intentional for the
    canonical artifact. Later memory renderers can choose not to inject details["tools"].
    """
    details: dict[str, str] = {
        "dataset": "nebius/SWE-rebench-openhands-trajectories",
    }

    for key, value in row.items():
        if key == "trajectory":
            continue
        details[key] = stringify_detail(value)

    # Convenience duplicates with stable names.
    # If the source row lacks one of these, this stores "".
    details.setdefault("trajectory_id", stringify_detail(row.get("trajectory_id")))
    details.setdefault("instance_id", stringify_detail(row.get("instance_id")))
    details.setdefault("repo", stringify_detail(row.get("repo")))
    details.setdefault("resolved", stringify_detail(row.get("resolved")))
    details.setdefault("exit_status", stringify_detail(row.get("exit_status")))
    details.setdefault("model_patch", stringify_detail(row.get("model_patch")))

    return details


def assistant_tool_call_to_api_action(
    tool_call: dict[str, Any],
    assistant_content: str | None,
    ApiAction,
):
    """
    Convert one assistant tool call into one ApiAction.

    Raw shape observed in your sample:
      {
        "function": {
          "name": "execute_bash",
          "arguments": {...}
        },
        "id": "...",
        "type": "function"
      }
    """
    function_obj = tool_call.get("function") or {}

    function_name = (
        function_obj.get("name")
        or tool_call.get("name")
        or "unknown_tool"
    )

    kwargs = ensure_dict(function_obj.get("arguments", {}))

    description = assistant_content.strip() if isinstance(assistant_content, str) and assistant_content.strip() else None

    # Preserve tool_call metadata inside kwargs without overwriting real args.
    if "id" in tool_call and "_openhands_tool_call_id" not in kwargs:
        kwargs["_openhands_tool_call_id"] = tool_call["id"]

    if "type" in tool_call and "_openhands_tool_call_type" not in kwargs:
        kwargs["_openhands_tool_call_type"] = tool_call["type"]

    # For think actions, the actual thought is valuable as ADP reasoning_content.
    # We still preserve it in kwargs as well.
    reasoning_content = None
    if function_name == "think":
        thought = kwargs.get("thought")
        if isinstance(thought, str) and thought.strip():
            reasoning_content = thought

    return ApiAction(
        function=function_name,
        kwargs=kwargs,
        description=description,
        reasoning_content=reasoning_content,
    )


def convert_message(
    msg: dict[str, Any],
    ApiAction,
    MessageAction,
    TextObservation,
) -> list[Any]:
    role = msg.get("role")
    content = msg.get("content")

    if content is None:
        content = ""
    elif not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, sort_keys=True)

    if role == "system":
        return [
            TextObservation(
                content=content,
                name="system",
                source="environment",
            )
        ]

    if role == "user":
        return [
            TextObservation(
                content=content,
                name="user",
                source="user",
            )
        ]

    if role == "tool":
        name = msg.get("name")
        if name is None:
            name = msg.get("tool_call_id") or "tool"

        return [
            TextObservation(
                content=content,
                name=str(name),
                source="environment",
            )
        ]

    if role == "assistant":
        tool_calls = msg.get("tool_calls") or []

        # Assistant action messages in your sample usually have exactly one tool call.
        # This supports zero, one, or many.
        if tool_calls:
            return [
                assistant_tool_call_to_api_action(
                    tool_call=tool_call,
                    assistant_content=content,
                    ApiAction=ApiAction,
                )
                for tool_call in tool_calls
            ]

        if content.strip():
            return [
                MessageAction(
                    content=content,
                    description=None,
                )
            ]

        # Empty assistant message with no tool calls: no semantic event to preserve.
        return []

    # Unknown role: preserve it rather than crashing.
    return [
        TextObservation(
            content=json.dumps(msg, ensure_ascii=False, sort_keys=True),
            name=f"unknown_role:{role}",
            source="environment",
        )
    ]


def convert_row(row: dict[str, Any], Trajectory, ApiAction, MessageAction, TextObservation):
    trajectory_id = row.get("trajectory_id")
    if trajectory_id is None:
        trajectory_id = row.get("id")

    if trajectory_id is None:
        raise ValueError("Raw row has neither trajectory_id nor id")

    raw_trajectory = row.get("trajectory")
    if not isinstance(raw_trajectory, list):
        raise ValueError(f"Row {trajectory_id} has non-list trajectory")

    content: list[Any] = []

    for msg in raw_trajectory:
        if not isinstance(msg, dict):
            msg = {
                "role": "unknown",
                "content": msg,
            }

        content.extend(
            convert_message(
                msg=msg,
                ApiAction=ApiAction,
                MessageAction=MessageAction,
                TextObservation=TextObservation,
            )
        )

    return Trajectory(
        id=str(trajectory_id),
        content=content,
        details=build_details(row),
    )


def dumps_pydantic(model: Any) -> str:
    """
    Supports Pydantic v2. Falls back to v1-style .json().
    Your uploaded schema uses Pydantic v2 field_validator, so model_dump_json is expected.
    """
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()

    return model.json()


def iter_jsonl(f: TextIO) -> Iterable[tuple[int, dict[str, Any]]]:
    for line_no, line in enumerate(f, start=1):
        if not line.strip():
            continue

        try:
            yield line_no, json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adp-root",
        default=None,
        help="Path to the ADP repo root containing schema/. "
             "If omitted, assumes schema is already importable via PYTHONPATH.",
    )
    parser.add_argument(
        "--input",
        default="-",
        help="Input raw JSONL path. Use '-' for stdin.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output ADP JSONL path. Use '-' for stdout.",
    )
    parser.add_argument(
        "--errors",
        default="raise",
        choices=["raise", "skip"],
        help="Whether to raise or skip rows that fail conversion.",
    )
    args = parser.parse_args()

    add_adp_to_path(args.adp_root)
    Trajectory, ApiAction, MessageAction, TextObservation = import_adp_schema()

    input_fh = sys.stdin if args.input == "-" else open(args.input, "r", encoding="utf-8")
    output_fh = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")

    converted = 0
    skipped = 0

    try:
        for line_no, row in iter_jsonl(input_fh):
            try:
                adp_trajectory = convert_row(
                    row=row,
                    Trajectory=Trajectory,
                    ApiAction=ApiAction,
                    MessageAction=MessageAction,
                    TextObservation=TextObservation,
                )
            except Exception as e:
                if args.errors == "skip":
                    skipped += 1
                    print(
                        f"[WARN] skipping line {line_no}: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    continue
                raise

            output_fh.write(dumps_pydantic(adp_trajectory))
            output_fh.write("\n")
            converted += 1

    finally:
        if input_fh is not sys.stdin:
            input_fh.close()
        if output_fh is not sys.stdout:
            output_fh.close()

    print(
        f"converted={converted} skipped={skipped}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
