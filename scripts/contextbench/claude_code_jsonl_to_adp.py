#!/usr/bin/env python3
"""
Convert Claude Code JSONL session logs to ADP standardized trajectories.

One input Claude Code JSONL session -> one ADP Trajectory JSON object.
Multiple inputs can be written as a combined JSONL file or as one .adp.json
file per input path.

This script intentionally uses only the Python standard library. The emitted
objects follow the Agent Data Protocol schema_version 1.1.0 shape:
  - Trajectory: schema_version, id, content, details
  - Actions: api_action, code_action, message_action
  - Observations: text_observation

Mapping policy, briefly:
  - User messages -> TextObservation(source="user")
  - Claude Code tool results -> TextObservation(source="environment")
  - Parallel Claude tool calls are sequentialized into action/result pairs by default
  - Bash tool calls -> CodeAction(language="bash")
  - Other Claude Code tools -> ApiAction(function=<Claude tool name>, kwargs=<input>)
  - Assistant final/explanatory text not consumed by a later tool call -> MessageAction
  - Claude `thinking` blocks -> Action.reasoning_content unless --drop-thinking is set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "1.1.0"

# Claude Code tools that are better represented as executable code in ADP.
CODE_TOOL_LANGUAGES = {
    "Bash": "bash",
}

# Keys in Claude Code tool input that are brief, user-facing action descriptions,
# not actual tool arguments in the target environment.
DESCRIPTION_KEYS = {
    "description",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object per line")
            obj.setdefault("_jsonl_lineno", lineno)
            records.append(obj)
    return records


def stable_short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:n]


def choose_leaf_uuid(records: list[dict[str, Any]]) -> str | None:
    """Choose the final message UUID for parent-chain reconstruction.

    Claude Code exports sometimes include a `summary.leafUuid`, but in the
    attached examples that UUID can refer to a compacted/omitted message. We
    use it only when it exists in the current file; otherwise we use the last
    message UUID present in the JSONL.
    """
    uuid_to_record = {r.get("uuid"): r for r in records if r.get("uuid")}
    for r in records:
        if r.get("type") == "summary" and r.get("leafUuid") in uuid_to_record:
            return str(r["leafUuid"])
    for r in reversed(records):
        if r.get("uuid"):
            return str(r["uuid"])
    return None


def reconstruct_message_chain(records: list[dict[str, Any]], use_parent_chain: bool = True) -> list[dict[str, Any]]:
    """Return the linear Claude Code conversation path.

    If parent UUIDs are available, walking the final leaf backwards avoids
    accidentally including abandoned branches. If reconstruction fails, we fall
    back to line order for message records.
    """
    message_records = [r for r in records if r.get("uuid") and isinstance(r.get("message"), dict)]
    if not use_parent_chain:
        return message_records

    uuid_to_record = {str(r["uuid"]): r for r in message_records}
    leaf = choose_leaf_uuid(records)
    if not leaf or leaf not in uuid_to_record:
        return message_records

    chain_reversed: list[dict[str, Any]] = []
    seen: set[str] = set()
    cur: str | None = leaf
    while cur:
        if cur in seen or cur not in uuid_to_record:
            break
        seen.add(cur)
        rec = uuid_to_record[cur]
        chain_reversed.append(rec)
        parent = rec.get("parentUuid")
        cur = str(parent) if parent else None

    chain = list(reversed(chain_reversed))
    # If the chain is implausibly short because a parent is missing, line order
    # is safer than silently dropping most of the session.
    if len(chain) < max(1, len(message_records) // 2):
        return message_records
    return chain


def as_blocks(message_content: Any) -> list[dict[str, Any]]:
    """Normalize Claude message.content into a list of content blocks."""
    if isinstance(message_content, str):
        return [{"type": "text", "text": message_content}]
    if isinstance(message_content, list):
        blocks: list[dict[str, Any]] = []
        for item in message_content:
            if isinstance(item, dict):
                blocks.append(item)
            else:
                blocks.append({"type": "unknown", "value": item})
        return blocks
    if message_content is None:
        return []
    return [{"type": "unknown", "value": message_content}]


def stringify_content(value: Any) -> str:
    """Convert Claude content/result payloads to deterministic text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text", "")))
            elif isinstance(item, dict) and "content" in item:
                pieces.append(stringify_content(item.get("content")))
            else:
                pieces.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
        return "\n".join(pieces)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def compact_text(parts: Iterable[str | None]) -> str | None:
    clean = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    if not clean:
        return None
    return "\n\n".join(clean)


def raw_provenance(
    rec: dict[str, Any],
    *,
    role: str,
    block_type: str,
    block_index: int | None = None,
    tool_use_id: str | None = None,
) -> dict[str, Any]:
    """Return deterministic provenance linking an ADP item to raw Claude JSONL."""
    out: dict[str, Any] = {}
    lineno = rec.get("_jsonl_lineno")
    if isinstance(lineno, int):
        out["raw_jsonl_line"] = lineno
    elif lineno is not None:
        try:
            out["raw_jsonl_line"] = int(lineno)
        except (TypeError, ValueError):
            pass
    if rec.get("uuid") is not None:
        out["raw_uuid"] = str(rec.get("uuid"))
    if rec.get("parentUuid") is not None:
        out["raw_parent_uuid"] = str(rec.get("parentUuid"))
    out["raw_role"] = str(role)
    out["raw_block_type"] = str(block_type)
    if block_index is not None:
        out["raw_block_index"] = block_index
    if tool_use_id:
        out["tool_use_id"] = str(tool_use_id)
    return out


def extract_issue_metadata(first_user_text: str) -> dict[str, Any]:
    """Best-effort extraction of SWE task metadata from the initial prompt."""
    meta: dict[str, Any] = {}
    for key in ("instance_id", "repo", "base_commit"):
        m = re.search(rf"(?m)^\s*{re.escape(key)}:\s*(.+?)\s*$", first_user_text)
        if m:
            meta[key] = m.group(1).strip()

    ps = re.search(r"(?ms)^\s*problem_statement:\s*(.*)$", first_user_text)
    if ps:
        # Keep the whole tail: in these SWE prompts, problem_statement is the
        # final field and often contains newlines/code blocks.
        meta["problem_statement"] = ps.group(1).strip()
    return meta


def make_text_observation(
    content: str,
    source: str,
    name: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    obs: dict[str, Any] = {
        "class_": "text_observation",
        "content": content,
        "name": name,
        "source": source,
    }
    if provenance:
        obs["provenance"] = provenance
    return obs


def split_tool_description(tool_input: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Return kwargs and a brief description from a Claude tool input object."""
    kwargs = deepcopy(tool_input)
    descriptions: list[str] = []
    for key in DESCRIPTION_KEYS:
        value = kwargs.pop(key, None)
        if isinstance(value, str) and value.strip():
            descriptions.append(value.strip())
    return kwargs, compact_text(descriptions)


def make_action(
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    visible_text: str | None,
    reasoning: str | None,
    lowercase_api_functions: bool = False,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map one Claude Code tool_use block to an ADP action."""
    kwargs, tool_desc = split_tool_description(tool_input)
    description = compact_text([visible_text, tool_desc])

    if tool_name in CODE_TOOL_LANGUAGES:
        # Bash is executable code, so ADP CodeAction is more faithful than a
        # generic APIAction. Keep non-command inputs in details only through the
        # trajectory-level raw_tool_inputs map.
        action: dict[str, Any] = {
            "class_": "code_action",
            "language": CODE_TOOL_LANGUAGES[tool_name],
            "content": stringify_content(tool_input.get("command", "")),
            "description": description,
        }
    else:
        function = tool_name.lower() if lowercase_api_functions else tool_name
        action = {
            "class_": "api_action",
            "function": function,
            "kwargs": kwargs,
            "description": description,
        }

    if reasoning:
        action["reasoning_content"] = reasoning
    if provenance:
        action["provenance"] = provenance
    return action


def make_message_action(
    content: str,
    reasoning: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action: dict[str, Any] = {
        "class_": "message_action",
        "content": content,
        "description": None,
    }
    if reasoning:
        action["reasoning_content"] = reasoning
    if provenance:
        action["provenance"] = provenance
    return action



def flush_ready_tool_pairs(
    content: list[dict[str, Any]],
    pending_tool_actions: list[tuple[str, dict[str, Any]]],
    pending_tool_results: dict[str, dict[str, Any]],
) -> None:
    """Emit buffered tool action/result pairs in original tool-call order."""
    while pending_tool_actions:
        tool_use_id, action = pending_tool_actions[0]
        if tool_use_id not in pending_tool_results:
            break
        observation = pending_tool_results.pop(tool_use_id)
        pending_tool_actions.pop(0)
        content.append(action)
        content.append(observation)

def flush_unpaired_tool_actions(
    content: list[dict[str, Any]],
    pending_tool_actions: list[tuple[str, dict[str, Any]]],
    warnings: list[str],
    *,
    reason: str,
) -> None:
    """Emit tool actions that never received a matching tool_result."""
    while pending_tool_actions:
        tool_use_id, action = pending_tool_actions.pop(0)
        warnings.append(f"Unpaired tool_use {tool_use_id!r} emitted without tool_result before {reason}")
        content.append(action)


def flush_pending_message(
    content: list[dict[str, Any]],
    pending_visible: list[str],
    pending_reasoning: list[str],
    pending_visible_provenances: list[dict[str, Any]] | None = None,
    *,
    drop_empty: bool = True,
) -> None:
    visible = compact_text(pending_visible)
    reasoning = compact_text(pending_reasoning)
    provenance = pending_visible_provenances[-1] if pending_visible_provenances else None
    if visible or (reasoning and not drop_empty):
        content.append(make_message_action(visible or "", reasoning=reasoning, provenance=provenance))
    pending_visible.clear()
    pending_reasoning.clear()
    if pending_visible_provenances is not None:
        pending_visible_provenances.clear()


def convert_records_to_adp(
    records: list[dict[str, Any]],
    *,
    source_file: str,
    use_parent_chain: bool = True,
    drop_thinking: bool = False,
    lowercase_api_functions: bool = False,
    include_file_history: bool = False,
    include_tool_use_result_details: bool = True,
    preserve_raw_tool_inputs: bool = True,
    preserve_parallel_tool_order: bool = False,
) -> dict[str, Any]:
    chain = reconstruct_message_chain(records, use_parent_chain=use_parent_chain)

    content: list[dict[str, Any]] = []
    pending_reasoning: list[str] = []
    pending_visible: list[str] = []
    pending_visible_provenances: list[dict[str, Any]] = []
    tool_id_to_name: dict[str, str] = {}
    tool_id_to_input: dict[str, Any] = {}
    tool_results: dict[str, Any] = {}
    # Internal buffer used to turn Claude's possible parallel tool-call layout
    #   action1, action2, result1, result2
    # into ADP's preferred sequential layout
    #   action1, result1, action2, result2
    # while still preserving every call and result exactly.
    pending_tool_actions: list[tuple[str, dict[str, Any]]] = []
    pending_tool_results: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    first_user_text: str | None = None

    # Session-level metadata observed in Claude Code exports.
    session_ids = sorted({str(r.get("sessionId")) for r in chain if r.get("sessionId")})
    versions = sorted({str(r.get("version")) for r in chain if r.get("version")})
    models = sorted({str(r.get("message", {}).get("model")) for r in chain if r.get("message", {}).get("model")})
    cwds = sorted({str(r.get("cwd")) for r in chain if r.get("cwd")})
    git_branches = sorted({str(r.get("gitBranch")) for r in chain if r.get("gitBranch")})

    for rec in chain:
        msg = rec.get("message") or {}
        role = msg.get("role") or rec.get("type")
        blocks = as_blocks(msg.get("content"))

        if role == "user":
            # A normal user message should appear after any pending assistant text.
            # Tool-result pseudo-user messages are environment observations and
            # should not force a pending assistant status message to be emitted.
            has_tool_result = any(b.get("type") == "tool_result" for b in blocks)
            if not has_tool_result:
                flush_ready_tool_pairs(content, pending_tool_actions, pending_tool_results)
                flush_unpaired_tool_actions(content, pending_tool_actions, warnings, reason="normal user message")
                for leftover_id, leftover_obs in list(pending_tool_results.items()):
                    warnings.append(f"tool_result {leftover_id!r} had no pending tool_use before normal user message")
                    content.append(leftover_obs)
                    pending_tool_results.pop(leftover_id, None)
                flush_pending_message(content, pending_visible, pending_reasoning, pending_visible_provenances)

            for bi, block in enumerate(blocks):
                btype = block.get("type")
                if btype == "tool_result":
                    tool_use_id = str(block.get("tool_use_id", ""))
                    tool_name = tool_id_to_name.get(tool_use_id, "tool")
                    result_text = stringify_content(block.get("content"))
                    is_error = block.get("is_error")
                    if is_error is True:
                        name = f"{tool_name}:{tool_use_id}:error" if tool_use_id else f"{tool_name}:error"
                    else:
                        name = f"{tool_name}:{tool_use_id}" if tool_use_id else tool_name
                    observation = make_text_observation(
                        result_text,
                        source="environment",
                        name=name,
                        provenance=raw_provenance(
                            rec,
                            role=str(role),
                            block_type="tool_result",
                            block_index=bi,
                            tool_use_id=tool_use_id,
                        ),
                    )
                    if preserve_parallel_tool_order:
                        content.append(observation)
                    else:
                        pending_ids = {pending_id for pending_id, _ in pending_tool_actions}
                        if tool_use_id in pending_ids:
                            pending_tool_results[tool_use_id] = observation
                            flush_ready_tool_pairs(content, pending_tool_actions, pending_tool_results)
                        elif not tool_use_id and pending_tool_actions:
                            # Malformed result with no id: attach FIFO.
                            fifo_id, _ = pending_tool_actions[0]
                            pending_tool_results[fifo_id] = observation
                            flush_ready_tool_pairs(content, pending_tool_actions, pending_tool_results)
                        else:
                            warnings.append(
                                f"tool_result {tool_use_id!r} at line {rec.get('_jsonl_lineno')} had no pending tool_use"
                            )
                            content.append(observation)
                    if include_tool_use_result_details and tool_use_id:
                        tool_results[tool_use_id] = rec.get("toolUseResult", block)
                elif btype == "text":
                    text = stringify_content(block.get("text"))
                    if first_user_text is None and text.strip():
                        first_user_text = text
                    content.append(
                        make_text_observation(
                            text,
                            source="user",
                            provenance=raw_provenance(rec, role=str(role), block_type="text", block_index=bi),
                        )
                    )
                else:
                    text = stringify_content(block)
                    warnings.append(f"Unknown user content block type {btype!r} at line {rec.get('_jsonl_lineno')}")
                    if first_user_text is None and text.strip():
                        first_user_text = text
                    content.append(
                        make_text_observation(
                            text,
                            source="user",
                            provenance=raw_provenance(rec, role=str(role), block_type="text", block_index=bi),
                        )
                    )

        elif role == "assistant":
            for bi, block in enumerate(blocks):
                btype = block.get("type")
                if btype == "thinking":
                    if not drop_thinking:
                        thought = stringify_content(block.get("thinking"))
                        if thought.strip():
                            pending_reasoning.append(thought)
                elif btype == "text":
                    text = stringify_content(block.get("text"))
                    if text.strip():
                        pending_visible.append(text)
                        pending_visible_provenances.append(raw_provenance(rec, role=str(role), block_type="text", block_index=bi))
                elif btype == "tool_use":
                    tool_name = str(block.get("name", ""))
                    tool_use_id = str(block.get("id", ""))
                    tool_input = block.get("input") or {}
                    if not isinstance(tool_input, dict):
                        tool_input = {"input": tool_input}
                    if tool_use_id:
                        tool_id_to_name[tool_use_id] = tool_name
                        if preserve_raw_tool_inputs:
                            tool_id_to_input[tool_use_id] = deepcopy(tool_input)
                    action = make_action(
                        tool_name=tool_name,
                        tool_input=tool_input,
                        visible_text=compact_text(pending_visible),
                        reasoning=compact_text(pending_reasoning),
                        lowercase_api_functions=lowercase_api_functions,
                        provenance=raw_provenance(rec, role=str(role), block_type="tool_use", block_index=bi, tool_use_id=tool_use_id),
                    )
                    if preserve_parallel_tool_order:
                        content.append(action)
                    else:
                        pending_tool_actions.append((tool_use_id, action))
                    pending_visible.clear()
                    pending_visible_provenances.clear()
                    pending_reasoning.clear()
                else:
                    warnings.append(f"Unknown assistant content block type {btype!r} at line {rec.get('_jsonl_lineno')}")
                    text = stringify_content(block)
                    if text.strip():
                        pending_visible.append(text)
                        pending_visible_provenances.append(raw_provenance(rec, role=str(role), block_type="text", block_index=bi))
        else:
            warnings.append(f"Unknown message role {role!r} at line {rec.get('_jsonl_lineno')}")

    flush_ready_tool_pairs(content, pending_tool_actions, pending_tool_results)
    flush_unpaired_tool_actions(content, pending_tool_actions, warnings, reason="end of trajectory")
    for leftover_id, leftover_obs in list(pending_tool_results.items()):
        warnings.append(f"tool_result {leftover_id!r} had no pending tool_use at end of trajectory")
        content.append(leftover_obs)
        pending_tool_results.pop(leftover_id, None)
    flush_pending_message(content, pending_visible, pending_reasoning, pending_visible_provenances)

    # Non-message records: keep useful context in details, but do not pollute
    # the action/observation sequence.
    summaries = [r for r in records if r.get("type") == "summary"]
    snapshots = [r for r in records if r.get("type") == "file-history-snapshot"]

    # Prefer the actual Claude sessionId as the ADP id. If missing, fall back to
    # a stable id derived from source filename and content.
    if len(session_ids) == 1:
        trajectory_id = session_ids[0]
    elif session_ids:
        trajectory_id = f"{Path(source_file).stem}-{stable_short_hash('|'.join(session_ids))}"
    else:
        trajectory_id = f"{Path(source_file).stem}-{stable_short_hash(json.dumps(records, sort_keys=True, ensure_ascii=False))}"

    details: dict[str, Any] = {
        "source_format": "claude_code_jsonl",
        "source_file": source_file,
        "session_ids": session_ids,
        "claude_code_versions": versions,
        "models": models,
        "cwd": cwds,
        "git_branches": git_branches,
        "raw_record_count": len(records),
        "message_record_count": len([r for r in records if r.get("uuid") and isinstance(r.get("message"), dict)]),
        "converted_chain_record_count": len(chain),
        "tool_use_id_to_function": tool_id_to_name,
        "summary": [s.get("summary") for s in summaries if s.get("summary")],
        "issue_metadata": extract_issue_metadata(first_user_text or ""),
        "warnings": warnings,
    }

    if preserve_raw_tool_inputs:
        details["raw_tool_inputs_by_id"] = tool_id_to_input
    if include_tool_use_result_details:
        details["raw_tool_results_by_id"] = tool_results
    if include_file_history:
        details["file_history_snapshots"] = snapshots
    else:
        details["file_history_snapshot_count"] = len(snapshots)

    return {
        "schema_version": SCHEMA_VERSION,
        "id": trajectory_id,
        "content": content,
        "details": details,
    }


def iter_input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.jsonl")))
        else:
            files.append(p)
    return files


def write_outputs(trajectories: list[tuple[Path, dict[str, Any]]], args: argparse.Namespace) -> None:
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for src, traj in trajectories:
            out_path = out_dir / f"{src.stem}.adp.json"
            out_path.write_text(json.dumps(traj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {out_path}", file=sys.stderr)
        return

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", encoding="utf-8") as f:
            for _, traj in trajectories:
                f.write(json.dumps(traj, ensure_ascii=False, sort_keys=args.sort_keys) + "\n")
        print(f"wrote {out_path}", file=sys.stderr)
        return

    # stdout JSONL by default.
    for _, traj in trajectories:
        print(json.dumps(traj, ensure_ascii=False, sort_keys=args.sort_keys))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Claude Code JSONL trajectories to ADP standardized trajectories."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Claude Code .jsonl files or directories containing .jsonl files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write all ADP trajectories to this JSONL file. Default: stdout.",
    )
    parser.add_argument(
        "--output-dir",
        help="Write one pretty-printed .adp.json file per input JSONL.",
    )
    parser.add_argument(
        "--line-order",
        action="store_true",
        help="Use JSONL line order instead of reconstructing the final parentUuid chain.",
    )
    parser.add_argument(
        "--preserve-parallel-tool-order",
        action="store_true",
        help=(
            "Preserve Claude's raw action1,action2,result1,result2 ordering for parallel tool calls. "
            "Default is to sequentialize to action1,result1,action2,result2, which better matches ADP's preferred alternation."
        ),
    )
    parser.add_argument(
        "--drop-thinking",
        action="store_true",
        help="Do not emit Claude thinking blocks as ADP reasoning_content.",
    )
    parser.add_argument(
        "--lowercase-api-functions",
        action="store_true",
        help="Lowercase non-Bash Claude Code APIAction.function names.",
    )
    parser.add_argument(
        "--include-file-history",
        action="store_true",
        help="Preserve file-history-snapshot records inside Trajectory.details.",
    )
    parser.add_argument(
        "--no-raw-tool-details",
        action="store_true",
        help="Do not preserve raw tool inputs/results in Trajectory.details.",
    )
    parser.add_argument(
        "--sort-keys",
        action="store_true",
        help="Sort JSON object keys in combined JSONL output.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print a short conversion summary to stderr.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = iter_input_files(args.inputs)
    if args.output:
        output_resolved = Path(args.output).resolve()
        files = [f for f in files if f.resolve() != output_resolved]
    if not files:
        print("No input .jsonl files found.", file=sys.stderr)
        return 2

    trajectories: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        if not path.exists():
            print(f"Input does not exist: {path}", file=sys.stderr)
            return 2
        records = read_jsonl(path)
        traj = convert_records_to_adp(
            records,
            source_file=path.name,
            use_parent_chain=not args.line_order,
            drop_thinking=args.drop_thinking,
            lowercase_api_functions=args.lowercase_api_functions,
            include_file_history=args.include_file_history,
            include_tool_use_result_details=not args.no_raw_tool_details,
            preserve_raw_tool_inputs=not args.no_raw_tool_details,
            preserve_parallel_tool_order=args.preserve_parallel_tool_order,
        )
        trajectories.append((path, traj))
        if args.stats:
            actions = sum(1 for x in traj["content"] if str(x.get("class_", "")).endswith("_action"))
            observations = sum(1 for x in traj["content"] if str(x.get("class_", "")).endswith("_observation"))
            warnings = len(traj["details"].get("warnings", []))
            print(
                f"{path.name}: id={traj['id']} content={len(traj['content'])} "
                f"actions={actions} observations={observations} warnings={warnings}",
                file=sys.stderr,
            )

    write_outputs(trajectories, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
