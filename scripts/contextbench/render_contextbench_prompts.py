#!/usr/bin/env python3
"""
Render frozen OpenHands prompts for a ContextBench prior-trajectory experiment.

Input:
  data/contextbench_phase2/run_manifest.jsonl

Expected manifest fields, based on the current project manifest:
  - run_id
  - condition: one of no_memory/raw/adp/memory
  - injection_file: null for no_memory, text file path for raw/adp/memory
  - output_dir: where the run will later store outputs
  - target_instance_id
  - target_repo
  - target_base_commit
  - target_problem_statement
  - sandbox_image
  - prior_instance_id / prior_repo / prior_trajectory_id, used only for audit metadata

Output:
  data/contextbench_phase2/prompts/<target_instance_id>/<condition>/prompt.txt
  data/contextbench_phase2/prompt_manifest.jsonl
  data/contextbench_phase2/prompt_render_report.json
  data/contextbench_phase2/forbidden_prompt_scan.txt

Optional:
  Also writes <output_dir>/prompt.txt for each run, so the OpenHands runner can
  consume the prompt directly from the run directory.

Design constraints:
  - deterministic
  - no LLM calls
  - no random IDs
  - no timestamps
  - no network access
  - stable JSON output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROMPT_TEMPLATE_VERSION = "contextbench_prompt_v1.0.0"
EXPECTED_CONDITIONS = ("no_memory", "raw", "adp", "memory")
EXPECTED_CONDITION_SET = set(EXPECTED_CONDITIONS)

# These should never appear in rendered prompts. Some of these can appear in
# audit manifests, but not in the text passed to the agent.
FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("field_FAIL_TO_PASS", re.compile(r"\bFAIL_TO_PASS\b")),
    ("field_PASS_TO_PASS", re.compile(r"\bPASS_TO_PASS\b")),
    ("field_target_fail_to_pass", re.compile(r"\btarget_fail_to_pass\b")),
    ("field_target_pass_to_pass", re.compile(r"\btarget_pass_to_pass\b")),
    ("field_test_patch", re.compile(r"\btest_patch\b")),
    ("field_model_patch", re.compile(r"\bmodel_patch\b")),
    ("diff_git", re.compile(r"diff --git")),
    ("gold_patch", re.compile(r"\bgold(?:en)?[_ -]?patch\b", re.I)),
    ("system_reminder_open", re.compile(r"<system-reminder>", re.I)),
    ("system_reminder_close", re.compile(r"</system-reminder>", re.I)),
    ("system_reminder_phrase", re.compile(r"Whenever you read a file", re.I)),
    ("reasoning_content", re.compile(r"\breasoning_content\b")),
    ("TodoWrite", re.compile(r"\bTodoWrite\b")),
    ("claude_dir", re.compile(r"(?:^|[/\\])\.claude(?:[/\\]|$)")),
    ("token_usage", re.compile(r"\.token_usage\b|\btoken_usage\b")),
    ("manual_yaml", re.compile(r"\bmanual\.ya?ml\b", re.I)),
    ("preds_json", re.compile(r"\b\w*_preds\.json\b|\ball_preds\.json\b|\bpreds\.json\b")),
]


# ---------------------------------------------------------------------------
# Stable text / JSON helpers
# ---------------------------------------------------------------------------


def normalize_text(value: Any) -> str:
    """Normalize text deterministically: LF newlines, trailing whitespace stripped."""
    if value is None:
        s = ""
    elif isinstance(value, str):
        s = value
    else:
        s = str(value)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in s.split("\n")).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def stable_json_dumps(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_text_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text_utf8(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
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
                raise ValueError(f"{path}:{lineno}: expected JSON object per line")
            yield lineno, obj


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def pick(row: dict[str, Any], *names: str, required: bool = True, default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and normalize_text(value) != "":
            return normalize_text(value)
    if required:
        raise KeyError(f"Missing required field. Tried {names}; available keys={sorted(row.keys())}")
    return default


def safe_path_component(value: str) -> str:
    value = normalize_text(value)
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("._")
    return value or "unnamed"


def relpath_or_str(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def resolve_existing_path(path_value: str, *, root_dir: Path, manifest_dir: Path) -> Path:
    """Resolve a manifest path without relying on the caller's CWD layout."""
    p = Path(path_value)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            root_dir / p,
            manifest_dir / p,
            Path.cwd() / p,
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not resolve path from manifest: "
        f"{path_value!r}. Tried: " + ", ".join(str(c) for c in candidates)
    )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


def prior_context_label(condition: str) -> str:
    if condition == "no_memory":
        return "No prior context is provided for this run."
    if condition == "raw":
        return "The following is a stripped raw transcript from an officially related prior issue."
    if condition == "adp":
        return "The following is a stripped ADP-rendered trajectory from an officially related prior issue."
    if condition == "memory":
        return "The following is deterministic extractive memory from an officially related prior issue."
    raise ValueError(f"Unknown condition: {condition}")


def render_prompt(row: dict[str, Any], *, root_dir: Path, manifest_dir: Path) -> tuple[str, dict[str, Any]]:
    run_id = pick(row, "run_id")
    condition = pick(row, "condition")
    if condition not in EXPECTED_CONDITION_SET:
        raise ValueError(f"{run_id}: invalid condition {condition!r}; expected {sorted(EXPECTED_CONDITION_SET)}")

    target_instance_id = pick(row, "target_instance_id", "related_instance_id", "instance_id")
    target_repo = pick(row, "target_repo", "repo", required=False)
    target_base_commit = pick(row, "target_base_commit", "base_commit", required=False)
    target_problem_statement = pick(row, "target_problem_statement", "related_problem_statement", "problem_statement")
    sandbox_image = pick(row, "sandbox_image", required=False)
    output_dir_value = pick(row, "output_dir", required=False)

    prior_instance_id = pick(row, "prior_instance_id", "experience_instance_id", required=False)
    prior_repo = pick(row, "prior_repo", "experience_repo", required=False)
    prior_trajectory_id = pick(row, "prior_trajectory_id", required=False)

    injection_file_value = pick(row, "injection_file", "injection_path", "injection_txt", required=False)
    injection_text = ""
    injection_sha256 = ""
    injection_path_resolved = ""

    if condition == "no_memory":
        if injection_file_value:
            raise ValueError(f"{run_id}: no_memory must not have an injection_file")
    else:
        if not injection_file_value:
            raise ValueError(f"{run_id}: condition={condition} requires injection_file")
        injection_path = resolve_existing_path(injection_file_value, root_dir=root_dir, manifest_dir=manifest_dir)
        injection_text = normalize_text(read_text_utf8(injection_path))
        if not injection_text:
            raise ValueError(f"{run_id}: injection file is empty: {injection_path}")
        injection_sha256 = sha256_text(injection_text)
        injection_path_resolved = relpath_or_str(injection_path, root_dir)

    # Keep this policy identical across all conditions. The only intended
    # condition difference is the PRIOR_CONTEXT payload.
    base_instruction = """
You are an autonomous software-engineering agent.

Your task is to resolve the CURRENT_ISSUE in the current repository by inspecting the code, making the necessary edits, and validating the change with appropriate tests.

A PRIOR_CONTEXT section is included below. It may contain read-only evidence from an officially related previous issue, or it may explicitly state that no prior context is provided for this run. If prior evidence is present, it is optional context only. It may contain irrelevant details, failed attempts, incomplete evidence, or edits that do not apply to the current repository state.

Use PRIOR_CONTEXT only if helpful. The CURRENT_ISSUE and current repository state are authoritative. Independently inspect the current repository and verify all conclusions. Do not copy prior patches blindly.
"""

    if condition == "no_memory":
        prior_block = f"""
PRIOR_CONTEXT
Condition: no_memory

{prior_context_label(condition)}
END_PRIOR_CONTEXT
"""
    else:
        prior_block = f"""
PRIOR_CONTEXT
Condition: {condition}

{prior_context_label(condition)}

{injection_text}
END_PRIOR_CONTEXT
"""

    current_issue_block = f"""
CURRENT_TARGET
Repository: {target_repo}
Base commit: {target_base_commit}
Instance ID: {target_instance_id}

CURRENT_ISSUE
{target_problem_statement}
END_CURRENT_ISSUE
"""

    final_instruction = """
FINAL_INSTRUCTION
Fix the CURRENT_ISSUE in the current repository. Do not modify unrelated behavior. Run relevant tests when practical, and leave the repository with the intended patch applied.
"""

    prompt = normalize_text(
        base_instruction + "\n\n" + prior_block + "\n\n" + current_issue_block + "\n\n" + final_instruction
    ) + "\n"

    meta: dict[str, Any] = {
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "run_id": run_id,
        "condition": condition,
        "target_instance_id": target_instance_id,
        "target_repo": target_repo,
        "target_base_commit": target_base_commit,
        "target_problem_statement_sha256": sha256_text(target_problem_statement),
        "sandbox_image": sandbox_image,
        "output_dir": output_dir_value,
        "prior_instance_id": prior_instance_id,
        "prior_repo": prior_repo,
        "prior_trajectory_id": prior_trajectory_id,
        "injection_file": injection_file_value or "",
        "injection_file_resolved": injection_path_resolved,
        "injection_sha256": injection_sha256,
        "prior_context_chars": len(injection_text),
        "prompt_chars": len(prompt),
        "prompt_sha256": sha256_text(prompt),
    }
    return prompt, meta


# ---------------------------------------------------------------------------
# Validation / reporting
# ---------------------------------------------------------------------------


def scan_forbidden(text: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    lines = text.split("\n")
    for name, pattern in FORBIDDEN_PATTERNS:
        for i, line in enumerate(lines, start=1):
            if pattern.search(line):
                excerpt = line.strip()
                if len(excerpt) > 240:
                    excerpt = excerpt[:240] + "..."
                hits.append({"pattern": name, "line": i, "excerpt": excerpt})
                break
    return hits


def length_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/contextbench_phase2/run_manifest.jsonl"))
    parser.add_argument("--root-dir", type=Path, default=Path("."), help="Repository root for resolving manifest-relative paths.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/contextbench_phase2/prompts"))
    parser.add_argument("--prompt-manifest", type=Path, default=Path("data/contextbench_phase2/prompt_manifest.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/contextbench_phase2/prompt_render_report.json"))
    parser.add_argument("--forbidden-scan", type=Path, default=Path("data/contextbench_phase2/forbidden_prompt_scan.txt"))
    parser.add_argument("--write-run-prompt", action=argparse.BooleanOptionalAction, default=True, help="Also write prompt.txt into each row's output_dir.")
    parser.add_argument("--fail-on-forbidden", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-prompt-chars", type=int, default=0, help="If >0, fail when any prompt exceeds this many characters.")
    args = parser.parse_args(argv)

    root_dir = args.root_dir.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else (root_dir / args.manifest)
    manifest = manifest.resolve()
    manifest_dir = manifest.parent

    if not manifest.exists():
        raise FileNotFoundError(manifest)

    args.out_dir = (root_dir / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    args.prompt_manifest = (root_dir / args.prompt_manifest).resolve() if not args.prompt_manifest.is_absolute() else args.prompt_manifest.resolve()
    args.report = (root_dir / args.report).resolve() if not args.report.is_absolute() else args.report.resolve()
    args.forbidden_scan = (root_dir / args.forbidden_scan).resolve() if not args.forbidden_scan.is_absolute() else args.forbidden_scan.resolve()

    prompt_rows: list[dict[str, Any]] = []
    forbidden_rows: list[str] = []
    condition_counts: Counter[str] = Counter()
    target_to_conditions: dict[str, set[str]] = defaultdict(set)
    target_to_issue_hashes: dict[str, set[str]] = defaultdict(set)
    prompt_paths_seen: set[str] = set()
    duplicate_prompt_paths: list[str] = []
    too_long: list[dict[str, Any]] = []
    prompt_lengths_by_condition: dict[str, list[int]] = defaultdict(list)
    prior_lengths_by_condition: dict[str, list[int]] = defaultdict(list)

    for source_line, row in iter_jsonl(manifest):
        prompt, meta = render_prompt(row, root_dir=root_dir, manifest_dir=manifest_dir)

        run_id = str(meta["run_id"])
        condition = str(meta["condition"])
        target_id = str(meta["target_instance_id"])

        condition_counts[condition] += 1
        target_to_conditions[target_id].add(condition)
        target_to_issue_hashes[target_id].add(str(meta["target_problem_statement_sha256"]))
        prompt_lengths_by_condition[condition].append(int(meta["prompt_chars"]))
        prior_lengths_by_condition[condition].append(int(meta["prior_context_chars"]))

        target_component = safe_path_component(target_id)
        condition_component = safe_path_component(condition)
        prompt_path = args.out_dir / target_component / condition_component / "prompt.txt"
        rel_prompt_path = relpath_or_str(prompt_path, root_dir)

        if rel_prompt_path in prompt_paths_seen:
            duplicate_prompt_paths.append(rel_prompt_path)
        prompt_paths_seen.add(rel_prompt_path)

        write_text_utf8(prompt_path, prompt)

        run_prompt_path_value = ""
        if args.write_run_prompt and meta.get("output_dir"):
            output_dir = Path(str(meta["output_dir"]))
            if not output_dir.is_absolute():
                output_dir = root_dir / output_dir
            run_prompt_path = output_dir / "prompt.txt"
            write_text_utf8(run_prompt_path, prompt)
            run_prompt_path_value = relpath_or_str(run_prompt_path, root_dir)

        hits = scan_forbidden(prompt)
        if hits:
            for hit in hits:
                forbidden_rows.append(
                    f"{rel_prompt_path}\t{hit['pattern']}\tline={hit['line']}\t{hit['excerpt']}"
                )

        if args.max_prompt_chars > 0 and len(prompt) > args.max_prompt_chars:
            too_long.append({
                "run_id": run_id,
                "condition": condition,
                "target_instance_id": target_id,
                "prompt_chars": len(prompt),
                "max_prompt_chars": args.max_prompt_chars,
            })

        meta["source_manifest_line"] = source_line
        meta["prompt_path"] = rel_prompt_path
        meta["run_prompt_path"] = run_prompt_path_value
        prompt_rows.append(meta)

    # Stable prompt manifest ordering.
    prompt_rows_sorted = sorted(prompt_rows, key=lambda r: (r["target_instance_id"], r["condition"], r["run_id"]))
    args.prompt_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.prompt_manifest.open("w", encoding="utf-8", newline="\n") as f:
        for row in prompt_rows_sorted:
            f.write(stable_json_dumps(row) + "\n")

    write_text_utf8(args.forbidden_scan, ("\n".join(forbidden_rows) + "\n") if forbidden_rows else "")

    bad_condition_sets = {
        target: sorted(conditions)
        for target, conditions in sorted(target_to_conditions.items())
        if conditions != EXPECTED_CONDITION_SET
    }
    inconsistent_target_issue_hashes = {
        target: sorted(hashes)
        for target, hashes in sorted(target_to_issue_hashes.items())
        if len(hashes) != 1
    }

    report = {
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "input_manifest": relpath_or_str(manifest, root_dir),
        "input_manifest_sha256": sha256_text(read_text_utf8(manifest)),
        "prompt_count": len(prompt_rows),
        "unique_targets": len(target_to_conditions),
        "expected_conditions": list(EXPECTED_CONDITIONS),
        "condition_counts": dict(sorted(condition_counts.items())),
        "bad_condition_sets": bad_condition_sets,
        "inconsistent_target_issue_hashes": inconsistent_target_issue_hashes,
        "duplicate_prompt_paths": sorted(duplicate_prompt_paths),
        "forbidden_hit_count": len(forbidden_rows),
        "too_long_count": len(too_long),
        "too_long": too_long[:50],
        "prompt_length_by_condition": {
            condition: length_summary(values)
            for condition, values in sorted(prompt_lengths_by_condition.items())
        },
        "prior_context_length_by_condition": {
            condition: length_summary(values)
            for condition, values in sorted(prior_lengths_by_condition.items())
        },
        "prompt_manifest": relpath_or_str(args.prompt_manifest, root_dir),
        "prompt_manifest_sha256": sha256_text("".join(stable_json_dumps(r) + "\n" for r in prompt_rows_sorted)),
        "prompt_out_dir": relpath_or_str(args.out_dir, root_dir),
        "forbidden_scan": relpath_or_str(args.forbidden_scan, root_dir),
        "write_run_prompt": bool(args.write_run_prompt),
    }

    write_text_utf8(args.report, stable_json_dumps(report, pretty=True) + "\n")
    print(stable_json_dumps(report, pretty=True))

    failed = False
    if bad_condition_sets:
        print("ERROR: some targets do not have exactly the expected four conditions", file=sys.stderr)
        failed = True
    if inconsistent_target_issue_hashes:
        print("ERROR: target issue text differs across conditions for at least one target", file=sys.stderr)
        failed = True
    if duplicate_prompt_paths:
        print("ERROR: duplicate prompt output paths", file=sys.stderr)
        failed = True
    if too_long:
        print("ERROR: some prompts exceed --max-prompt-chars", file=sys.stderr)
        failed = True
    if forbidden_rows and args.fail_on_forbidden:
        print(f"ERROR: forbidden prompt hits found: {len(forbidden_rows)}", file=sys.stderr)
        print(f"See: {relpath_or_str(args.forbidden_scan, root_dir)}", file=sys.stderr)
        failed = True

    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
