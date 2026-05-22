#!/usr/bin/env python3
"""
Run OpenHands on ContextBench/SWEContextBench prompt manifests.

This orchestrator supports both smoke and full execution through the same code path.
Smoke mode is only a row-selection mode; workspace preparation, OpenHands execution,
patch extraction, and prediction writing are identical to full mode.

Design:
  - Load prompt_manifest.jsonl.
  - Select rows by mode/target/condition.
  - For each selected row:
      1. Seed a local isolated workspace from row["sandbox_image"]:/testbed.
      2. Verify git HEAD matches row["target_base_commit"].
      3. Copy the exact rendered prompt into the run directory.
      4. Run OpenHands headless from inside the workspace.
      5. Stage all changes and collect a git patch.
      6. Write strict prediction JSON and audit metadata.
  - Write condition-separated prediction JSONL files so evaluators do not see
    duplicate instance IDs across experimental conditions.

Assumptions confirmed by probes:
  - OpenHands CLI edits the host current working directory.
  - SWEContextBench images contain the target checkout at /testbed.
  - /testbed is a clean git checkout at target_base_commit.

Example smoke dry-run:
  python scripts/contextbench/run_openhands_contextbench.py \
    --manifest data/contextbench_phase2/prompt_manifest.jsonl \
    --mode smoke \
    --dry-run

Example smoke run:
  python scripts/contextbench/run_openhands_contextbench.py \
    --manifest data/contextbench_phase2/prompt_manifest.jsonl \
    --mode smoke

Example full dry-run:
  python scripts/contextbench/run_openhands_contextbench.py \
    --manifest data/contextbench_phase2/prompt_manifest.jsonl \
    --mode full \
    --dry-run

Example full run:
  python scripts/contextbench/run_openhands_contextbench.py \
    --manifest data/contextbench_phase2/prompt_manifest.jsonl \
    --mode full \
    --confirm-full-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "contextbench_openhands_orchestrator_v1.0.0"
DEFAULT_CONDITIONS = ["no_memory", "raw", "adp", "memory"]
DEFAULT_SMOKE_TARGET = "astropy__astropy-15082"
DEFAULT_EXECUTION_ROOT = Path("data/contextbench_phase2/execution")
DEFAULT_MODEL_NAME = "openhands-qwen3-coder-30b-ollama-contextbench-memory-repr"
DEFAULT_LLM_MODEL = "openai/qwen3-coder:30b"
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11435/v1"
DEFAULT_LLM_API_KEY = "dummy"
REQUIRED_ROW_FIELDS = [
    "condition",
    "prompt_path",
    "prompt_sha256",
    "run_id",
    "sandbox_image",
    "target_base_commit",
    "target_instance_id",
    "target_repo",
]


class OrchestratorError(RuntimeError):
    """Expected orchestration error with a user-actionable message."""


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    cwd: str | None
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool = False


@dataclass(frozen=True)
class StreamingCommandResult:
    argv: list[str]
    cwd: str | None
    exit_code: int
    stdout_path: str
    stderr_path: str
    elapsed_seconds: float
    timed_out: bool = False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(obj) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise OrchestratorError(f"JSONL file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise OrchestratorError(f"Invalid JSON on {path}:{lineno}: {e}") from e
            if not isinstance(row, dict):
                raise OrchestratorError(f"Expected JSON object on {path}:{lineno}")
            row["_manifest_lineno"] = lineno
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_capture(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: int | None = None,
    check: bool = False,
) -> CommandResult:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        result = CommandResult(
            argv=argv,
            cwd=str(cwd) if cwd else None,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed_seconds=round(time.monotonic() - start, 3),
            timed_out=False,
        )
    except subprocess.TimeoutExpired as e:
        result = CommandResult(
            argv=argv,
            cwd=str(cwd) if cwd else None,
            exit_code=124,
            stdout=e.stdout or "",
            stderr=e.stderr or "",
            elapsed_seconds=round(time.monotonic() - start, 3),
            timed_out=True,
        )

    if check and result.exit_code != 0:
        raise OrchestratorError(
            "Command failed:\n"
            f"  cwd: {result.cwd}\n"
            f"  argv: {' '.join(result.argv)}\n"
            f"  exit: {result.exit_code}\n"
            f"  stdout:\n{result.stdout[-4000:]}\n"
            f"  stderr:\n{result.stderr[-4000:]}"
        )
    return result


def run_streaming(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int | None = None,
) -> StreamingCommandResult:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    timed_out = False

    with stdout_path.open("w", encoding="utf-8", errors="replace") as out_f, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as err_f:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdout=out_f,
            stderr=err_f,
            text=True,
        )
        try:
            exit_code = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            exit_code = proc.wait()
            err_f.write(f"\n[orchestrator] TIMEOUT after {timeout_seconds} seconds\n")

    return StreamingCommandResult(
        argv=argv,
        cwd=str(cwd),
        exit_code=exit_code if not timed_out else 124,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        elapsed_seconds=round(time.monotonic() - start, 3),
        timed_out=timed_out,
    )


def require_executable(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise OrchestratorError(f"Required executable not found on PATH: {name}")
    return found


def condition_sort_key(condition: str, condition_order: list[str]) -> tuple[int, str]:
    try:
        return (condition_order.index(condition), condition)
    except ValueError:
        return (999, condition)


def load_and_select_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.manifest)

    for row in rows:
        missing = [k for k in REQUIRED_ROW_FIELDS if k not in row]
        if missing:
            line = row.get("_manifest_lineno", "?")
            raise OrchestratorError(f"Manifest row {line} is missing required fields: {missing}")

    allowed_conditions = set(args.conditions)
    rows = [r for r in rows if r["condition"] in allowed_conditions]

    if args.mode == "smoke":
        rows = [r for r in rows if r["target_instance_id"] == args.smoke_target]
        expected = set(args.conditions)
        actual = {r["condition"] for r in rows}
        if actual != expected:
            raise OrchestratorError(
                f"Smoke selection for target {args.smoke_target} has conditions {sorted(actual)}, "
                f"expected {sorted(expected)}"
            )
    elif args.mode == "full":
        if not args.dry_run and not args.confirm_full_run:
            raise OrchestratorError(
                "Refusing to run full mode without --confirm-full-run. "
                "Use --dry-run first, then pass --confirm-full-run deliberately."
            )
    else:
        raise OrchestratorError(f"Unknown mode: {args.mode}")

    seen_pairs: set[tuple[str, str]] = set()
    duplicates: list[tuple[str, str]] = []
    for row in rows:
        pair = (row["target_instance_id"], row["condition"])
        if pair in seen_pairs:
            duplicates.append(pair)
        seen_pairs.add(pair)
    if duplicates:
        raise OrchestratorError(f"Duplicate target/condition pairs selected: {duplicates[:10]}")

    rows.sort(
        key=lambda r: (
            r["target_instance_id"],
            condition_sort_key(r["condition"], args.conditions),
            r["run_id"],
        )
    )

    if args.max_runs is not None:
        rows = rows[: args.max_runs]

    if not rows:
        raise OrchestratorError("No rows selected.")

    return rows


def validate_prompt_file(row: dict[str, Any], repo_root: Path) -> Path:
    prompt_path = repo_root / row["prompt_path"]
    if not prompt_path.exists():
        raise OrchestratorError(f"Prompt file does not exist: {prompt_path}")

    actual_sha = sha256_file(prompt_path)
    expected_sha = row.get("prompt_sha256")
    if expected_sha and actual_sha != expected_sha:
        raise OrchestratorError(
            f"Prompt SHA mismatch for {prompt_path}: expected {expected_sha}, got {actual_sha}"
        )

    text = prompt_path.read_text(encoding="utf-8", errors="replace")
    expected_condition_marker = f"Condition: {row['condition']}"
    if expected_condition_marker not in text:
        raise OrchestratorError(
            f"Prompt {prompt_path} does not contain expected marker: {expected_condition_marker!r}"
        )
    if row["target_instance_id"] not in text:
        raise OrchestratorError(
            f"Prompt {prompt_path} does not appear to contain target instance id {row['target_instance_id']}"
        )

    return prompt_path


def run_dir_for(execution_root: Path, row: dict[str, Any]) -> Path:
    return execution_root / "runs" / row["target_instance_id"] / row["condition"]


def success_meta_exists(run_dir: Path) -> bool:
    meta_path = run_dir / "run_meta.json"
    prediction_path = run_dir / "prediction.json"
    if not meta_path.exists() or not prediction_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(meta.get("orchestrator_success")) and meta.get("status") in {
        "success",
        "success_empty_patch",
    }


def build_run_plan(rows: list[dict[str, Any]], args: argparse.Namespace, repo_root: Path) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for row in rows:
        prompt_path = validate_prompt_file(row, repo_root)
        rd = run_dir_for(args.execution_root, row)
        plan.append(
            {
                "run_id": row["run_id"],
                "target_instance_id": row["target_instance_id"],
                "condition": row["condition"],
                "target_repo": row["target_repo"],
                "target_base_commit": row["target_base_commit"],
                "sandbox_image": row["sandbox_image"],
                "prompt_path": str(prompt_path),
                "prompt_sha256": row["prompt_sha256"],
                "run_dir": str(rd),
                "workspace_dir": str(rd / "workspace"),
                "manifest_lineno": row.get("_manifest_lineno"),
            }
        )
    return plan


def seed_workspace_from_image(
    *,
    sandbox_image: str,
    workspace_dir: Path,
    image_repo_path: str,
    pulled_images: set[str],
    pull_images: bool,
) -> None:
    workspace_dir.mkdir(parents=True, exist_ok=True)

    if pull_images and sandbox_image not in pulled_images:
        print(f"[pull] {sandbox_image}", flush=True)
        run_capture(["docker", "pull", sandbox_image], check=True)
        pulled_images.add(sandbox_image)

    create = run_capture(["docker", "create", sandbox_image], check=True)
    container_id = create.stdout.strip()
    if not container_id:
        raise OrchestratorError(f"docker create returned no container id for image {sandbox_image}")

    try:
        src = f"{container_id}:{image_repo_path.rstrip('/')}/."
        print(f"[seed] {sandbox_image}:{image_repo_path} -> {workspace_dir}", flush=True)
        run_capture(["docker", "cp", src, str(workspace_dir)], check=True)
    finally:
        run_capture(["docker", "rm", "-f", container_id], check=False)


def git_output(workspace_dir: Path, args: list[str], *, check: bool = True) -> CommandResult:
    return run_capture(["git", *args], cwd=workspace_dir, check=check)


def verify_workspace(row: dict[str, Any], workspace_dir: Path) -> dict[str, Any]:
    head = git_output(workspace_dir, ["rev-parse", "HEAD"], check=True).stdout.strip()
    expected = row["target_base_commit"]
    if head != expected:
        raise OrchestratorError(
            f"Workspace HEAD mismatch for {row['run_id']}: expected {expected}, got {head}"
        )

    status = git_output(workspace_dir, ["status", "--short"], check=True).stdout
    if status.strip():
        raise OrchestratorError(
            f"Seeded workspace is not clean for {row['run_id']}:\n{status[:4000]}"
        )

    return {"git_head": head, "initial_git_status_short": status}


def build_openhands_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "LLM_MODEL": args.llm_model,
            "LLM_BASE_URL": args.llm_base_url,
            "LLM_API_KEY": args.llm_api_key,
            "OPENHANDS_SUPPRESS_BANNER": "1",
        }
    )
    return env


def env_summary(env: dict[str, str]) -> dict[str, Any]:
    keys = ["LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "OPENHANDS_SUPPRESS_BANNER"]
    summary: dict[str, Any] = {}
    for k in keys:
        if k == "LLM_API_KEY":
            summary[k + "_PRESENT"] = bool(env.get(k))
        else:
            summary[k] = env.get(k, "")
    return summary


def collect_patch(workspace_dir: Path, patch_path: Path) -> str:
    # Stage all changes so newly-created files and deletions are included.
    git_output(workspace_dir, ["add", "-A"], check=True)
    diff = git_output(
        workspace_dir,
        ["diff", "--cached", "--binary", "--no-ext-diff"],
        check=True,
    ).stdout
    patch_path.write_text(diff, encoding="utf-8", errors="replace")
    return diff


def strict_prediction(row: dict[str, Any], model_name: str, patch: str) -> dict[str, Any]:
    # Keep this strict for evaluator compatibility.
    return {
        "instance_id": row["target_instance_id"],
        "model_name_or_path": model_name,
        "model_patch": patch,
    }


def audit_prediction(row: dict[str, Any], model_name: str, patch: str) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "condition": row["condition"],
        "target_instance_id": row["target_instance_id"],
        "target_repo": row["target_repo"],
        "target_base_commit": row["target_base_commit"],
        "sandbox_image": row["sandbox_image"],
        "model_name_or_path": model_name,
        "model_patch": patch,
    }


def write_prediction_files(
    *,
    row: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    patch: str,
) -> dict[str, str]:
    pred = strict_prediction(row, args.model_name_or_path, patch)
    audit_pred = audit_prediction(row, args.model_name_or_path, patch)

    prediction_path = run_dir / "prediction.json"
    audit_prediction_path = run_dir / "prediction_audit.json"
    write_json(prediction_path, pred)
    write_json(audit_prediction_path, audit_pred)

    # Convenience export for workflows that expect a per-instance JSON object.
    contextbench_dir = (
        args.execution_root
        / "predictions"
        / "contextbench_files"
        / args.mode
        / row["condition"]
    )
    contextbench_file = contextbench_dir / f"{row['target_instance_id']}_preds.json"
    write_json(contextbench_file, {row["target_instance_id"]: pred})

    return {
        "prediction_path": str(prediction_path),
        "prediction_audit_path": str(audit_prediction_path),
        "contextbench_prediction_file": str(contextbench_file),
    }


def materialize_prompt(prompt_src: Path, run_dir: Path) -> Path:
    dst = run_dir / "prompt.txt"
    shutil.copy2(prompt_src, dst)
    return dst


def prepare_run_dir(run_dir: Path, *, force: bool, resume: bool) -> str:
    if run_dir.exists():
        if resume and success_meta_exists(run_dir) and not force:
            return "skip_success"
        if force or resume:
            shutil.rmtree(run_dir)
        else:
            raise OrchestratorError(
                f"Run directory already exists: {run_dir}\n"
                "Use --resume to skip successful runs/rerun failed ones, or --force to rerun all selected runs."
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    return "run"


def execute_one(row: dict[str, Any], args: argparse.Namespace, repo_root: Path, pulled_images: set[str]) -> dict[str, Any]:
    run_dir = run_dir_for(args.execution_root, row)
    prep = prepare_run_dir(run_dir, force=args.force, resume=args.resume)
    if prep == "skip_success":
        print(f"[skip] {row['run_id']} already has successful run_meta.json and prediction.json", flush=True)
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        return {"status": "skipped_success", "run_id": row["run_id"], "run_dir": str(run_dir), "meta": meta}

    workspace_dir = run_dir / "workspace"
    stdout_path = run_dir / "stdout.jsonl"
    stderr_path = run_dir / "stderr.log"
    patch_path = run_dir / "patch.diff"
    command_path = run_dir / "command.json"
    meta_path = run_dir / "run_meta.json"
    prompt_src = validate_prompt_file(row, repo_root)
    prompt_dst = materialize_prompt(prompt_src, run_dir)

    start_iso = utc_now_iso()
    meta: dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "status": "started",
        "orchestrator_success": False,
        "started_at": start_iso,
        "finished_at": None,
        "mode": args.mode,
        "run_id": row["run_id"],
        "target_instance_id": row["target_instance_id"],
        "condition": row["condition"],
        "target_repo": row["target_repo"],
        "target_base_commit": row["target_base_commit"],
        "sandbox_image": row["sandbox_image"],
        "image_repo_path": args.image_repo_path,
        "manifest_lineno": row.get("_manifest_lineno"),
        "manifest_prompt_path": row["prompt_path"],
        "manifest_prompt_sha256": row.get("prompt_sha256", ""),
        "run_dir": str(run_dir),
        "workspace_dir": str(workspace_dir),
        "prompt_path": str(prompt_dst),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "patch_path": str(patch_path),
        "prediction_path": str(run_dir / "prediction.json"),
        "model_name_or_path": args.model_name_or_path,
    }
    write_json(meta_path, meta)

    try:
        seed_workspace_from_image(
            sandbox_image=row["sandbox_image"],
            workspace_dir=workspace_dir,
            image_repo_path=args.image_repo_path,
            pulled_images=pulled_images,
            pull_images=not args.no_pull,
        )
        workspace_info = verify_workspace(row, workspace_dir)
        meta.update(workspace_info)

        openhands_argv = [
            args.openhands_bin,
            "--headless",
            "--json",
            "--file",
            str(prompt_dst.resolve()),
            "--override-with-envs",
        ]
        oh_env = build_openhands_env(args)
        write_json(
            command_path,
            {
                "argv": openhands_argv,
                "cwd": str(workspace_dir),
                "env_summary": env_summary(oh_env),
            },
        )

        print(f"[run] {row['run_id']} cwd={workspace_dir}", flush=True)
        result = run_streaming(
            openhands_argv,
            cwd=workspace_dir,
            env=oh_env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=args.timeout_seconds,
        )
        meta["openhands_result"] = {
            "argv": result.argv,
            "cwd": result.cwd,
            "exit_code": result.exit_code,
            "stdout_path": result.stdout_path,
            "stderr_path": result.stderr_path,
            "elapsed_seconds": result.elapsed_seconds,
            "timed_out": result.timed_out,
        }

        patch = collect_patch(workspace_dir, patch_path)
        pred_paths = write_prediction_files(row=row, args=args, run_dir=run_dir, patch=patch)
        meta.update(pred_paths)
        meta["patch_sha256"] = hashlib.sha256(patch.encode("utf-8", errors="replace")).hexdigest()
        meta["patch_chars"] = len(patch)
        meta["patch_empty"] = patch == ""

        if result.exit_code == 0:
            meta["status"] = "success_empty_patch" if patch == "" else "success"
            meta["orchestrator_success"] = True
        else:
            meta["status"] = "openhands_failed"
            meta["orchestrator_success"] = False

    except Exception as e:
        meta["status"] = "orchestrator_exception"
        meta["orchestrator_success"] = False
        meta["exception_type"] = type(e).__name__
        meta["exception"] = str(e)
        write_json(meta_path, meta)
        raise
    finally:
        meta["finished_at"] = utc_now_iso()
        write_json(meta_path, meta)

    print(f"[done] {row['run_id']} status={meta['status']} patch_chars={meta.get('patch_chars')}", flush=True)
    return {"status": meta["status"], "run_id": row["run_id"], "run_dir": str(run_dir), "meta": meta}


def read_prediction_if_exists(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / "prediction.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def read_audit_prediction_if_exists(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / "prediction_audit.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def write_aggregate_predictions(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    pred_root = args.execution_root / "predictions"
    audit_rows: list[dict[str, Any]] = []
    by_condition: dict[str, list[dict[str, Any]]] = {c: [] for c in args.conditions}
    missing: list[dict[str, str]] = []

    for row in rows:
        rd = run_dir_for(args.execution_root, row)
        pred = read_prediction_if_exists(rd)
        audit_pred = read_audit_prediction_if_exists(rd)
        if pred is None:
            missing.append({"run_id": row["run_id"], "prediction_path": str(rd / "prediction.json")})
            continue
        by_condition.setdefault(row["condition"], []).append(pred)
        if audit_pred is not None:
            audit_rows.append(audit_pred)
        else:
            audit_rows.append(
                {
                    "run_id": row["run_id"],
                    "condition": row["condition"],
                    "target_instance_id": row["target_instance_id"],
                    "prediction_missing_audit": True,
                }
            )

    audit_path = pred_root / f"{args.mode}_all_audit_predictions.jsonl"
    write_jsonl(audit_path, audit_rows)

    condition_paths: dict[str, str] = {}
    duplicate_warnings: dict[str, list[str]] = {}
    for condition, preds in by_condition.items():
        if not preds:
            continue
        seen: set[str] = set()
        dups: list[str] = []
        for pred in preds:
            iid = pred.get("instance_id", "")
            if iid in seen:
                dups.append(iid)
            seen.add(iid)
        if dups:
            duplicate_warnings[condition] = sorted(set(dups))

        condition_path = pred_root / f"{args.mode}_{condition}_predictions.jsonl"
        write_jsonl(condition_path, preds)
        condition_paths[condition] = str(condition_path)

    return {
        "audit_predictions_path": str(audit_path),
        "condition_prediction_paths": condition_paths,
        "missing_predictions": missing,
        "duplicate_instance_ids_by_condition": duplicate_warnings,
    }


def write_report(
    *,
    rows: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    results: list[dict[str, Any]],
    prediction_report: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    report_root = args.execution_root / "reports"
    report_path = report_root / f"{args.mode}_run_report.json"

    status_counts: dict[str, int] = {}
    for result in results:
        status = result.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    selected_counts: dict[str, int] = {}
    for row in rows:
        selected_counts[row["condition"]] = selected_counts.get(row["condition"], 0) + 1

    report = {
        "script_version": SCRIPT_VERSION,
        "mode": args.mode,
        "created_at": utc_now_iso(),
        "manifest": str(args.manifest),
        "execution_root": str(args.execution_root),
        "selected_run_count": len(rows),
        "selected_condition_counts": selected_counts,
        "status_counts": status_counts,
        "dry_run": args.dry_run,
        "model_name_or_path": args.model_name_or_path,
        "llm_model": args.llm_model,
        "llm_base_url": args.llm_base_url,
        "llm_api_key_present": bool(args.llm_api_key),
        "openhands_bin": args.openhands_bin,
        "image_repo_path": args.image_repo_path,
        "timeout_seconds": args.timeout_seconds,
        "plan_path": str(args.execution_root / "plans" / f"{args.mode}_run_plan.json"),
        "prediction_report": prediction_report,
        "results": results,
        "plan_preview": plan[:10],
    }
    write_json(report_path, report)
    return report_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenHands ContextBench experiments.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/contextbench_phase2/prompt_manifest.jsonl"),
        help="Prompt manifest JSONL. Prefer data/contextbench_phase2/prompt_manifest.jsonl.",
    )
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--smoke-target", default=DEFAULT_SMOKE_TARGET)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    parser.add_argument("--image-repo-path", default="/testbed")
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL))
    parser.add_argument("--llm-base-url", default=os.environ.get("LLM_BASE_URL", DEFAULT_LLM_BASE_URL))
    parser.add_argument("--llm-api-key", default=os.environ.get("LLM_API_KEY", DEFAULT_LLM_API_KEY))
    parser.add_argument("--openhands-bin", default=os.environ.get("OPENHANDS_BIN", "openhands"))
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-pull", action="store_true", help="Do not docker pull images before seeding workspaces.")
    parser.add_argument(
        "--stop-on-error",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: true for smoke, false for full.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root = Path.cwd()
    args.manifest = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    args.execution_root = args.execution_root if args.execution_root.is_absolute() else repo_root / args.execution_root
    if args.stop_on_error is None:
        args.stop_on_error = args.mode == "smoke"

    try:
        require_executable("docker")
        args.openhands_bin = require_executable(args.openhands_bin)
        require_executable("git")

        rows = load_and_select_rows(args)
        plan = build_run_plan(rows, args, repo_root)

        args.execution_root.mkdir(parents=True, exist_ok=True)
        plan_path = args.execution_root / "plans" / f"{args.mode}_run_plan.json"
        write_json(
            plan_path,
            {
                "script_version": SCRIPT_VERSION,
                "created_at": utc_now_iso(),
                "mode": args.mode,
                "manifest": str(args.manifest),
                "execution_root": str(args.execution_root),
                "selected_run_count": len(rows),
                "conditions": args.conditions,
                "smoke_target": args.smoke_target if args.mode == "smoke" else None,
                "dry_run": args.dry_run,
                "plan": plan,
            },
        )
        print(f"[plan] wrote {plan_path}", flush=True)
        print(f"[plan] selected {len(rows)} runs", flush=True)
        for item in plan[:20]:
            print(
                f"  - {item['run_id']} image={item['sandbox_image']} prompt={item['prompt_path']}",
                flush=True,
            )
        if len(plan) > 20:
            print(f"  ... {len(plan) - 20} more", flush=True)

        if args.dry_run:
            prediction_report = {
                "audit_predictions_path": None,
                "condition_prediction_paths": {},
                "missing_predictions": [],
                "duplicate_instance_ids_by_condition": {},
                "dry_run": True,
            }
            report_path = write_report(
                rows=rows,
                plan=plan,
                results=[],
                prediction_report=prediction_report,
                args=args,
            )
            print(f"[dry-run] wrote report {report_path}", flush=True)
            return 0

        pulled_images: set[str] = set()
        results: list[dict[str, Any]] = []
        had_error = False

        for idx, row in enumerate(rows, start=1):
            print(f"\n=== [{idx}/{len(rows)}] {row['run_id']} ===", flush=True)
            try:
                result = execute_one(row, args, repo_root, pulled_images)
                results.append(result)
                if result.get("status") not in {"success", "success_empty_patch", "skipped_success"}:
                    had_error = True
                    if args.stop_on_error:
                        print(f"[stop] stopping after non-success status: {result.get('status')}", flush=True)
                        break
            except Exception as e:
                had_error = True
                results.append(
                    {
                        "status": "orchestrator_exception",
                        "run_id": row.get("run_id"),
                        "exception_type": type(e).__name__,
                        "exception": str(e),
                    }
                )
                print(f"[error] {row.get('run_id')}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                if args.stop_on_error:
                    break

        prediction_report = write_aggregate_predictions(rows, args)
        report_path = write_report(
            rows=rows,
            plan=plan,
            results=results,
            prediction_report=prediction_report,
            args=args,
        )
        print(f"\n[report] wrote {report_path}", flush=True)
        print(f"[predictions] {json_dumps(prediction_report)}", flush=True)

        if had_error:
            print("[done] completed with one or more errors", file=sys.stderr, flush=True)
            return 1
        print("[done] completed successfully", flush=True)
        return 0

    except OrchestratorError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
