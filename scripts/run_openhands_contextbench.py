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
  - In host runtime, OpenHands CLI edits the host current working directory.
  - SWEContextBench images contain the target checkout at /testbed.
  - /testbed is a clean git checkout at target_base_commit.
  - Image runtime is used only when sandbox_image exposes the OpenHands CLI.

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
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "contextbench_openhands_orchestrator_v1.1.0"
DEFAULT_CONDITIONS = ["no_memory", "raw", "adp", "memory"]
DEFAULT_SMOKE_TARGET = "astropy__astropy-15082"
DEFAULT_EXECUTION_ROOT = Path("data/contextbench_phase2/execution")
DEFAULT_MODEL_NAME = "openhands-qwen3-coder-30b-ollama-contextbench-memory-repr"
DEFAULT_LLM_MODEL = "openai/qwen3-coder:30b"
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11435/v1"
DEFAULT_LLM_API_KEY = "dummy"
DEFAULT_OPENHANDS_RUNTIME = "auto"
DEFAULT_CONTAINER_RUN_DIR = "/contextbench_run"
DEFAULT_IMAGE_OPENHANDS_BIN = "openhands"
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
RUN_SCOPED_DIRS = {
    "home": ".openhands_home",
    "tmp": ".tmp",
    "cache": ".cache",
    "state": ".state",
}
FORBIDDEN_PROMPT_PATTERNS = [
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "target_fail_to_pass",
    "target_pass_to_pass",
    "gold_patch",
    "test_patch",
    "model_patch",
    "diff --git",
    "<system-reminder>",
    "</system-reminder>",
    "reasoning_content",
    "TodoWrite",
    "_preds.json",
    ".claude",
    ".token_usage",
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
            stdin=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        try:
            exit_code = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            err_f.write(f"\n[orchestrator] TIMEOUT after {timeout_seconds} seconds\n")
            err_f.flush()
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                exit_code = proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                finally:
                    exit_code = proc.wait()

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

    hits = [p for p in FORBIDDEN_PROMPT_PATTERNS if p in text]
    if hits:
        raise OrchestratorError(
            f"Prompt {prompt_path} contains forbidden patterns: {hits}"
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


def resolve_loose(path: Path) -> Path:
    return path.resolve(strict=False)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        resolve_loose(path).relative_to(resolve_loose(parent))
        return True
    except ValueError:
        return False


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


def ensure_run_scoped_dirs(run_dir: Path) -> dict[str, Path]:
    paths = {name: run_dir / dirname for name, dirname in RUN_SCOPED_DIRS.items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_host_openhands_env(args: argparse.Namespace, *, run_dir: Path, workspace_dir: Path) -> dict[str, str]:
    scoped_dirs = ensure_run_scoped_dirs(run_dir)
    env = dict(os.environ)
    env.update(
        {
            "LLM_MODEL": args.llm_model,
            "LLM_BASE_URL": args.llm_base_url,
            "LLM_API_KEY": args.llm_api_key,
            "OPENHANDS_SUPPRESS_BANNER": "1",
            "HOME": str(scoped_dirs["home"]),
            "TMPDIR": str(scoped_dirs["tmp"]),
            "XDG_CACHE_HOME": str(scoped_dirs["cache"]),
            "XDG_STATE_HOME": str(scoped_dirs["state"]),
            "CONTEXTBENCH_WORKSPACE_ROOT": str(workspace_dir.resolve()),
            "PYTHONNOUSERSITE": "1",
        }
    )
    return env


def build_docker_cli_env(args: argparse.Namespace) -> dict[str, str]:
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


def env_overrides_for_metadata(env: dict[str, str]) -> dict[str, str]:
    keys = [
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "OPENHANDS_SUPPRESS_BANNER",
        "HOME",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "XDG_STATE_HOME",
        "CONTEXTBENCH_WORKSPACE_ROOT",
        "PYTHONNOUSERSITE",
    ]
    out = {
        "LLM_MODEL": env.get("LLM_MODEL", ""),
        "LLM_BASE_URL": env.get("LLM_BASE_URL", ""),
        "LLM_API_KEY": "<redacted>" if env.get("LLM_API_KEY") else "",
    }
    for key in keys:
        if key in out:
            continue
        if key in env:
            out[key] = env[key]
    return out


def workspace_guard_allowed_files(run_dir: Path) -> set[Path]:
    names = {
        "prompt.txt",
        "stdout.jsonl",
        "stderr.log",
        "run_meta.json",
        "command.json",
        "patch.diff",
        "prediction.json",
        "prediction_audit.json",
    }
    return {resolve_loose(run_dir / name) for name in names}


def scan_off_workspace_writes(
    *,
    run_dir: Path,
    workspace_dir: Path,
    since_ns: int,
    limit: int = 200,
) -> list[str]:
    """Detect files modified after since_ns outside the current workspace.

    This is detection, not prevention. The scan is intentionally scoped to the
    target's execution directory so it catches run-local drift like ../test.py
    without walking arbitrary host paths.
    """
    scan_root = run_dir.parent
    allowed_files = workspace_guard_allowed_files(run_dir)
    scoped_dirs = ensure_run_scoped_dirs(run_dir)
    skipped_dirs = [resolve_loose(workspace_dir), *(resolve_loose(p) for p in scoped_dirs.values())]

    violations: list[str] = []
    if not scan_root.exists():
        return violations

    for root, dirs, files in os.walk(scan_root):
        root_path = Path(root)
        kept_dirs: list[str] = []
        for dirname in dirs:
            candidate = root_path / dirname
            if any(is_relative_to(candidate, skipped) for skipped in skipped_dirs):
                continue
            kept_dirs.append(dirname)
        dirs[:] = kept_dirs

        for filename in files:
            candidate = resolve_loose(root_path / filename)
            if candidate in allowed_files:
                continue
            if any(is_relative_to(candidate, skipped) for skipped in skipped_dirs):
                continue
            try:
                stat = candidate.stat()
            except FileNotFoundError:
                continue
            if stat.st_mtime_ns >= since_ns:
                try:
                    display = str(candidate.relative_to(run_dir.parent))
                except ValueError:
                    display = str(candidate)
                violations.append(display)
                if len(violations) >= limit:
                    return violations
    return violations


def base_openhands_args(args: argparse.Namespace, prompt_path: str) -> list[str]:
    openhands_args = [
        "--headless",
        "--json",
        "--file",
        prompt_path,
        "--override-with-envs",
    ]
    if args.always_approve:
        openhands_args.append("--always-approve")
    return openhands_args


def image_supports_openhands(
    *,
    sandbox_image: str,
    image_openhands_bin: str,
    cache: dict[tuple[str, str], bool],
) -> bool:
    key = (sandbox_image, image_openhands_bin)
    if key in cache:
        return cache[key]

    probe = run_capture(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "sh",
            sandbox_image,
            "-lc",
            f"command -v {shlex.quote(image_openhands_bin)} >/dev/null 2>&1",
        ],
        check=False,
    )
    cache[key] = probe.exit_code == 0
    return cache[key]


def choose_openhands_runtime(
    *,
    row: dict[str, Any],
    args: argparse.Namespace,
    image_openhands_cache: dict[tuple[str, str], bool],
) -> tuple[str, str]:
    requested = args.openhands_runtime
    if requested == "host":
        return "host", "forced_host"

    image_ready = image_supports_openhands(
        sandbox_image=row["sandbox_image"],
        image_openhands_bin=args.image_openhands_bin,
        cache=image_openhands_cache,
    )
    if requested == "image":
        if not image_ready:
            raise OrchestratorError(
                f"Image runtime requested, but {row['sandbox_image']} does not expose "
                f"{args.image_openhands_bin!r}. Install OpenHands in the image or use --openhands-runtime host."
            )
        return "image", "forced_image"

    if requested != "auto":
        raise OrchestratorError(f"Unknown OpenHands runtime: {requested}")

    if image_ready:
        return "image", "auto_image_available"
    if not args.host_openhands_available:
        raise OrchestratorError(
            f"Auto runtime could not use image {row['sandbox_image']} because "
            f"{args.image_openhands_bin!r} is not available, and host OpenHands is also unavailable."
        )
    return "host", "auto_image_missing_openhands"


def build_host_openhands_command(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    workspace_dir: Path,
    prompt_dst: Path,
) -> tuple[list[str], Path, dict[str, str], dict[str, Any]]:
    argv = [
        args.openhands_bin,
        *base_openhands_args(args, str(prompt_dst.resolve())),
    ]
    env = build_host_openhands_env(args, run_dir=run_dir, workspace_dir=workspace_dir)
    metadata = {
        "runtime": "host",
        "argv": argv,
        "cwd": str(workspace_dir),
        "env_overrides": env_overrides_for_metadata(env),
    }
    return argv, workspace_dir, env, metadata


def build_image_openhands_command(
    *,
    row: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    workspace_dir: Path,
) -> tuple[list[str], Path, dict[str, str], dict[str, Any]]:
    ensure_run_scoped_dirs(run_dir)
    container_run_dir = args.container_run_dir.rstrip("/")
    container_workspace = args.image_repo_path.rstrip("/")
    container_prompt = f"{container_run_dir}/prompt.txt"
    container_home = f"{container_run_dir}/{RUN_SCOPED_DIRS['home']}"
    container_tmp = f"{container_run_dir}/{RUN_SCOPED_DIRS['tmp']}"
    container_cache = f"{container_run_dir}/{RUN_SCOPED_DIRS['cache']}"
    container_state = f"{container_run_dir}/{RUN_SCOPED_DIRS['state']}"

    argv = ["docker", "run", "--rm"]
    if args.image_run_network:
        argv.extend(["--network", args.image_run_network])
    if args.image_run_as_current_user and hasattr(os, "getuid") and hasattr(os, "getgid"):
        argv.extend(["--user", f"{os.getuid()}:{os.getgid()}"])

    argv.extend(
        [
            "--workdir",
            container_workspace,
            "--mount",
            f"type=bind,src={workspace_dir.resolve()},dst={container_workspace}",
            "--mount",
            f"type=bind,src={run_dir.resolve()},dst={container_run_dir}",
            "--env",
            "LLM_MODEL",
            "--env",
            "LLM_BASE_URL",
            "--env",
            "LLM_API_KEY",
            "--env",
            "OPENHANDS_SUPPRESS_BANNER=1",
            "--env",
            f"HOME={container_home}",
            "--env",
            f"TMPDIR={container_tmp}",
            "--env",
            f"XDG_CACHE_HOME={container_cache}",
            "--env",
            f"XDG_STATE_HOME={container_state}",
            "--env",
            f"CONTEXTBENCH_WORKSPACE_ROOT={container_workspace}",
            "--env",
            "PYTHONNOUSERSITE=1",
            row["sandbox_image"],
            args.image_openhands_bin,
            *base_openhands_args(args, container_prompt),
        ]
    )

    env = build_docker_cli_env(args)
    metadata = {
        "runtime": "image",
        "argv": argv,
        "cwd": str(run_dir),
        "container_image": row["sandbox_image"],
        "container_workspace": container_workspace,
        "container_run_dir": container_run_dir,
        "image_run_network": args.image_run_network,
        "image_run_as_current_user": bool(args.image_run_as_current_user),
        "env_overrides": env_overrides_for_metadata(
            {
                **env,
                "HOME": container_home,
                "TMPDIR": container_tmp,
                "XDG_CACHE_HOME": container_cache,
                "XDG_STATE_HOME": container_state,
                "CONTEXTBENCH_WORKSPACE_ROOT": container_workspace,
                "PYTHONNOUSERSITE": "1",
            }
        ),
    }
    return argv, run_dir, env, metadata


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


def execute_one(
    row: dict[str, Any],
    args: argparse.Namespace,
    repo_root: Path,
    pulled_images: set[str],
    image_openhands_cache: dict[tuple[str, str], bool],
) -> dict[str, Any]:
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
        "openhands_runtime_requested": args.openhands_runtime,
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

        runtime, runtime_reason = choose_openhands_runtime(
            row=row,
            args=args,
            image_openhands_cache=image_openhands_cache,
        )
        if runtime == "image":
            openhands_argv, openhands_cwd, oh_env, command_meta = build_image_openhands_command(
                row=row,
                args=args,
                run_dir=run_dir,
                workspace_dir=workspace_dir,
            )
        else:
            openhands_argv, openhands_cwd, oh_env, command_meta = build_host_openhands_command(
                args=args,
                run_dir=run_dir,
                workspace_dir=workspace_dir,
                prompt_dst=prompt_dst,
            )
        command_meta["runtime_reason"] = runtime_reason
        write_json(command_path, command_meta)
        meta["openhands_runtime"] = runtime
        meta["openhands_runtime_reason"] = runtime_reason

        guard_started_ns = time.time_ns()
        print(f"[run] {row['run_id']} runtime={runtime} cwd={openhands_cwd}", flush=True)
        result = run_streaming(
            openhands_argv,
            cwd=openhands_cwd,
            env=oh_env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=args.timeout_seconds,
        )
        off_workspace_writes = scan_off_workspace_writes(
            run_dir=run_dir,
            workspace_dir=workspace_dir,
            since_ns=guard_started_ns,
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
        meta["workspace_guardrail"] = {
            "enabled": True,
            "fail_on_violation": bool(args.fail_on_off_workspace_writes),
            "scan_root": str(run_dir.parent),
            "off_workspace_write_count": len(off_workspace_writes),
            "off_workspace_writes": off_workspace_writes,
        }

        patch = collect_patch(workspace_dir, patch_path)
        pred_paths = write_prediction_files(row=row, args=args, run_dir=run_dir, patch=patch)
        meta.update(pred_paths)
        meta["patch_sha256"] = hashlib.sha256(patch.encode("utf-8", errors="replace")).hexdigest()
        meta["patch_chars"] = len(patch)
        meta["patch_empty"] = patch == ""

        if off_workspace_writes and args.fail_on_off_workspace_writes:
            meta["status"] = "workspace_guardrail_failed"
            meta["orchestrator_success"] = False
        elif result.exit_code == 0:
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
        "openhands_runtime": args.openhands_runtime,
        "image_openhands_bin": args.image_openhands_bin,
        "container_run_dir": args.container_run_dir,
        "image_run_network": args.image_run_network,
        "image_run_as_current_user": bool(args.image_run_as_current_user),
        "fail_on_off_workspace_writes": bool(args.fail_on_off_workspace_writes),
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
    parser.add_argument(
        "--openhands-runtime",
        choices=["auto", "host", "image"],
        default=os.environ.get("CONTEXTBENCH_OPENHANDS_RUNTIME", DEFAULT_OPENHANDS_RUNTIME),
        help=(
            "Where to run OpenHands. auto prefers sandbox_image if the image contains OpenHands, "
            "otherwise falls back to the host OpenHands binary."
        ),
    )
    parser.add_argument(
        "--image-openhands-bin",
        default=os.environ.get("CONTEXTBENCH_IMAGE_OPENHANDS_BIN", DEFAULT_IMAGE_OPENHANDS_BIN),
        help="OpenHands executable name/path to use inside sandbox_image when --openhands-runtime=image/auto.",
    )
    parser.add_argument(
        "--container-run-dir",
        default=os.environ.get("CONTEXTBENCH_CONTAINER_RUN_DIR", DEFAULT_CONTAINER_RUN_DIR),
        help="Container path where the run directory is mounted for image runtime.",
    )
    parser.add_argument(
        "--image-run-network",
        default=os.environ.get("CONTEXTBENCH_IMAGE_RUN_NETWORK", "host"),
        help="Docker network mode for image runtime. Default host keeps localhost LLM endpoints reachable on Linux.",
    )
    parser.add_argument(
        "--image-run-as-current-user",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run image-runtime containers with the current uid:gid to avoid root-owned workspace edits.",
    )
    parser.add_argument(
        "--fail-on-off-workspace-writes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mark a run failed if files are modified outside the current workspace boundary.",
    )
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
    parser.add_argument(
        "--always-approve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --always-approve to OpenHands. Default true for non-interactive benchmark runs.",
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
        args.host_openhands_available = False
        if args.openhands_runtime in {"auto", "host"}:
            found_openhands = shutil.which(args.openhands_bin)
            if found_openhands:
                args.openhands_bin = found_openhands
                args.host_openhands_available = True
            elif args.openhands_runtime == "host":
                raise OrchestratorError(f"Required executable not found on PATH: {args.openhands_bin}")
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
        image_openhands_cache: dict[tuple[str, str], bool] = {}
        results: list[dict[str, Any]] = []
        had_error = False

        for idx, row in enumerate(rows, start=1):
            print(f"\n=== [{idx}/{len(rows)}] {row['run_id']} ===", flush=True)
            try:
                result = execute_one(row, args, repo_root, pulled_images, image_openhands_cache)
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
