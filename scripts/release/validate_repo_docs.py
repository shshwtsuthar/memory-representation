#!/usr/bin/env python3
"""Validate release documentation invariants."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = [
    "README.md",
    "CITATION.cff",
    "docs/artifacts.md",
    "docs/pipeline.md",
    "docs/reproducibility.md",
    "docs/releases/v0.1-workshop-submission.md",
]

REQUIRED_README_TEXT = [
    "shshwtsuthar/memory-representation-contextbench-artifacts",
    "shshwtsuthar/memory-representation-contextbench-traces",
    "shshwtsuthar/memory-representation-nebius-openhands-adp-v0.1",
    "95 valid targets",
    "380 evaluated runs",
    "complementarity",
]

FORBIDDEN_PHRASES = [
    "raw trajectories are universally best",
    "our method solves 28/95",
    "proves causal",
]

MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"FAIL: {message}", file=sys.stderr)


def is_external_link(target: str) -> bool:
    return (
        "://" in target
        or target.startswith("#")
        or target.startswith("mailto:")
    )


def strip_link_target(target: str) -> str:
    cleaned = target.split("#", 1)[0]
    cleaned = cleaned.split("?", 1)[0]
    if cleaned.startswith("<") and cleaned.endswith(">"):
        cleaned = cleaned[1:-1]
    return cleaned


def validate_local_links(markdown_file: Path, failures: list[str]) -> None:
    text = markdown_file.read_text(encoding="utf-8")
    for match in MARKDOWN_LINK_RE.finditer(text):
        target = strip_link_target(match.group(1).strip())
        if not target or is_external_link(target):
            continue
        if target.startswith("/"):
            fail(f"{markdown_file.relative_to(ROOT)} links to absolute local path {target}", failures)
            continue
        resolved = (markdown_file.parent / target).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError:
            fail(f"{markdown_file.relative_to(ROOT)} links outside repository: {target}", failures)
            continue
        if not resolved.exists():
            fail(f"{markdown_file.relative_to(ROOT)} has missing local link: {target}", failures)


def main() -> int:
    failures: list[str] = []

    for rel_path in REQUIRED_FILES:
        if not (ROOT / rel_path).exists():
            fail(f"missing required file: {rel_path}", failures)

    readme_path = ROOT / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        readme_lower = readme.lower()
        for required in REQUIRED_README_TEXT:
            if required not in readme:
                fail(f"README.md missing required text: {required}", failures)
        for forbidden in FORBIDDEN_PHRASES:
            if forbidden in readme_lower:
                fail(f"README.md contains forbidden phrase: {forbidden}", failures)

    for markdown_file in [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]:
        if markdown_file.exists():
            validate_local_links(markdown_file, failures)

    if failures:
        print(f"{len(failures)} validation failure(s)", file=sys.stderr)
        return 1

    print("release documentation validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
