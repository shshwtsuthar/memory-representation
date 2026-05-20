#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import zstandard as zstd
from tqdm import tqdm
from transformers import AutoTokenizer


def iter_jsonl(path: str) -> Iterable[str]:
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


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def compact_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def first_user_text_from_raw(raw_row: dict) -> str:
    for msg in raw_row.get("trajectory", []):
        if msg.get("role") == "user":
            return msg.get("content") or ""
    return ""


def adp_class_and_api_counts(adp_row: dict) -> tuple[Counter, Counter]:
    classes = Counter()
    apis = Counter()

    for item in adp_row.get("content", []):
        cls = item.get("class_")
        classes[cls] += 1
        if cls == "api_action":
            apis[item.get("function", "unknown")] += 1

    return classes, apis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-glob",
        default="data/nebius/jsonl_shards/*.jsonl.zst",
    )
    parser.add_argument(
        "--adp-glob",
        default="data/nebius/adp_shards/*.adp.jsonl.zst",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        help="Tokenizer model name. Use the same tokenizer as the evaluation model.",
    )
    parser.add_argument(
        "--out",
        default="data/nebius/successful_small_trajectories.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    raw_paths = sorted(glob.glob(args.raw_glob))
    adp_paths = sorted(glob.glob(args.adp_glob))

    if not raw_paths:
        raise FileNotFoundError(f"No raw shards matched: {args.raw_glob}")
    if not adp_paths:
        raise FileNotFoundError(f"No ADP shards matched: {args.adp_glob}")
    if len(raw_paths) != len(adp_paths):
        raise ValueError(f"Shard count mismatch: raw={len(raw_paths)} adp={len(adp_paths)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rows_out = []
    total_seen = 0
    total_success = 0

    for raw_path, adp_path in zip(raw_paths, adp_paths):
        raw_iter = iter_jsonl(raw_path)
        adp_iter = iter_jsonl(adp_path)

        for raw_line, adp_line in tqdm(zip(raw_iter, adp_iter), desc=Path(raw_path).name):
            total_seen += 1

            raw_row = json.loads(raw_line)
            adp_row = json.loads(adp_line)

            raw_id = str(raw_row.get("trajectory_id"))
            adp_id = str(adp_row.get("id"))

            if raw_id != adp_id:
                raise ValueError(f"ID mismatch: raw={raw_id} adp={adp_id}")

            resolved = raw_row.get("resolved")
            if resolved != 1:
                if args.limit and total_seen >= args.limit:
                    break
                continue

            total_success += 1

            raw_min = compact_json(raw_row)
            adp_min = compact_json(adp_row)
            issue_text = first_user_text_from_raw(raw_row)
            model_patch = raw_row.get("model_patch") or ""

            classes, apis = adp_class_and_api_counts(adp_row)

            raw_tokens = count_tokens(tokenizer, raw_min)
            adp_tokens = count_tokens(tokenizer, adp_min)
            issue_tokens = count_tokens(tokenizer, issue_text)
            patch_tokens = count_tokens(tokenizer, model_patch)

            rows_out.append(
                {
                    "trajectory_id": raw_id,
                    "instance_id": raw_row.get("instance_id", ""),
                    "repo": raw_row.get("repo", ""),
                    "resolved": resolved,
                    "exit_status": raw_row.get("exit_status", ""),
                    "raw_tokens": raw_tokens,
                    "adp_tokens": adp_tokens,
                    "issue_tokens": issue_tokens,
                    "patch_tokens": patch_tokens,
                    "raw_message_count": len(raw_row.get("trajectory", [])),
                    "adp_content_count": len(adp_row.get("content", [])),
                    "api_action_count": classes.get("api_action", 0),
                    "text_observation_count": classes.get("text_observation", 0),
                    "message_action_count": classes.get("message_action", 0),
                    "execute_bash_count": apis.get("execute_bash", 0),
                    "str_replace_editor_count": apis.get("str_replace_editor", 0),
                    "think_count": apis.get("think", 0),
                    "finish_count": apis.get("finish", 0),
                    "task_tracker_count": apis.get("task_tracker", 0),
                }
            )

            if args.limit and total_seen >= args.limit:
                break

        if args.limit and total_seen >= args.limit:
            break

    rows_out.sort(key=lambda r: (r["adp_tokens"], r["raw_tokens"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "trajectory_id",
        "instance_id",
        "repo",
        "resolved",
        "exit_status",
        "raw_tokens",
        "adp_tokens",
        "issue_tokens",
        "patch_tokens",
        "raw_message_count",
        "adp_content_count",
        "api_action_count",
        "text_observation_count",
        "message_action_count",
        "execute_bash_count",
        "str_replace_editor_count",
        "think_count",
        "finish_count",
        "task_tracker_count",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print("total_seen:", total_seen)
    print("total_success:", total_success)
    print("written_success_rows:", len(rows_out))
    print("output:", out_path)

    under_10k_adp = sum(1 for r in rows_out if r["adp_tokens"] <= 10_000)
    under_20k_adp = sum(1 for r in rows_out if r["adp_tokens"] <= 20_000)
    under_10k_raw = sum(1 for r in rows_out if r["raw_tokens"] <= 10_000)
    under_20k_raw = sum(1 for r in rows_out if r["raw_tokens"] <= 20_000)

    print("successful ADP <= 10k:", under_10k_adp)
    print("successful ADP <= 20k:", under_20k_adp)
    print("successful raw <= 10k:", under_10k_raw)
    print("successful raw <= 20k:", under_20k_raw)

    print("\nTop 20 smallest successful trajectories by ADP tokens:")
    for r in rows_out[:20]:
        print(
            r["trajectory_id"],
            "repo=", r["repo"],
            "adp_tokens=", r["adp_tokens"],
            "raw_tokens=", r["raw_tokens"],
            "messages=", r["raw_message_count"],
            "patch_tokens=", r["patch_tokens"],
        )


if __name__ == "__main__":
    main()
