# scripts/parquet_to_jsonl.py
import json
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

IN = Path("data/nebius/raw_hf/trajectories.parquet")
OUT = Path("data/nebius/nebius_openhands_all.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

ROLE2FIELD_NAMES = {
    "system": ["role", "content"],
    "assistant": ["role", "content", "tool_calls"],
    "user": ["role", "content"],
    "tool": ["role", "content", "name", "tool_call_id"],
}

def maybe_json_loads(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value

def clean_row(row):
    trajectory = []

    for msg in row["trajectory"]:
        role = msg["role"]
        kept = {
            field_name: msg[field_name]
            for field_name in ROLE2FIELD_NAMES[role]
            if field_name in msg
        }

        if role == "assistant" and kept.get("tool_calls") is not None:
            for tool_call in kept["tool_calls"]:
                fn = tool_call.get("function", {})
                if "arguments" in fn:
                    fn["arguments"] = maybe_json_loads(fn["arguments"])

        trajectory.append(kept)

    row["trajectory"] = trajectory
    return row

table = pq.read_table(IN)
rows = table.to_pylist()

with OUT.open("w", encoding="utf-8") as f:
    for row in tqdm(rows):
        f.write(json.dumps(clean_row(row), ensure_ascii=False) + "\n")

print(f"Wrote {len(rows)} rows to {OUT}")
