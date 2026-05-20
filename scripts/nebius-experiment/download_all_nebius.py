# scripts/download_all_nebius.py
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

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

        # Nebius stores assistant tool-call arguments as strings for efficiency.
        # Deserialize them now so later conversion to ADP is easier.
        if role == "assistant" and kept.get("tool_calls") is not None:
            for tool_call in kept["tool_calls"]:
                fn = tool_call.get("function", {})
                if "arguments" in fn:
                    fn["arguments"] = maybe_json_loads(fn["arguments"])

        trajectory.append(kept)

    row = dict(row)
    row["trajectory"] = trajectory
    return row

def main():
    ds = load_dataset(
        "nebius/SWE-rebench-openhands-trajectories",
        split="train",
    )

    print(ds)
    print(f"Writing to {OUT}")

    with OUT.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, total=len(ds)):
            row = clean_row(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"Rows written: {sum(1 for _ in OUT.open())}")
    print(f"Output file: {OUT}")

if __name__ == "__main__":
    main()
