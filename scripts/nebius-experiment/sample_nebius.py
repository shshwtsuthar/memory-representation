# scripts/sample_nebius.py
import json
from datasets import load_dataset

N = 20

ds = load_dataset(
    "nebius/SWE-rebench-openhands-trajectories",
    split="train",
    streaming=True,
)

role2field_names = {
    "system": ["role", "content"],
    "assistant": ["role", "content", "tool_calls"],
    "user": ["role", "content"],
    "tool": ["role", "content", "name", "tool_call_id"],
}

def filter_and_deserialize(row):
    trajectory = []

    for msg in row["trajectory"]:
        msg = {
            field_name: msg[field_name]
            for field_name in role2field_names[msg["role"]]
        }

        if msg["role"] == "assistant" and msg.get("tool_calls") is not None:
            for i, tool_call in enumerate(msg["tool_calls"]):
                fn = tool_call.get("function", {})
                if "arguments" in fn and isinstance(fn["arguments"], str):
                    try:
                        msg["tool_calls"][i]["function"]["arguments"] = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        pass

        trajectory.append(msg)

    row = dict(row)
    row["trajectory"] = trajectory
    return row

with open("sample_nebius_openhands.jsonl", "w") as f:
    for i, row in enumerate(ds):
        if i >= N:
            break
        row = filter_and_deserialize(row)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote {N} rows to sample_nebius_openhands.jsonl")
