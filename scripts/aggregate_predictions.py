#!/usr/bin/env python3
"""
Walk an execution root and produce per-condition prediction JSONL files
from the per-run prediction.json artifacts. Idempotent.

Usage:
  python3 scripts/aggregate_predictions.py <execution_root>
"""

import json
import sys
from pathlib import Path

CONDITIONS = ["no_memory", "raw", "adp", "memory"]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: aggregate_predictions.py <execution_root>", file=sys.stderr)
        return 2

    root = Path(argv[1])
    runs_dir = root / "runs"
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} does not exist", file=sys.stderr)
        return 2

    out_dir = root / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for cond in CONDITIONS:
        preds: list[dict] = []
        for target_dir in sorted(runs_dir.iterdir()):
            if not target_dir.is_dir():
                continue
            pred_path = target_dir / cond / "prediction.json"
            if not pred_path.exists():
                continue
            try:
                pred = json.loads(pred_path.read_text(encoding="utf-8"))
                preds.append(pred)
            except Exception as e:
                print(f"WARN: failed to read {pred_path}: {e}", file=sys.stderr)

        out_path = out_dir / f"full_{cond}_predictions.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, sort_keys=True, ensure_ascii=False) + "\n")

        non_empty = sum(1 for p in preds if (p.get("model_patch") or "").strip())
        summary[cond] = {"count": len(preds), "non_empty": non_empty, "path": str(out_path)}

    print(json.dumps({"aggregate_summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
