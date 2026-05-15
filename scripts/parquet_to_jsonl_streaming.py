# scripts/parquet_to_jsonl_streaming.py
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm
import zstandard as zstd


ROLE2FIELD_NAMES = {
    "system": ["role", "content"],
    "assistant": ["role", "content", "tool_calls"],
    "user": ["role", "content"],
    "tool": ["role", "content", "name", "tool_call_id"],
}


def find_default_parquet() -> Path:
    candidates = list(Path("data/nebius/raw_hf").rglob("*.parquet"))
    if not candidates:
        raise FileNotFoundError("No .parquet file found under data/nebius/raw_hf")
    if len(candidates) > 1:
        print("Found multiple parquet files:")
        for p in candidates:
            print(" ", p)
        print("Using first one:", candidates[0])
    return candidates[0]


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
            for field_name in ROLE2FIELD_NAMES.get(role, ["role", "content"])
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


class ZstdShardWriter:
    def __init__(self, out_dir: Path, rows_per_shard: int):
        self.out_dir = out_dir
        self.rows_per_shard = rows_per_shard
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.shard_idx = -1
        self.rows_in_shard = 0
        self.raw_fh = None
        self.zstd_writer = None
        self.cctx = zstd.ZstdCompressor(level=3)

    def _open_next_shard(self):
        self.close()

        self.shard_idx += 1
        self.rows_in_shard = 0

        path = self.out_dir / f"part-{self.shard_idx:05d}.jsonl.zst"
        self.raw_fh = path.open("wb")
        self.zstd_writer = self.cctx.stream_writer(self.raw_fh)

        print(f"\nWriting shard: {path}")

    def write_row(self, row):
        if self.zstd_writer is None or self.rows_in_shard >= self.rows_per_shard:
            self._open_next_shard()

        line = json.dumps(row, ensure_ascii=False) + "\n"
        self.zstd_writer.write(line.encode("utf-8"))
        self.rows_in_shard += 1

    def close(self):
        if self.zstd_writer is not None:
            self.zstd_writer.flush(zstd.FLUSH_FRAME)
            self.zstd_writer.close()
            self.zstd_writer = None

        if self.raw_fh is not None:
            self.raw_fh.close()
            self.raw_fh = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/nebius/jsonl_shards"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rows-per-shard", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    parquet_path = args.input or find_default_parquet()

    print("Input:", parquet_path)
    print("Output dir:", args.out_dir)
    print("Batch size:", args.batch_size)
    print("Rows per shard:", args.rows_per_shard)
    print("Limit:", args.limit)

    pf = pq.ParquetFile(parquet_path, memory_map=True)
    total_rows = pf.metadata.num_rows
    if args.limit is not None:
        total_rows = min(total_rows, args.limit)

    writer = ZstdShardWriter(args.out_dir, args.rows_per_shard)

    written = 0

    try:
        with tqdm(total=total_rows) as pbar:
            for batch in pf.iter_batches(batch_size=args.batch_size):
                rows = batch.to_pylist()

                for row in rows:
                    if args.limit is not None and written >= args.limit:
                        break

                    writer.write_row(clean_row(row))
                    written += 1
                    pbar.update(1)

                del rows
                del batch

                if args.limit is not None and written >= args.limit:
                    break

    finally:
        writer.close()

    print(f"\nDone. Wrote {written} rows into {args.out_dir}")


if __name__ == "__main__":
    main()
