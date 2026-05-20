import glob
import json
from pathlib import Path

import zstandard as zstd


def iter_zst_jsonl(paths):
    for path in paths:
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


def main():
    paths = sorted(glob.glob("data/nebius/jsonl_shards/*.jsonl.zst"))
    if not paths:
        raise FileNotFoundError("No shards found at data/nebius/jsonl_shards/*.jsonl.zst")

    success = []
    failure = []

    for line in iter_zst_jsonl(paths):
        row = json.loads(line)

        if row.get("resolved") == 1 and len(success) < 50:
            success.append(row)
        elif row.get("resolved") == 0 and len(failure) < 50:
            failure.append(row)

        if len(success) == 50 and len(failure) == 50:
            break

    out = Path("data/nebius/dev/raw_100_mixed.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for row in success + failure:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("wrote", len(success), "success and", len(failure), "failure")
    print("output:", out)


if __name__ == "__main__":
    main()
