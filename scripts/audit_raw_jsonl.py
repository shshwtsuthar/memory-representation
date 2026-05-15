import argparse
import json
from collections import Counter
from pathlib import Path

import zstandard as zstd
from tqdm import tqdm


def open_text(path: Path):
    if path.suffix == ".zst":
        fh = path.open("rb")
        dctx = zstd.ZstdDecompressor()
        stream = dctx.stream_reader(fh)
        return fh, open(stream.fileno(), "r", encoding="utf-8", errors="replace")
    return None, path.open("r", encoding="utf-8", errors="replace")


def iter_lines(paths):
    for path in paths:
        if str(path).endswith(".zst"):
            with path.open("rb") as fh:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(fh) as reader:
                    text = ""
                    while True:
                        chunk = reader.read(1024 * 1024)
                        if not chunk:
                            break
                        text += chunk.decode("utf-8", errors="replace")
                        while "\n" in text:
                            line, text = text.split("\n", 1)
                            if line.strip():
                                yield line
                    if text.strip():
                        yield text
        else:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                yield from f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    paths = []
    for pattern in args.paths:
        matches = sorted(Path().glob(pattern))
        paths.extend(matches if matches else [Path(pattern)])

    rows = 0
    resolved = Counter()
    exit_status = Counter()
    role_counts = Counter()
    message_lengths = []
    repos = Counter()

    for line in tqdm(iter_lines(paths)):
        row = json.loads(line)
        rows += 1

        resolved[row.get("resolved")] += 1
        exit_status[row.get("exit_status")] += 1
        repos[row.get("repo")] += 1

        traj = row.get("trajectory") or []
        message_lengths.append(len(traj))

        for msg in traj:
            role_counts[msg.get("role")] += 1

    print("rows:", rows)
    print("resolved:", resolved)
    print("exit_status top:", exit_status.most_common(10))
    print("roles:", role_counts)
    print("repos:", len(repos))
    print("avg messages:", sum(message_lengths) / max(len(message_lengths), 1))
    print("max messages:", max(message_lengths) if message_lengths else 0)


if __name__ == "__main__":
    main()
