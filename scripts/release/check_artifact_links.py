#!/usr/bin/env python3
"""Check that public Hugging Face artifact URLs are reachable."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request


ARTIFACT_URLS = {
    "shshwtsuthar/memory-representation-contextbench-artifacts": "https://huggingface.co/datasets/shshwtsuthar/memory-representation-contextbench-artifacts",
    "shshwtsuthar/memory-representation-contextbench-traces": "https://huggingface.co/datasets/shshwtsuthar/memory-representation-contextbench-traces",
    "shshwtsuthar/memory-representation-nebius-openhands-adp-v0.1": "https://huggingface.co/datasets/shshwtsuthar/memory-representation-nebius-openhands-adp-v0.1",
}


def request_url(url: str, method: str) -> int:
    request = urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": "memory-representation-release-check/0.1"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.status


def main() -> int:
    failed = False
    for repo_id, url in ARTIFACT_URLS.items():
        try:
            status = request_url(url, "HEAD")
        except urllib.error.HTTPError as exc:
            if exc.code in {405, 501}:
                try:
                    status = request_url(url, "GET")
                except Exception as get_exc:  # noqa: BLE001
                    print(f"FAIL {repo_id}: {get_exc}", file=sys.stderr)
                    failed = True
                    continue
            else:
                print(f"FAIL {repo_id}: HTTP {exc.code}", file=sys.stderr)
                failed = True
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {repo_id}: {exc}", file=sys.stderr)
            failed = True
            continue

        if 200 <= status < 400:
            print(f"OK   {repo_id}: HTTP {status}")
        else:
            print(f"FAIL {repo_id}: HTTP {status}", file=sys.stderr)
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
