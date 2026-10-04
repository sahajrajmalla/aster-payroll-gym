#!/usr/bin/env python3
"""No-clone sandbox quickstart: Python standard library only; no model calls."""

import argparse
import json
import urllib.error
import urllib.request


def request(url: str, payload: dict | None = None, token: str = "") -> dict:
    headers = {"Content-Type": "application/json", "X-Run-Token": token}
    body = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=120) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Public sandbox base URL")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    try:
        issued = request(base + "/tasks?tier=2&n=1")
        # Deliberately malformed baseline proves independent scoring; no oracle needed.
        result = request(base + "/submit", {
            "run_id": issued["run_id"],
            "answers": [{"task_id": task["id"], "answer": "{}"} for task in issued["tasks"]],
        }, issued["run_token"])
        print(json.dumps(result, indent=2))
    except urllib.error.HTTPError as exc:
        print(f"Sandbox returned HTTP {exc.code}: {exc.read().decode()}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
