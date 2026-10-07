"""Guarded cloud diagnostics; no model imports, weights or inference."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .guard import CloudOnlyError, require_colab, require_cuda
from .workflow import code_revision

EXPECTED_VERSIONS = {"torch": "2.8.0", "transformers": "4.57.1", "trl": "0.26.2",
                     "verifiers": "0.1.14", "peft": "0.17.1", "accelerate": "1.10.1"}


def _check_environments() -> None:
    import verifiers as vf

    from aster_gym.environment import load_environment

    for mode in ("single", "tool"):
        if not isinstance(load_environment(mode=mode, n=3, judge=None), vf.Environment):
            raise RuntimeError("Environment adapter type mismatch")


def run_preflight(output: str | Path, *, explicit: bool) -> bool:
    """Save bounded, secret-free checks and preserve previous diagnostic attempts."""
    require_colab(explicit=explicit)  # Local invocation fails before writing or importing ML.
    row: dict[str, Any] = {"started_at": datetime.now(UTC).isoformat(), "code_revision": code_revision(),
                           "status": "running", "checks": {"hosted_colab": True}, "versions": {}}
    stage = "CUDA_CHECK_FAILED"
    try:
        torch = require_cuda(explicit=explicit)
        row["checks"]["cuda"] = True
        row["cuda_device"] = torch.cuda.get_device_name(0)
        stage = "DEPENDENCY_MISMATCH"
        for package, expected in EXPECTED_VERSIONS.items():
            observed = version(package).split("+")[0]
            row["versions"][package] = observed
            if observed != expected:
                raise RuntimeError("Pinned dependency mismatch")
        row["checks"]["dependencies"] = True
        stage = "ENVIRONMENT_STARTUP_FAILED"
        _check_environments()
        row["checks"]["environments"] = True
        row["status"] = "passed"
    except Exception as error:
        if stage == "CUDA_CHECK_FAILED" and isinstance(error, CloudOnlyError):
            stage = "CUDA_REQUIRED"
        row.update(status="failed", failure_code=stage, error_type=type(error).__name__)
        # Never publish exception text, environment values, credentials or tracebacks.
    row["finished_at"] = datetime.now(UTC).isoformat()
    path = Path(output)
    prior = json.loads(path.read_text()) if path.exists() else {"attempts": []}
    prior["attempts"].append(row)
    prior["status"] = row["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(prior, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(json.dumps(row, indent=2, allow_nan=False))
    return row["status"] == "passed"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Colab-only GPU and dependency preflight; no models loaded")
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-preflight", action="store_true")
    args = parser.parse_args(argv)
    if not run_preflight(args.output, explicit=args.start_preflight):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
