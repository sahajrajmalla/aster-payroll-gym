"""Failure-tolerant Colab orchestration; never imports the model stack."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .guard import require_colab


def code_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def record_skip(*, label: str, reason_code: str, status_path: str | Path, explicit: bool) -> None:
    """Preserve a deliberate skipped phase without running a command or inventing scores."""
    require_colab(explicit=explicit)
    path = Path(status_path)
    report = json.loads(path.read_text()) if path.exists() else {"steps": []}
    report["steps"].append({"label": label, "status": "skipped", "reason_code": reason_code,
                            "code_revision": code_revision(), "skipped_at": datetime.now(UTC).isoformat()})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(f"{label}: skipped ({reason_code}); experimental evidence remains pending")


def run_step(command: list[str], *, label: str, status_path: str | Path, explicit: bool) -> bool:
    """Record command outcomes without turning a failed experiment into a result.

    A failed training command must not block unrelated remote evaluation or export.
    Commands carry no credentials; secrets are supplied only through the environment.
    GPU entrypoints repeat the stricter CUDA check themselves.
    """
    require_colab(explicit=explicit)
    path = Path(status_path)
    report = json.loads(path.read_text()) if path.exists() else {"steps": []}
    entry: dict[str, object] = {"label": label, "started_at": datetime.now(UTC).isoformat(),
                               "code_revision": code_revision()}
    try:
        result = subprocess.run(command, check=False)
        entry.update(exit_code=result.returncode,
                     status="command_finished" if result.returncode == 0 else "command_failed")
    except OSError as error:
        entry.update(exit_code=None, status="command_failed", error_type=type(error).__name__)
    entry["finished_at"] = datetime.now(UTC).isoformat()
    report["steps"].append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    succeeded = entry["exit_code"] == 0
    print(f"{label}: {'command finished; inspect saved evidence' if succeeded else 'failed; evidence preserved'}")
    return succeeded
