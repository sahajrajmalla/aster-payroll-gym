"""Versioned identities, independent of optional model libraries."""

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Literal

RULESET_VERSION: Final[Literal["ASTER-1.0"]] = "ASTER-1.0"
REWARD_VERSION: Final[Literal["1.0"]] = "1.0"
SCHEMA_VERSION: Final[Literal["1.0"]] = "1.0"
GENERATOR_VERSION: Final[Literal["1.0"]] = "1.0"


def stable_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()


def rules_text() -> str:
    packaged = Path(__file__).parent / "assets" / "aster-payroll-v1.md"
    source = Path(__file__).resolve().parents[2] / "rules" / "aster-payroll-v1.md"
    return (packaged if packaged.exists() else source).read_text(encoding="utf-8")


def rules_hash() -> str:
    return hashlib.sha256(rules_text().encode()).hexdigest()


def implementation_hash() -> str:
    root = Path(__file__).parent
    return stable_hash({str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*.py"))})
