"""Cross-check the rule review cohort without importing the production oracle."""

import json
import runpy
from pathlib import Path

import pytest

_script = runpy.run_path("scripts/audit_rule_fidelity.py")
audit, independent_result = _script["audit"], _script["independent_result"]


def test_independent_cohort_audit_detects_changed_oracle_amount(tmp_path):
    packet = json.loads(Path("reviews/task-review-packet.json").read_text())
    source = tmp_path / "packet.json"
    source.write_text(json.dumps(packet))
    assert audit(source, tmp_path / "audit.json")["disagreements"] == 0
    packet["seed_reviews"][4]["reference_result"]["result"]["gross_cents"] += 1
    source.write_text(json.dumps(packet))
    assert audit(source, tmp_path / "audit.json")["disagreements"] == 1


def test_independent_audit_does_not_accept_task_rates_as_authority():
    row = json.loads(Path("reviews/task-review-packet.json").read_text())["seed_reviews"][0]
    row["inputs"]["schedules"][0]["allowance_cents"] += 1
    with pytest.raises(ValueError, match="R8"):
        independent_result(row["inputs"])
