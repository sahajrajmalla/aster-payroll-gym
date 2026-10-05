"""Independent client acceptance fixtures; no real hosted-deployment claims."""

import ast
import copy
import importlib.util
import json
import os
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aster_gym.api import create_app

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sandbox_smoke.py"
SPEC = importlib.util.spec_from_file_location("sandbox_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def public_example():
    return {
        "id": "public-fixture", "ruleset_version": "ASTER-1.0", "schema_version": "1.0",
        "context_files": {
            "rules.md": "ASTER-1.0 synthetic fixture",
            "request.json": json.dumps({"kind": "payslip", "pay_date": "2030-01-31"}),
            "payroll-records.json": json.dumps({
                "salary_records": [{"effective": "2030-01-01", "signed": True,
                                    "monthly_salary_cents": 300000}],
                "period_days": 31, "paid_days": 31, "bonus_cents": 0,
                "ytd_year": 2030, "ytd_pensionable_cents": 0,
            }),
            "schedules.json": json.dumps([{
                "start": "2030-01-01", "end": "2030-12-31", "contribution_rate": "0.05",
                "annual_ceiling_cents": 1200000, "allowance_cents": 10000,
                "bands": [[100000, "0"], [300000, "0.10"], [None, "0.20"]],
            }]),
        },
    }


def arguments(tmp_path, **overrides):
    args = dict(url="http://127.0.0.1:8000", tier=2, receipt=tmp_path / "private.json",
                proof=tmp_path / "public.json", verify_restart=False, restart_confirmed=False,
                resume=False, timeout=1, attempts=1, backoff=0)
    return SimpleNamespace(**{**args, **overrides})


def test_independent_hand_calculation_and_contribution_boundary():
    task = public_example()
    assert smoke.independent_answer(task)["result"] == {
        "gross_cents": 300000, "contribution_cents": 15000, "taxable_cents": 275000,
        "tax_cents": 17500, "net_cents": 267500,
    }
    payroll = json.loads(task["context_files"]["payroll-records.json"])
    payroll["ytd_pensionable_cents"] = 1199990
    task["context_files"]["payroll-records.json"] = json.dumps(payroll)
    # 10 eligible cents * 5% = half a cent, rounded HALF_UP to one cent.
    assert smoke.independent_answer(task)["result"] == {
        "gross_cents": 300000, "contribution_cents": 1, "taxable_cents": 289999,
        "tax_cents": 19000, "net_cents": 280999,
    }


@pytest.mark.parametrize("trap,code,field,clause", [
    ("missing", "MISSING_INPUT", "ytd_pensionable_cents", "R4"),
    ("conflict", "SALARY_CONFLICT", "salary_records", "R2"),
    ("schedule", "NO_SCHEDULE", "schedules", "R8"),
])
def test_traps_are_diagnosed_from_public_evidence(trap, code, field, clause):
    task = public_example()
    payroll = json.loads(task["context_files"]["payroll-records.json"])
    if trap == "missing":
        del payroll["ytd_pensionable_cents"]
    elif trap == "conflict":
        payroll["salary_records"].append({"effective": "2030-01-01", "signed": True,
                                          "monthly_salary_cents": 400000})
    else:
        task["context_files"]["request.json"] = '{"kind":"payslip","pay_date":"2031-01-31"}'
        payroll["ytd_year"] = 2031
    task["context_files"]["payroll-records.json"] = json.dumps(payroll)
    answer = smoke.independent_answer(task)
    assert answer["decision"] == "needs_information" and answer["result"] is None
    assert answer["issue_codes"] == [code] and answer["missing_fields"] == [field]
    assert set(answer["citations"]) == {clause, "R9", "R10"}


def test_private_receipt_and_restart_use_same_public_run(tmp_path, monkeypatch):
    database = f"sqlite:///{tmp_path / 'api.sqlite'}"
    app = create_app(database, judge=None)
    client = TestClient(app)

    def via_public_api(base, path, payload=None, token="", **kwargs):
        result = client.request("POST" if payload is not None else "GET", path,
                                json=payload, headers={"X-Run-Token": token})
        return result.status_code, result.json()

    monkeypatch.setattr(smoke, "_request", via_public_api)
    args = arguments(tmp_path)
    with client:
        proof = smoke.run_smoke(args)
        original_id = proof["run_id"]
        args.resume = True
        resumed = smoke.run_smoke(args)
        assert resumed["run_id"] == original_id
    assert proof["status"] == "submission_verified" and proof["score"] == 1
    assert proof["deployment_scope"] == "local_only"
    assert os.stat(args.receipt).st_mode & 0o777 == 0o600
    receipt = json.loads(args.receipt.read_text())
    assert "run_token" not in args.proof.read_text()
    assert receipt["issued"]["run_token"] not in args.proof.read_text()
    assert "ground_truth" not in args.proof.read_text()
    assert receipt["issued"]["run_id"] == proof["run_id"]

    # A genuinely new application/store instance reads the persisted SQLite run.
    client = TestClient(create_app(database, judge=None))
    args = arguments(tmp_path, url=None, verify_restart=True, restart_confirmed=True)
    with client:
        restarted = smoke.run_smoke(args)
    assert restarted["status"] == "restart_persistence_verified"
    assert restarted["post_restart_run_hash"] == proof["completed_run_hash"]


def test_pending_judge_never_counts_as_success(tmp_path, monkeypatch):
    client = TestClient(create_app(f"sqlite:///{tmp_path / 'pending.sqlite'}", judge=None))

    def via_public_api(base, path, payload=None, token="", **kwargs):
        result = client.request("POST" if payload is not None else "GET", path,
                                json=payload, headers={"X-Run-Token": token})
        return result.status_code, result.json()

    monkeypatch.setattr(smoke, "_request", via_public_api)
    args = arguments(tmp_path, tier=3)
    with client:
        proof = smoke.run_smoke(args)
    assert proof["status"] == "pending_judge_or_scoring" and proof["score"] is None
    assert json.loads(args.receipt.read_text())["completed_run"] is None


def test_restart_requires_confirmation_and_unchanged_response(tmp_path, monkeypatch):
    args = arguments(tmp_path, verify_restart=True)
    with pytest.raises(smoke.SmokeError, match="Restart the service first"):
        smoke.run_smoke(args)
    task = public_example()
    issued = {"run_id": "r", "run_token": "private-token", "tasks": [task]}
    payload = {"run_id": "r", "answers": [{"task_id": task["id"], "answer": smoke.independent_answer(task)}]}
    original = {"run_id": "r", "status": "complete", "scores": [{"score": 1, "status": "complete"}],
                "transcript": {"tasks": [task], "submitted_answers": payload["answers"],
                               "scores": [{"score": 1, "status": "complete"}]}}
    smoke.write_json(args.receipt, {"base_url": args.url, "issued": issued, "payload": payload,
                                   "tier": 2, "completed_run": original, "proof": {}}, private=True)
    changed = copy.deepcopy(original)
    changed["scores"][0]["score"] = .2
    monkeypatch.setattr(smoke, "_request", lambda *a, **k: (200, changed))
    args.restart_confirmed = True
    with pytest.raises(smoke.SmokeError, match="passing score"):
        smoke.run_smoke(args)


@pytest.mark.parametrize("url", [
    "https://user:secret@example.org", "https://example.org?key=secret",
    "https://example.org/path-containing-secret", "http://example.org", "https://example.org#secret",
])
def test_url_credentials_and_unsafe_public_urls_rejected(url):
    with pytest.raises(smoke.SmokeError):
        smoke.checked_base(url)


def test_http_retries_are_bounded_and_do_not_print_private_body(monkeypatch):
    class Outage:
        attempts = 0

        def open(self, request, timeout):
            self.attempts += 1
            raise urllib.error.HTTPError(request.full_url, 503, "private database detail", {}, None)

    outage = Outage()
    monkeypatch.setattr(smoke.urllib.request, "build_opener", lambda *a: outage)
    monkeypatch.setattr(smoke.time, "sleep", lambda *a: None)
    with pytest.raises(smoke.SmokeError, match="Sandbox HTTP 503") as failure:
        smoke._request("https://example.org", "/healthz", attempts=3, backoff=0)
    assert outage.attempts == 3
    assert "database detail" not in str(failure.value)


def test_token_redirect_and_symlink_receipt_are_refused(tmp_path):
    with pytest.raises(smoke.SmokeError, match="redirects are refused"):
        smoke.NoRedirect().redirect_request(None, None, 302, "", {}, "https://attacker.example")
    target = tmp_path / "target"
    target.write_text("keep")
    receipt = tmp_path / "receipt"
    receipt.symlink_to(target)
    with pytest.raises(smoke.SmokeError, match="symlinks"):
        smoke.write_json(receipt, {"token": "private"}, private=True)
    assert target.read_text() == "keep"


def test_smoke_has_no_project_or_model_imports():
    tree = ast.parse(SCRIPT.read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name.split(".")[0] in {"aster_gym", "torch", "transformers", "trl", "peft"}
                   for name in imports)
