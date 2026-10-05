#!/usr/bin/env python3
"""Independent public sandbox acceptance check; standard library, no model or oracle imports.

Fetch one public payslip, calculate from its public documents, submit, and retrieve.
Keep the receipt private. After restarting the service, verify the same saved run:
  python scripts/sandbox_smoke.py --verify-restart --restart-confirmed
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

MAX_RESPONSE_BYTES = 256 * 1024
RULESET = "ASTER-1.0"
SCHEMA = "1.0"
RETRYABLE_HTTP = {429, 502, 503, 504}


class SmokeError(ValueError):
    """A safe error that excludes server internals, tokens and connection secrets."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request: Any, fp: Any, code: int, message: str,
                         headers: Any, new_url: str) -> None:
        raise SmokeError("Sandbox redirects are refused to protect the private run token")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def checked_base(value: str) -> tuple[str, bool]:
    parsed = urllib.parse.urlsplit(value)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if (parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment
            or not parsed.hostname or parsed.path not in {"", "/"}
            or parsed.scheme not in ({"http", "https"} if local else {"https"})):
        raise SmokeError("Use a root HTTPS sandbox URL without credentials, query or fragment; HTTP is local only")
    try:
        parsed.port
    except ValueError as error:
        raise SmokeError("Invalid sandbox port") from error
    return value.rstrip("/"), local


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SmokeError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise SmokeError("Non-finite JSON value")


def decode(data: bytes | str) -> Any:
    try:
        return json.loads(data, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise SmokeError("Invalid JSON document") from error


def _request(base: str, path: str, *, payload: dict | None = None, token: str = "",
             timeout: float = 60, attempts: int = 3, backoff: float = 2) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json", "X-Run-Token": token}
    body = canonical(payload) if payload is not None else None
    for attempt in range(attempts):
        retry_delay = backoff
        try:
            request = urllib.request.Request(base + path, body, headers)
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
                data = response.read(MAX_RESPONSE_BYTES + 1)
                if len(data) > MAX_RESPONSE_BYTES:
                    raise SmokeError("Sandbox response exceeds acceptance-check limit")
                decoded = decode(data)
                if not isinstance(decoded, dict):
                    raise SmokeError("Sandbox response must be an object")
                return response.status, decoded
        except urllib.error.HTTPError as error:
            error.close()
            if error.code not in RETRYABLE_HTTP or attempt + 1 == attempts:
                raise SmokeError(f"Sandbox HTTP {error.code}; no private error body retained") from error
            try:
                retry_delay = max(backoff, min(10.0, float(error.headers.get("Retry-After", "0"))))
            except ValueError:
                pass
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if attempt + 1 == attempts:
                raise SmokeError("Sandbox connection failed after bounded retries") from error
        time.sleep(retry_delay)
    raise SmokeError("Sandbox retries exhausted")


def _cents(value: Any, field: str) -> int:
    if type(value) is not int or not 0 <= value <= 100_000_000:
        raise SmokeError(f"Unsupported public monetary field: {field}")
    return value


def independent_answer(task: dict) -> dict:
    """Apply R1–R10 to issued public evidence; never read internal labels or expected answers."""
    if task.get("ruleset_version") != RULESET or task.get("schema_version") != SCHEMA:
        raise SmokeError("Unsupported public task version")
    docs = task.get("context_files")
    if not isinstance(docs, dict) or not isinstance(docs.get("rules.md"), str):
        raise SmokeError("Public rulebook is missing")
    if "ASTER-1.0" not in docs["rules.md"]:
        raise SmokeError("Public rulebook version mismatch")
    request = decode(docs.get("request.json", "{}"))
    if not isinstance(request, dict) or request.get("kind") != "payslip":
        raise SmokeError("Acceptance solver supports issued monthly payslips only")
    try:
        pay_date = date.fromisoformat(request["pay_date"])
    except (ValueError, TypeError, KeyError) as error:
        raise SmokeError("Unsupported public payment date") from error
    payroll: dict[str, Any] = {}
    for name, content in docs.items():
        if name == "payroll-records.json" or name.startswith("record-"):
            fragment = decode(content)
            if not isinstance(fragment, dict) or set(payroll).intersection(fragment):
                raise SmokeError("Conflicting public payroll fragments")
            payroll.update(fragment)
    schedules = decode(docs.get("schedules.json", "[]"))
    if not isinstance(schedules, list):
        raise SmokeError("Unsupported public schedules")
    active = [s for s in schedules if isinstance(s, dict)
              and s.get("start", "9999") <= pay_date.isoformat() <= s.get("end", "0000")]
    issues: dict[str, str] = {}
    clauses = {"R9", "R10"}
    if not active:
        issues["NO_SCHEDULE"] = "schedules"
        clauses.add("R8")
    elif len(active) != 1:
        raise SmokeError("Unsupported public schedule conflict")
    salaries = payroll.get("salary_records")
    if not isinstance(salaries, list):
        raise SmokeError("Unsupported public salary evidence")
    applicable = [s for s in salaries if isinstance(s, dict) and s.get("signed") is True
                  and isinstance(s.get("effective"), str) and s["effective"] <= pay_date.isoformat()]
    if not applicable:
        raise SmokeError("No applicable public salary record")
    latest = max(s["effective"] for s in applicable)
    amounts = {_cents(s.get("monthly_salary_cents"), "salary") for s in applicable if s["effective"] == latest}
    if len(amounts) != 1:
        issues["SALARY_CONFLICT"] = "salary_records"
        clauses.add("R2")
    if payroll.get("ytd_pensionable_cents") is None:
        issues["MISSING_INPUT"] = "ytd_pensionable_cents"
        clauses.add("R4")
    if issues:
        fields = sorted(set(issues.values()))
        return {"decision": "needs_information", "result": None, "issue_codes": sorted(issues),
                "missing_fields": fields, "citations": sorted(clauses),
                "explanation": "Please provide authoritative " + ", ".join(fields)
                + "; the current evidence does not determine a safe payslip."}
    schedule = active[0]
    days = payroll.get("period_days")
    paid = payroll.get("paid_days")
    if (type(days) is not int or days != calendar.monthrange(pay_date.year, pay_date.month)[1]
            or type(paid) is not int or not 0 <= paid <= days or payroll.get("ytd_year") != pay_date.year):
        raise SmokeError("Unsupported public attendance or YTD year")
    salary = next(iter(amounts))
    bonus = _cents(payroll.get("bonus_cents"), "bonus")
    ytd = _cents(payroll.get("ytd_pensionable_cents"), "YTD earnings")
    ceiling = _cents(schedule.get("annual_ceiling_cents"), "contribution ceiling")
    allowance = _cents(schedule.get("allowance_cents"), "allowance")

    def rounded(value: Decimal) -> int:
        return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))

    gross = rounded(Decimal(salary) * Decimal(paid) / Decimal(days) + Decimal(bonus))
    contribution = rounded(Decimal(min(gross, max(ceiling - ytd, 0))) * Decimal(schedule["contribution_rate"]))
    taxable = max(gross - contribution - allowance, 0)
    tax_total = Decimal(0)
    lower = 0
    for upper, rate in schedule["bands"]:
        slice_end = taxable if upper is None else min(taxable, upper)
        tax_total += Decimal(max(slice_end - lower, 0)) * Decimal(rate)
        if upper is not None:
            lower = upper
    tax = rounded(tax_total)
    return {"decision": "answer", "result": {"gross_cents": gross, "contribution_cents": contribution,
            "taxable_cents": taxable, "tax_cents": tax, "net_cents": gross - contribution - tax},
            "issue_codes": [], "missing_fields": [],
            "citations": ["R1", "R10", "R2", "R3", "R4", "R5", "R6", "R7", "R8"], "explanation": ""}


def write_json(path: Path, value: dict, *, private: bool) -> None:
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise SmokeError("Receipt/proof paths must not use symlinks")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".aster-smoke-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.chmod(temporary, 0o600 if private else 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _complete(run: dict, issued: dict, payload: dict, tier: int) -> bool:
    if run.get("status") != "complete":
        return False
    scores = run.get("scores", [])
    if len(scores) != 1:
        raise SmokeError("Unexpected public score count")
    value = scores[0].get("score")
    if (scores[0].get("status") != "complete" or type(value) not in (int, float)
            or not math.isfinite(value) or value < (1.0 if tier == 2 else .975)):
        raise SmokeError("Independently derived answer did not receive a passing score")
    transcript = run.get("transcript", {})
    if (run.get("run_id") != issued["run_id"] or transcript.get("tasks") != issued["tasks"]
            or transcript.get("submitted_answers") != payload["answers"] or transcript.get("scores") != scores):
        raise SmokeError("Public saved transcript differs from the issued task or submitted answer")
    return True


def run_smoke(args: argparse.Namespace) -> dict:
    options = {"timeout": args.timeout, "attempts": args.attempts, "backoff": args.backoff}
    start = time.monotonic()
    if args.receipt.resolve() == args.proof.resolve():
        raise SmokeError("Private receipt and public proof must use different paths")
    if args.verify_restart:
        if not args.restart_confirmed:
            raise SmokeError("Restart the service first, then add --restart-confirmed; a refetch alone proves no restart")
        receipt = decode(args.receipt.read_bytes())
        base, local = checked_base(receipt["base_url"])
        if args.url and checked_base(args.url)[0] != base:
            raise SmokeError("URL differs from the private receipt")
        if not receipt.get("completed_run"):
            raise SmokeError("Saved run is not complete; pending scoring is not persistence acceptance")
        status, current = _request(base, "/runs/" + receipt["issued"]["run_id"],
                                   token=receipt["issued"]["run_token"], **options)
        _complete(current, receipt["issued"], receipt["payload"], receipt["tier"])
        if status != 200 or fingerprint(current) != fingerprint(receipt["completed_run"]):
            raise SmokeError("Saved run changed or disappeared after the confirmed restart")
        proof = {**receipt["proof"], "status": "restart_persistence_verified",
                 "restart_confirmation": "Operator explicitly confirmed service restart before this request",
                 "restart_retrieval_status": status, "post_restart_run_hash": fingerprint(current)}
    else:
        if not args.url and not args.resume:
            raise SmokeError("Supply the sandbox base URL for the first check")
        if args.resume:
            receipt = decode(args.receipt.read_bytes())
            base, local = checked_base(receipt["base_url"])
            if args.url and checked_base(args.url)[0] != base:
                raise SmokeError("URL differs from the private receipt")
            issued, payload, proof = receipt["issued"], receipt["payload"], receipt["proof"]
            args.tier = receipt["tier"]
        else:
            if args.receipt.exists():
                raise SmokeError("Private receipt already exists; use --resume, --verify-restart, or a new receipt path")
            base, local = checked_base(args.url)
        health_status, health = _request(base, "/healthz", **options)
        if (health_status != 200 or health.get("status") != "ok"
                or health.get("ruleset_version") != RULESET or health.get("reward_version") != "1.0"):
            raise SmokeError("Sandbox is not healthy")
        if not args.resume:
            fetch_status, issued = _request(base, f"/tasks?tier={args.tier}&n=1", **options)
            if fetch_status != 200 or len(issued.get("tasks", [])) != 1:
                raise SmokeError("Sandbox did not issue exactly one public task")
            answer = independent_answer(issued["tasks"][0])
            payload = {"run_id": issued["run_id"], "answers": [{"task_id": issued["tasks"][0]["id"],
                                                               "answer": answer}]}
            proof = {"status": "submission_pending", "evidence_kind": "independent_public_contract_smoke",
                     "base_url": base, "deployment_scope": "local_only" if local else "public_endpoint",
                     "tier": args.tier,
                     "answer_source": "Independent Decimal calculation from issued public documents",
                     "agent_model_calls": 0, "remote_judge_required": args.tier == 3,
                     "server_versions": {key: health[key] for key in ("ruleset_version", "reward_version")},
                     "health_status": health_status, "fetch_status": fetch_status,
                     "run_id": issued["run_id"], "submitted_answer_hash": fingerprint(payload),
                     "task_context_hash": fingerprint(issued["tasks"]),
                     "limitations": ["Tier-3 explanation grading requires the configured remote judge"]
                     if args.tier == 3 else []}
            receipt = {"base_url": base, "issued": issued, "payload": payload, "tier": args.tier,
                       "proof": proof, "completed_run": None}
        # Persist before sending: a timeout may occur after the server accepted the answer.
        write_json(args.receipt, receipt, private=True)
        submit_status = 202
        result: dict = {}
        for attempt in range(args.attempts):
            submit_status, result = _request(base, "/submit", payload=payload,
                                             token=issued["run_token"], **options)
            if submit_status != 202 or result.get("status") == "complete":
                break
            if attempt + 1 < args.attempts:
                time.sleep(args.backoff)
        retrieval_status, current = _request(base, "/runs/" + issued["run_id"], token=issued["run_token"], **options)
        complete = _complete(current, issued, payload, args.tier)
        if retrieval_status != 200 or (complete and submit_status != 200):
            raise SmokeError("Completed submission/retrieval requires HTTP 200")
        if fingerprint(current) != fingerprint(result):
            raise SmokeError("Submission and immediate retrieval disagree")
        proof.update(status="submission_verified" if complete else "pending_judge_or_scoring",
                     submit_status=submit_status, retrieval_status=retrieval_status,
                     score=(current.get("scores") or [{}])[0].get("score"), run_state=current.get("status"),
                     completed_run_hash=fingerprint(current) if complete else None)
        receipt.update(proof=proof, completed_run=current if complete else None)
        write_json(args.receipt, receipt, private=True)
    proof.update(verified_at=datetime.now(timezone.utc).isoformat(), elapsed_s=round(time.monotonic() - start, 3))
    write_json(args.proof, proof, private=False)
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", help="Root public HTTPS URL; localhost HTTP is allowed for local checks")
    parser.add_argument("--tier", type=int, choices=(2, 3), default=2)
    parser.add_argument("--receipt", type=Path, default=Path("tmp/sandbox-smoke-private.json"))
    parser.add_argument("--proof", type=Path, default=Path("tmp/sandbox-smoke-proof.json"))
    parser.add_argument("--verify-restart", action="store_true")
    parser.add_argument("--restart-confirmed", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Retry the saved immutable submission; issue no new task")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--attempts", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--backoff", type=float, default=2)
    args = parser.parse_args()
    if args.resume and args.verify_restart:
        parser.error("--resume and --verify-restart are mutually exclusive")
    if not 0 < args.timeout <= 120 or not 0 <= args.backoff <= 10:
        parser.error("timeout must be 0..120 seconds and backoff 0..10 seconds")
    try:
        proof = run_smoke(args)
        print(json.dumps(proof, indent=2))
        if proof["status"] == "pending_judge_or_scoring":
            raise SystemExit(2)
    except (SmokeError, OSError, KeyError, TypeError, ValueError) as error:
        # Only safe SmokeError strings are printed; private connection/payload exceptions are not.
        message = str(error) if isinstance(error, SmokeError) else "Invalid private receipt or unsupported response"
        print(json.dumps({"status": "failed", "error": message}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
