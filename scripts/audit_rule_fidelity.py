"""Second implementation for the fixed review cohort; integer/rational arithmetic.

No production calculator, generator, schema or model imports. Constants transcribed
from written R8. Unsupported inputs fail closed. This is an automated cross-check,
not a human signature or a second ground-truth generator.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


def half_up(numerator: int, denominator: int) -> int:
    """R1: exact positive rational half-up, independent of Decimal implementation."""
    if numerator < 0 or denominator <= 0:
        raise ValueError("Unsupported rational")
    return (2 * numerator + denominator) // (2 * denominator)


def check_money(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= 100_000_000:
        raise ValueError("Audit supports valid nonnegative integer cents only")
    return value


def independent_result(inputs: dict[str, Any]) -> dict[str, Any]:
    """R1–R11 substantive fields for the frozen 27-row review cohort."""
    paid_date = date.fromisoformat(inputs["pay_date"])
    issues: set[str] = set()
    fields: set[str] = set()

    def block(code: str, field: str) -> None:
        issues.add(code)
        fields.add(field)

    active = []
    for schedule in inputs["schedules"]:
        year = int(schedule["id"].removeprefix("ASTER-"))
        if year not in (2029, 2030):
            raise ValueError("Unsupported schedule")
        expected = {"id": f"ASTER-{year}", "start": f"{year}-01-01", "end": f"{year}-12-31",
                    "contribution_rate": "0.04" if year == 2029 else "0.05",
                    "annual_ceiling_cents": 1_000_000 if year == 2029 else 1_200_000,
                    "allowance_cents": 8000 if year == 2029 else 10000,
                    "bands": [[100000, "0"], [300000, "0.10"], [None, "0.20"]]}
        if json.dumps(schedule, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError("R8: document does not match written schedule")
        if schedule["start"] <= paid_date.isoformat() <= schedule["end"]:
            active.append(year)
    if not active:
        block("NO_SCHEDULE", "schedules")
    elif len(active) != 1:
        block("SCHEDULE_CONFLICT", "schedules")

    salary = 0
    if inputs["kind"] == "payslip":
        records = [record for record in inputs["salary_records"] if record["signed"] is True
                   and date.fromisoformat(record["effective"]) <= paid_date]
        if not records:
            block("NO_SALARY", "salary_records")
        else:
            latest = max(record["effective"] for record in records)
            amounts = {check_money(record["monthly_salary_cents"]) for record in records
                       if record["effective"] == latest}
            if len(amounts) != 1:
                block("SALARY_CONFLICT", "salary_records")
            else:
                salary = amounts.pop()
        for field in ("bonus_cents", "ytd_pensionable_cents", "paid_days", "period_days", "ytd_year"):
            if inputs.get(field) is None:
                block("MISSING_INPUT", field)
        if inputs.get("ytd_year") != paid_date.year:
            raise ValueError("Unsupported unmatched YTD year in audit cohort")
        if (inputs["period_days"] != calendar.monthrange(paid_date.year, paid_date.month)[1]
                or type(inputs["paid_days"]) is not int
                or not 0 <= inputs["paid_days"] <= inputs["period_days"]):
            raise ValueError("Unsupported attendance")
    elif inputs["kind"] != "retrieval":
        raise ValueError("Unsupported request")

    result: dict[str, int] | None = None
    if not issues:
        year = active[0]
        ceiling, allowance, rate = (1_000_000, 8000, 4) if year == 2029 else (1_200_000, 10000, 5)
        if inputs["kind"] == "retrieval":
            query = inputs["query_field"]
            result = {query: {"annual_ceiling_cents": ceiling, "allowance_cents": allowance}[query]}
        else:
            days = inputs["period_days"]
            gross = half_up(salary * inputs["paid_days"] + check_money(inputs["bonus_cents"]) * days, days)
            remainder = max(ceiling - check_money(inputs["ytd_pensionable_cents"]), 0)
            contribution = half_up(min(gross, remainder) * rate, 100)
            taxable = max(gross - contribution - allowance, 0)
            tax = half_up(min(max(taxable - 100000, 0), 200000) + 2 * max(taxable - 300000, 0), 10)
            result = {"gross_cents": gross, "contribution_cents": contribution,
                      "taxable_cents": taxable, "tax_cents": tax, "net_cents": gross - contribution - tax}
    return {"decision": "needs_information" if issues else "answer", "result": result,
            "issue_codes": sorted(issues), "missing_fields": sorted(fields)}


def audit(packet_path: Path, output: Path) -> dict[str, Any]:
    packet = json.loads(packet_path.read_text())
    rows = []
    for group in ("seed_reviews", "fidelity_reviews", "transfer_reviews"):
        for row in packet[group]:
            actual = independent_result(row["inputs"])
            reference = {key: row["reference_result"][key] for key in actual}
            rows.append({"group": group, "task_id": row["task_id"], "independent_result": actual,
                         "matches": actual == reference,
                         "clauses": ["R8", "R9", "R11"] if row["inputs"]["kind"] == "retrieval"
                         else [f"R{number}" for number in range(1, 10)], "human_reviewed": False})
    report = {"status": "passed" if all(row["matches"] for row in rows) else "failed",
              "evidence_kind": "independent_automated_rule_crosscheck", "checked_at": datetime.now(timezone.utc).isoformat(),
              "method": "Exact integer rational arithmetic; separate R8 constants; no production oracle imports",
              "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
              "rules_sha256": hashlib.sha256(Path("rules/aster-payroll-v1.md").read_bytes()).hexdigest(),
              "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "checked": len(rows), "disagreements": sum(not row["matches"] for row in rows),
              "limitation": "Substantive outputs only. Not independent human review, citation fidelity or blind approval.",
              "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=Path("reviews/task-review-packet.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/independent-rule-audit.json"))
    args = parser.parse_args()
    checked = audit(args.packet, args.output)
    print(json.dumps({key: checked[key] for key in ("status", "checked", "disagreements", "limitation")}))
    raise SystemExit(0 if checked["status"] == "passed" else 1)
