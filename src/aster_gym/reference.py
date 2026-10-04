"""Independent calculator implementing the written R1–R11 authority.

This module never imports the generator or an LLM. Authoritative schedule numbers
are transcribed from R8; hand-derived tests and the human fidelity audit check them.
"""

import calendar
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from .schemas import Answer


def authoritative_schedules() -> list[dict[str, Any]]:
    """R8: return fresh schedule objects so task mutation cannot alter authority."""
    return [
        {"id": "ASTER-2029", "start": "2029-01-01", "end": "2029-12-31",
         "contribution_rate": "0.04", "annual_ceiling_cents": 1_000_000,
         "allowance_cents": 8000, "bands": [[100_000, "0"], [300_000, "0.10"], [None, "0.20"]]},
        {"id": "ASTER-2030", "start": "2030-01-01", "end": "2030-12-31",
         "contribution_rate": "0.05", "annual_ceiling_cents": 1_200_000,
         "allowance_cents": 10_000, "bands": [[100_000, "0"], [300_000, "0.10"], [None, "0.20"]]},
    ]


def _round(value: Decimal) -> int:
    """R1/R3/R4/R6: half up to exactly one integer cent."""
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _money(value: Any) -> bool:
    """R1: bool is intentionally not an integer input."""
    return type(value) is int and 0 <= value <= 100_000_000


def _same_typed(left: Any, right: Any) -> bool:
    """R1/R8: equality must not silently accept floats or booleans as cents."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_typed(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_typed(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def solve(inputs: dict[str, Any]) -> Answer:
    """R1–R11: determine disposition from evidence, never a task's tier or tags."""
    issues: dict[str, set[str]] = {}
    clauses: set[str] = {"R1", "R10"}
    blocking_clauses: set[str] = {"R9", "R10"}

    def issue(code: str, field: str, clause: str) -> None:
        issues.setdefault(code, set()).add(field)
        clauses.update([clause, "R9"])
        blocking_clauses.add(clause)

    def require(field: str, validator, clause: str) -> Any:
        if field not in inputs or inputs[field] is None:
            issue("MISSING_INPUT", field, clause)
            return None
        value = inputs[field]
        if not validator(value):
            issue("INVALID_INPUT", field, clause)
            return None
        return value

    def parse_date(value: Any) -> date | None:
        try:
            if not isinstance(value, str) or len(value) != 10:
                return None
            parsed = date.fromisoformat(value)
            return parsed if parsed.isoformat() == value else None
        except ValueError:
            return None

    raw_date = require("pay_date", lambda v: parse_date(v) is not None, "R8")
    pay_date = parse_date(raw_date)
    kind = inputs.get("kind")
    if kind not in ("retrieval", "payslip"):
        issue("INVALID_INPUT", "kind", "R10")

    # R8: context schedules must be faithful copies of the separately written authority.
    records = require("schedules", lambda v: isinstance(v, list) and len(v) <= 10, "R8")
    schedule = None
    if records is not None:
        authority = {s["id"]: s for s in authoritative_schedules()}
        valid = []
        for record in records:
            if (not isinstance(record, dict) or not isinstance(record.get("id"), str)
                    or not _same_typed(record, authority.get(record.get("id")))):
                issue("INVALID_INPUT", "schedules", "R8")
                continue
            if pay_date and record["start"] <= pay_date.isoformat() <= record["end"]:
                valid.append(record)
        if pay_date:
            if not valid:
                issue("NO_SCHEDULE", "schedules", "R8")
            elif len(valid) > 1:
                issue("SCHEDULE_CONFLICT", "schedules", "R8")
            else:
                schedule = valid[0]
        clauses.add("R8")

    salary = None
    paid = days = bonus = ytd = None
    if kind == "retrieval":
        clauses.add("R11")
        require("query_field", lambda v: v in ("annual_ceiling_cents", "allowance_cents"), "R11")
    elif kind == "payslip":
        # R2: latest applicable signed salary, with equal-authority conflict detection.
        salaries = require("salary_records", lambda v: isinstance(v, list) and len(v) <= 20, "R2")
        applicable = []
        if salaries is not None:
            for record in salaries:
                if not isinstance(record, dict):
                    issue("INVALID_INPUT", "salary_records", "R2")
                    continue
                if type(record.get("signed")) is not bool:
                    issue("INVALID_INPUT", "salary_records", "R2")
                    continue
                # R2 explicitly excludes unsigned and future records from authority.
                if not record["signed"]:
                    continue
                effective = parse_date(record.get("effective"))
                if effective is None:
                    issue("INVALID_INPUT", "salary_records", "R2")
                    continue
                if pay_date and effective > pay_date:
                    continue
                if pay_date:
                    applicable.append(record)
            if pay_date:
                if not applicable:
                    issue("NO_SALARY", "salary_records", "R2")
                else:
                    latest = max(r["effective"] for r in applicable)
                    current = [r for r in applicable if r["effective"] == latest]
                    # R2: stale signed records cannot override a valid newer one.
                    if any(not _money(r.get("monthly_salary_cents")) for r in current):
                        issue("INVALID_INPUT", "salary_records", "R2")
                    else:
                        candidates = {r["monthly_salary_cents"] for r in current}
                        if len(candidates) != 1:
                            issue("SALARY_CONFLICT", "salary_records", "R2")
                        else:
                            salary = candidates.pop()
        clauses.update(["R2", "R3", "R4", "R5", "R6", "R7"])
        days = require("period_days", lambda v: type(v) is int and 28 <= v <= 31, "R1")
        paid = require("paid_days", lambda v: type(v) is int and 0 <= v <= 31, "R3")
        if pay_date and days is not None and days != calendar.monthrange(pay_date.year, pay_date.month)[1]:
            issue("INVALID_INPUT", "period_days", "R1")
        if paid is not None and days is not None and paid > days:
            issue("INVALID_INPUT", "paid_days", "R3")
        bonus = require("bonus_cents", _money, "R3")
        ytd = require("ytd_pensionable_cents", _money, "R4")
        year = require("ytd_year", lambda v: type(v) is int and 1900 <= v <= 2200, "R4")
        if pay_date and year is not None and year != pay_date.year:
            issue("INVALID_INPUT", "ytd_year", "R4")

    if issues:
        fields = sorted({field for group in issues.values() for field in group})
        return Answer(decision="needs_information", result=None, issue_codes=sorted(issues),
                      missing_fields=fields, citations=sorted(blocking_clauses),
                      explanation="Resolve " + ", ".join(fields) + " using authoritative evidence before calculation.")

    assert schedule is not None
    if kind == "retrieval":
        return Answer(decision="answer", result={inputs["query_field"]: schedule[inputs["query_field"]]},
                      issue_codes=[], missing_fields=[], citations=sorted(clauses), explanation="")

    assert salary is not None and paid is not None and days is not None and bonus is not None and ytd is not None
    # R3: single rounding after salary proration and bonus addition.
    gross = _round(Decimal(salary) * Decimal(paid) / Decimal(days) + Decimal(bonus))
    # R4: contribution ceiling applies to pensionable earnings, not contribution paid.
    remaining = max(schedule["annual_ceiling_cents"] - ytd, 0)
    contribution = _round(Decimal(min(gross, remaining)) * Decimal(schedule["contribution_rate"]))
    # R5: allowance is not prorated; the taxable base cannot be negative.
    taxable = max(gross - contribution - schedule["allowance_cents"], 0)
    # R6: marginal slices, round sum only once.
    tax_amount = Decimal(0)
    lower = 0
    for upper, rate in schedule["bands"]:
        width = max(min(taxable, upper) - lower, 0) if upper is not None else max(taxable - lower, 0)
        tax_amount += Decimal(width) * Decimal(rate)
        if upper is not None:
            lower = upper
    tax = _round(tax_amount)
    # R7: identity with no independently rounded net.
    result = {"gross_cents": gross, "contribution_cents": contribution, "taxable_cents": taxable,
              "tax_cents": tax, "net_cents": gross - contribution - tax}
    return Answer(decision="answer", result=result, issue_codes=[], missing_fields=[],
                  citations=sorted(clauses), explanation="")
