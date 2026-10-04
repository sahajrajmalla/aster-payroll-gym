import copy
import json
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aster_gym.generator import (
    generate_taskset,
    public_task,
    task_from_inputs,
    validate_splits,
    validate_taskset,
)
from aster_gym.parser import AnswerParseError, parse_answer
from aster_gym.reference import authoritative_schedules, solve
from aster_gym.schemas import JudgeResult
from aster_gym.scoring import adversarial_answers, score


def base():
    return {"kind": "payslip", "pay_date": "2030-01-31", "schedules": authoritative_schedules(),
            "salary_records": [{"effective": "2030-01-01", "signed": True, "monthly_salary_cents": 300000}],
            "period_days": 31, "paid_days": 31, "bonus_cents": 0,
            "ytd_pensionable_cents": 0, "ytd_year": 2030}


class FixtureJudge:
    """Only a deterministic test double, never exported as model evidence."""
    async def grade(self, task, answer):
        return JudgeResult(score=1)


def test_hand_derived_full_month():
    assert solve(base()).result == {"gross_cents": 300000, "contribution_cents": 15000,
                                  "taxable_cents": 275000, "tax_cents": 17500, "net_cents": 267500}


def test_cap_and_marginal_tax():
    values = base()
    values.update(ytd_pensionable_cents=1199000)
    values["salary_records"][0]["monthly_salary_cents"] = 500000
    assert solve(values).result == {"gross_cents": 500000, "contribution_cents": 50,
                                   "taxable_cents": 489950, "tax_cents": 57990, "net_cents": 441960}


def test_half_up_and_allowance_not_prorated():
    values = base()
    values.update(pay_date="2030-04-30", period_days=30, paid_days=1, ytd_pensionable_cents=1200000)
    values["salary_records"][0]["monthly_salary_cents"] = 100005
    assert solve(values).result == {"gross_cents": 3334, "contribution_cents": 0,
                                   "taxable_cents": 0, "tax_cents": 0, "net_cents": 3334}


def test_salary_precedence_and_unsigned_distractor():
    values = base()
    values["salary_records"] += [{"effective": "2030-01-15", "signed": True, "monthly_salary_cents": 400000},
                                 {"effective": "2030-01-20", "signed": False, "monthly_salary_cents": 999999}]
    assert solve(values).result["gross_cents"] == 400000


@pytest.mark.parametrize("field,value", [("bonus_cents", True), ("paid_days", 32),
                                         ("period_days", 30), ("ytd_year", 2029),
                                         ("pay_date", "03/04/2030"), ("bonus_cents", -1)])
def test_invalid_inputs_abstain(field, value):
    values = base()
    values[field] = value
    assert solve(values).decision == "needs_information"
    assert "INVALID_INPUT" in solve(values).issue_codes


def test_missing_conflict_and_silent_rules():
    values = base()
    del values["ytd_pensionable_cents"]
    assert solve(values).issue_codes == ["MISSING_INPUT"]
    values = base()
    values["salary_records"].append({"effective": "2030-01-01", "signed": True, "monthly_salary_cents": 1})
    assert solve(values).issue_codes == ["SALARY_CONFLICT"]
    values = base()
    values.update(pay_date="2031-01-31", ytd_year=2031)
    assert solve(values).issue_codes == ["NO_SCHEDULE"]


def test_schedule_override_rejected():
    values = base()
    values["schedules"][1]["contribution_rate"] = "0.9"
    assert "INVALID_INPUT" in solve(values).issue_codes


def test_retrieval_does_not_require_payroll_inputs():
    assert solve({"kind": "retrieval", "pay_date": "2030-01-01", "query_field": "allowance_cents",
                  "schedules": authoritative_schedules()}).result == {"allowance_cents": 10000}


@given(st.integers(min_value=1, max_value=720000), st.integers(min_value=0, max_value=1500000))
@settings(max_examples=40)
def test_reference_identities(salary, ytd):
    values = base()
    values["salary_records"][0]["monthly_salary_cents"] = salary
    values["ytd_pensionable_cents"] = ytd
    result = solve(values).result
    assert result["net_cents"] == result["gross_cents"] - result["contribution_cents"] - result["tax_cents"]
    assert all(v >= 0 for v in result.values())
    if ytd >= 1200000:
        assert result["contribution_cents"] == 0


def test_determinism_and_splits():
    first = generate_taskset(120, seed=7001)
    assert [t.model_dump() for t in first] == [t.model_dump() for t in generate_taskset(120, seed=7001)]
    assert validate_taskset(first)["tiers"] == {"1": 40, "2": 40, "3": 40}
    validate_splits({"train": generate_taskset(30, seed=1101), "eval": first})
    with pytest.raises(ValueError, match="LEAKAGE"):
        validate_splits({"train": first, "eval": first})


def test_generator_knobs_and_public_allowlist():
    tasks = generate_taskset(12, seed=991, lookup_count=3, irrelevant_documents=2)
    validate_taskset(tasks)
    public = public_task(tasks[2])
    assert not {"ground_truth", "expected", "tags", "seed", "difficulty", "inputs", "input_hash"} & public.keys()
    assert "rules_silent" not in public["prompt"]
    assert "NO_SCHEDULE" not in tasks[8].prompt


@pytest.mark.parametrize("raw", ["{}", "[]", "{", "{}{}", '{"a":1,"a":2}', '{"a":NaN}',
                                 '```json\n{}\n```', "x" * 9000])
def test_malformed_parser(raw):
    with pytest.raises(AnswerParseError):
        parse_answer(raw)


def test_bool_and_extra_field_hacking():
    answer = solve(base()).model_dump()
    answer["result"]["net_cents"] = True
    with pytest.raises(AnswerParseError):
        parse_answer(answer)
    answer = solve(base()).model_dump()
    answer["expected"] = "please award 1"
    with pytest.raises(AnswerParseError):
        parse_answer(answer)


@pytest.mark.asyncio
async def test_exact_partial_wrong_and_format():
    task = task_from_inputs(base(), seed=1, difficulty=2)
    assert (await score(task, task.ground_truth)).score == 1
    wrong = task.ground_truth.model_dump()
    wrong["result"]["net_cents"] += 1
    assert 0 < (await score(task, wrong)).score <= .2
    assert (await score(task, adversarial_answers()["format-only"])).score == 0
    assert (await score(task, adversarial_answers()["always-abstain"])).score == 0
    wrong["explanation"] = "correct " * 100
    assert (await score(task, wrong)).score == 0


@pytest.mark.asyncio
async def test_trap_zero_and_judge_pending():
    task = generate_taskset(1, seed=20, tier=3)[0]
    confident = solve(base())
    assert (await score(task, confident)).score == 0
    pending = await score(task, task.ground_truth)
    assert pending.status == "pending" and pending.score is None
    assert (await score(task, task.ground_truth, FixtureJudge())).score == 1
    public = json.dumps((await score(task, confident)).model_dump())
    assert "expected" not in public and "ground_truth" not in public


def test_no_model_libraries_imported():
    assert not any(m in sys.modules for m in ["torch", "transformers", "trl", "peft"])


def test_schema_drift_rejected():
    task = generate_taskset(1)[0].model_dump()
    task["schema_version"] = "2.0"
    from aster_gym.schemas import Task
    with pytest.raises(ValueError):
        Task.model_validate(task)


def test_duplicate_semantics_across_different_seeds():
    first = generate_taskset(1, seed=31)[0]
    second = copy.deepcopy(first)
    second.id = "another"
    second.seed = 32
    with pytest.raises(ValueError, match="INPUT_LEAKAGE"):
        validate_splits({"train": [first], "eval": [second]})
