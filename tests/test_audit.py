"""Independent clause, adversarial reward, and public-boundary regression checks.

These lightweight fixtures are software tests, never genuine model evidence.
"""

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from aster_gym.api import create_app
from aster_gym.generator import (
    input_fingerprint,
    public_task,
    task_from_inputs,
    validate_splits,
    validate_taskset,
)
from aster_gym.public import PublicTask
from aster_gym.reference import authoritative_schedules, solve
from aster_gym.schemas import JudgeResult, Task
from aster_gym.scoring import score
from aster_gym.tools import ToolSession, calculate, parse_tool_arguments


def valid_inputs():
    return {
        "kind": "payslip", "pay_date": "2030-01-31", "period_days": 31,
        "paid_days": 31, "bonus_cents": 0, "ytd_year": 2030,
        "ytd_pensionable_cents": 1200000,
        "salary_records": [{"effective": "2030-01-01", "signed": True, "monthly_salary_cents": 300000}],
        "schedules": authoritative_schedules(),
    }


@pytest.mark.parametrize("salary,tax,net", [
    (110000, 0, 110000), (110001, 0, 110001), (110005, 1, 110004),
    (310000, 20000, 290000), (310005, 20001, 290004),
])
def test_r6_marginal_boundaries_hand_derived(salary, tax, net):
    inputs = valid_inputs()
    inputs["salary_records"][0]["monthly_salary_cents"] = salary
    actual = solve(inputs)
    assert actual.result["tax_cents"] == tax
    assert actual.result["net_cents"] == net


def test_r4_contribution_half_cent_and_zero_attendance():
    inputs = valid_inputs()
    inputs["salary_records"][0]["monthly_salary_cents"] = 10
    inputs["ytd_pensionable_cents"] = 0
    assert solve(inputs).result == {
        "gross_cents": 10, "contribution_cents": 1, "taxable_cents": 0,
        "tax_cents": 0, "net_cents": 9,
    }
    inputs["paid_days"] = 0
    assert set(solve(inputs).result.values()) == {0}


def test_r8_2029_schedule_is_active_only_in_its_interval():
    inputs = valid_inputs()
    inputs.update(pay_date="2029-01-31", ytd_year=2029, ytd_pensionable_cents=0)
    inputs["salary_records"][0]["effective"] = "2029-01-01"
    assert solve(inputs).result == {
        "gross_cents": 300000, "contribution_cents": 12000, "taxable_cents": 280000,
        "tax_cents": 18000, "net_cents": 270000,
    }


@pytest.mark.parametrize("mutation", [
    lambda schedule: schedule.update(annual_ceiling_cents=1200000.0),
    lambda schedule: schedule.update(allowance_cents=10000.0),
    lambda schedule: schedule["bands"][0].__setitem__(0, 100000.0),
])
def test_r1_schedule_cents_do_not_accept_equal_floats(mutation):
    inputs = valid_inputs()
    mutation(inputs["schedules"][1])
    assert solve(inputs).decision == "needs_information"
    assert "INVALID_INPUT" in solve(inputs).issue_codes


def test_r2_irrelevant_records_cannot_force_unjustified_abstention():
    inputs = valid_inputs()
    expected = solve(inputs).result
    inputs["salary_records"] += [
        {"signed": False, "effective": "bad", "monthly_salary_cents": "do not calculate"},
        {"signed": True, "effective": "2030-02-01", "monthly_salary_cents": -1},
        {"signed": True, "effective": "2029-12-01", "monthly_salary_cents": "stale placeholder"},
    ]
    assert solve(inputs).result == expected
    inputs["salary_records"].append({"signed": True, "effective": "2030-01-20", "monthly_salary_cents": True})
    assert "INVALID_INPUT" in solve(inputs).issue_codes


def test_r9_independent_missing_and_conflicting_issues_all_reported():
    inputs = valid_inputs()
    del inputs["bonus_cents"]
    del inputs["ytd_pensionable_cents"]
    inputs["salary_records"].append({"signed": True, "effective": "2030-01-01", "monthly_salary_cents": 1})
    answer = solve(inputs)
    assert answer.issue_codes == ["MISSING_INPUT", "SALARY_CONFLICT"]
    assert answer.missing_fields == ["bonus_cents", "salary_records", "ytd_pensionable_cents"]
    assert set(answer.citations) == {"R2", "R3", "R4", "R9", "R10"}


def test_normalized_document_order_cannot_evade_split_leakage():
    inputs = valid_inputs()
    inputs["salary_records"].append({"signed": True, "effective": "2029-01-01", "monthly_salary_cents": 1})
    reordered = copy.deepcopy(inputs)
    reordered["salary_records"].reverse()
    reordered["schedules"].reverse()
    assert input_fingerprint(inputs) == input_fingerprint(reordered)
    train = task_from_inputs(inputs, seed=5001, difficulty=2)
    holdout = task_from_inputs(reordered, seed=6001, difficulty=2)
    with pytest.raises(ValueError, match="SPLIT_INPUT_LEAKAGE"):
        validate_splits({"train": [train], "eval": [holdout]})


def test_tampered_ground_truth_and_reference_documents_rejected():
    task = task_from_inputs(valid_inputs(), seed=1, difficulty=2)
    task.ground_truth.result["net_cents"] += 1
    with pytest.raises(ValueError, match="reference disagreement"):
        validate_taskset([task])
    task = task_from_inputs(valid_inputs(), seed=1, difficulty=2)
    task.context_files["payroll-records.json"] = "{}"
    with pytest.raises(ValueError, match="payroll context"):
        validate_taskset([task])


@pytest.mark.parametrize("difficulty", [True, "1", 1.0, 0, 4])
def test_task_tier_is_bounded_strict_integer(difficulty):
    payload = task_from_inputs(valid_inputs(), seed=1, difficulty=2).model_dump()
    payload["difficulty"] = difficulty
    with pytest.raises(ValidationError):
        Task.model_validate(payload)


class MustNotCallJudge:
    async def grade(self, *args):
        raise AssertionError("deterministic gates must run before the explanation judge")


@pytest.mark.asyncio
async def test_hard_gates_precede_judge_and_length_or_citation_hacking():
    ordinary = task_from_inputs(valid_inputs(), seed=1, difficulty=2)
    inputs = valid_inputs()
    del inputs["ytd_pensionable_cents"]
    trap = task_from_inputs(inputs, seed=2, difficulty=3)
    confident = ordinary.ground_truth.model_dump()
    assert (await score(trap, confident, MustNotCallJudge())).gate == "UNSAFE_COMPUTATION"
    abstention = trap.ground_truth.model_dump()
    assert (await score(ordinary, abstention, MustNotCallJudge())).gate == "UNJUSTIFIED_ABSTENTION"
    wrong_blocker = copy.deepcopy(abstention)
    wrong_blocker["missing_fields"] = ["salary_records"]
    graded = await score(trap, wrong_blocker, MustNotCallJudge())
    assert graded.gate == "CORRECTNESS_CAP" and graded.score <= .20
    wrong_citation = copy.deepcopy(confident)
    wrong_citation["citations"] = ["R999"]
    assert (await score(ordinary, wrong_citation, MustNotCallJudge())).score <= .20
    too_long = copy.deepcopy(confident)
    too_long["explanation"] = "x" * 401
    assert (await score(ordinary, too_long)).gate == "INVALID_CONTRACT"


@pytest.mark.asyncio
async def test_wrong_amount_partial_credit_is_exact_and_not_judge_dependent():
    task = task_from_inputs(valid_inputs(), seed=1, difficulty=2)
    answer = task.ground_truth.model_dump()
    answer["result"]["net_cents"] += 1
    report = await score(task, answer, MustNotCallJudge())
    assert report.score == .20  # Four of five cents fields earn .25 * .8.
    assert report.components["fields"].score == .8
    assert report.components["exact"].score == 0


@pytest.mark.asyncio
async def test_eligible_blocking_explanation_outage_stays_pending():
    class UnavailableJudge:
        async def grade(self, task, answer):
            return JudgeResult(score=0, status="pending", code="JUDGE_QUOTA_EXHAUSTED")
    inputs = valid_inputs()
    del inputs["ytd_pensionable_cents"]
    task = task_from_inputs(inputs, seed=1, difficulty=3)
    report = await score(task, task.ground_truth, UnavailableJudge())
    assert report.score is None and report.status == "pending"
    assert report.error_code == "JUDGE_QUOTA_EXHAUSTED"


@pytest.mark.parametrize("expression", [
    "__import__('os').environ", "2 ** 999999", "[1,2]", "True + 1", "1 / 0", "1e1000", "9" * 257,
])
def test_calculator_attacks_fail_closed(expression):
    assert calculate(expression) == {"error_code": "INVALID_EXPRESSION"}


def test_tool_scope_limit_duplicate_json_and_no_oracle_access():
    task = task_from_inputs(valid_inputs(), seed=1, difficulty=2)
    tools = ToolSession(task, max_calls=3)
    assert tools.call("read_document", {"document_id": "../ground_truth"}) == {"error_code": "DOCUMENT_NOT_FOUND"}
    assert tools.call("read_document", {"document_id": "ground_truth"}) == {"error_code": "DOCUMENT_NOT_FOUND"}
    assert tools.call("calculate", {"expression": "1 + 2"})["value"] == "3"
    assert tools.call("read_document", {"document_id": "rules.md"}) == {"error_code": "TOOL_LIMIT"}
    assert parse_tool_arguments('{"expression":"1","expression":"2"}') is None
    assert parse_tool_arguments('{"expression":NaN}') is None


def test_private_oracle_sentinel_is_absent_from_api_and_public_dto(tmp_path: Path):
    app = create_app(f"sqlite:///{tmp_path / 'private.sqlite'}", judge=None)
    with TestClient(app) as client:
        run = client.get("/tasks?tier=2&n=1").json()
        row = app.state.store.get_run(run["run_id"], run["run_token"])
        sentinel = "PRIVATE_EXPECTED_ANSWER_DO_NOT_SERIALIZE"
        tasks = json.loads(row["tasks_json"])
        tasks[0]["ground_truth"]["explanation"] = sentinel
        with app.state.store.transaction() as connection:
            from sqlalchemy import update
            connection.execute(update(app.state.store.runs).where(app.state.store.runs.c.id == run["run_id"])
                               .values(tasks_json=json.dumps(tasks)))
        headers = {"X-Run-Token": run["run_token"]}
        before = client.get(f"/runs/{run['run_id']}", headers=headers)
        submitted = client.post("/submit", json={"run_id": run["run_id"], "answers": [
            {"task_id": run["tasks"][0]["id"], "answer": "{}"}]}, headers=headers)
        after = client.get(f"/runs/{run['run_id']}", headers=headers)
        assert all(sentinel not in response.text for response in (before, submitted, after))
        assert submitted.status_code == 200
        assert "ground_truth" not in json.dumps(client.get("/openapi.json").json()["components"]["schemas"])
    private_task = task_from_inputs(valid_inputs(), seed=1, difficulty=2)
    with pytest.raises(ValidationError):
        PublicTask.model_validate(private_task.model_dump())
    assert PublicTask.model_validate(public_task(private_task)).context_files


def test_irrelevant_payroll_cannot_hide_same_retrieval_across_splits():
    original = {"kind": "retrieval", "pay_date": "2030-01-31", "query_field": "allowance_cents",
                "schedules": authoritative_schedules(), "bonus_cents": 100}
    other = {**original, "bonus_cents": 987654, "salary_records": []}
    assert input_fingerprint(original) == input_fingerprint(other)
    first = task_from_inputs(original, seed=9001, difficulty=1)
    second = task_from_inputs(other, seed=9002, difficulty=1)
    with pytest.raises(ValueError, match="SPLIT_INPUT_LEAKAGE"):
        validate_splits({"train": [first], "evaluation": [second]})


def test_generator_exclusion_is_deterministic_and_reports_template_inflation():
    from aster_gym.generator import generate_taskset
    first = generate_taskset(12, seed=78, tier=1)
    excluded = {t.input_hash for t in first}
    second = generate_taskset(12, seed=78, tier=1, exclude_fingerprints=excluded)
    repeat = generate_taskset(12, seed=78, tier=1, exclude_fingerprints=set(reversed(sorted(excluded))))
    assert [t.model_dump() for t in second] == [t.model_dump() for t in repeat]
    assert not excluded & {t.input_hash for t in second}
    report = validate_taskset(second)
    assert sum(report["retrieval_template_groups"].values()) == 12
    assert len(report["retrieval_template_groups"]) <= 4
