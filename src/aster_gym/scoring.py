"""One clause-auditable rubric for eval, cloud RL, and the external sandbox."""

from typing import Any, Literal

from .parser import AnswerParseError, parse_answer
from .schemas import Answer, ComponentScore, ScoreReport, Task
from .versions import REWARD_VERSION, RULESET_VERSION

WEIGHTS = {
    "ordinary": {"exact": .60, "fields": .25, "action": .10, "format": .05},
    "trap": {"exact": .55, "fields": .25, "action": .10, "judge": .05, "format": .05},
}


def _set_f1(expected: list[str], actual: list[str]) -> float:
    target, given = set(expected), set(actual)
    if not target and not given:
        return 1.0
    return 2 * len(target & given) / (len(target) + len(given)) if target or given else 0.0


async def score(task: Task, raw: str | dict | Answer, judge=None) -> ScoreReport:
    """R1/R9/R10: exact cents, graded fields, zero unsafe/unnecessary dispositions."""
    expected = task.ground_truth
    # Applicability depends on the rule-determined disposition, not the tier label.
    weights = WEIGHTS["trap" if expected.decision == "needs_information" else "ordinary"]
    components = {key: ComponentScore(score=0, weight=weight, clauses=["R10"], code="NOT_EARNED")
                  for key, weight in weights.items()}

    def mark(key: str, value: float, clauses: list[str], code: str) -> None:
        components[key] = ComponentScore(score=value, weight=weights[key], clauses=clauses, code=code)

    def report(value: float | None, *, status: Literal["complete", "pending"] = "complete", gate: str | None = None,
               error: str | None = None) -> ScoreReport:
        return ScoreReport(task_id=task.id, status=status, score=value, components=components,
                           gate=gate, error_code=error, ruleset_version=RULESET_VERSION,
                           reward_version=REWARD_VERSION)

    try:
        answer = parse_answer(raw)
    except AnswerParseError:
        return report(0, gate="INVALID_CONTRACT")
    mark("format", 1, ["R1", "R10"], "VALID_CONTRACT")
    if answer.decision != expected.decision:
        return report(0, gate="UNSAFE_COMPUTATION" if expected.decision == "needs_information"
                      else "UNJUSTIFIED_ABSTENTION")
    citations_valid = (set(expected.citations) <= set(answer.citations)
                       and all(c in {f"R{i}" for i in range(1, 12)} for c in answer.citations))
    if expected.decision == "answer":
        wanted = expected.result or {}
        supplied = answer.result or {}
        hits = sum(supplied.get(field) == value for field, value in wanted.items())
        fields = hits / max(len(wanted), len(supplied), 1)
        semantic_exact = wanted == supplied and not answer.issue_codes and not answer.missing_fields
        mark("fields", fields, expected.citations, "FIELD_ACCURACY")
    else:
        fields = (_set_f1(expected.issue_codes, answer.issue_codes)
                  + _set_f1(expected.missing_fields, answer.missing_fields)) / 2
        semantic_exact = (set(expected.issue_codes) == set(answer.issue_codes)
                          and set(expected.missing_fields) == set(answer.missing_fields))
        mark("fields", fields, expected.citations, "BLOCKER_SET_F1")
    exact = semantic_exact and citations_valid
    mark("exact", float(exact), expected.citations, "EXACT_WITH_CLAUSES" if exact else "SUBSTANTIVE_MISMATCH")
    mark("action", float(semantic_exact), ["R9"], "CALIBRATED_ACTION" if semantic_exact else "WRONG_CONTENT")
    if not exact:
        # R9/R10: no formatting or safety bonus can make rejected work look acceptable.
        return report(min(.20, .25 * fields), gate="CORRECTNESS_CAP")
    if "judge" in weights:
        if not answer.explanation.strip():
            return report(.20, gate="MISSING_BLOCKING_EXPLANATION")
        if judge is None:
            components["judge"].code = "JUDGE_UNAVAILABLE"
            return report(None, status="pending", error="JUDGE_UNAVAILABLE")
        try:
            result = await judge.grade(task, answer)
        except Exception:
            # Operational failures are not model failures. Do not disclose exception/expected values.
            return report(None, status="pending", error="JUDGE_UNAVAILABLE")
        if result.status != "complete":
            return report(None, status="pending", error=result.code)
        mark("judge", result.score, ["R10"], result.code)
        if result.score == 0:
            return report(.20, gate="UNUSABLE_BLOCKING_EXPLANATION")
    total = sum(item.weight * item.score for item in components.values())
    return report(round(total, 8))


def make_judge(budget=None):
    """Only remote inference; absent credentials leave applicable scores pending."""
    from .judge import RemoteJudge

    return RemoteJudge.from_env(budget=budget)


def adversarial_answers() -> dict[str, dict[str, Any]]:
    """Handwritten policies, never model outputs or ground truth."""
    common = {"issue_codes": [], "missing_fields": [], "citations": ["R10"], "explanation": ""}
    return {
        "format-only": {**common, "decision": "answer", "result": {}},
        "constant-payslip": {**common, "decision": "answer", "result": {
            "gross_cents": 300000, "contribution_cents": 15000, "taxable_cents": 275000,
            "tax_cents": 17500, "net_cents": 267500}},
        "always-abstain": {**common, "decision": "needs_information", "result": None,
                           "issue_codes": ["MISSING_INPUT"], "missing_fields": ["unknown"],
                           "explanation": "Please provide more information."},
    }
