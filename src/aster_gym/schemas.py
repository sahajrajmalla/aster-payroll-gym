"""Strict contracts for internal records; never serialize Task to a public API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from .versions import REWARD_VERSION, RULESET_VERSION, SCHEMA_VERSION


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Answer(StrictModel):
    decision: Literal["answer", "needs_information"]
    result: dict[str, StrictInt] | None
    issue_codes: list[str] = Field(max_length=12)
    missing_fields: list[str] = Field(max_length=20)
    citations: list[str] = Field(max_length=11)
    explanation: str = Field(max_length=400)

    @field_validator("issue_codes", "missing_fields", "citations")
    @classmethod
    def unique_items(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)) or any(not v or len(v) > 80 for v in values):
            raise ValueError("lists require unique, nonempty bounded strings")
        return values

    @field_validator("result")
    @classmethod
    def bounded_result(cls, values: dict[str, int] | None) -> dict[str, int] | None:
        if values is not None and (
            len(values) > 8 or any(not k or len(k) > 64 or not 0 <= v <= 200_000_000 for k, v in values.items())
        ):
            raise ValueError("result exceeds contract limits")
        return values

    @model_validator(mode="after")
    def disposition_shape(self) -> "Answer":
        # R9/R10: unresolved work may never carry money, even in unused fields.
        if self.decision == "needs_information" and self.result is not None:
            raise ValueError("unresolved work requires result:null")
        if self.decision == "answer" and self.result is None:
            raise ValueError("completed work requires a result object")
        return self


class Task(StrictModel):
    id: str = Field(min_length=1, max_length=80)
    slice: Literal["monthly-payroll"] = "monthly-payroll"
    difficulty: StrictInt = Field(ge=1, le=3)
    prompt: str = Field(min_length=1, max_length=8000)
    context_files: dict[str, str]
    ground_truth: Answer
    verifier: Literal["aster-exact-cents-v1"] = "aster-exact-cents-v1"
    tags: list[str]
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    ruleset_version: Literal["ASTER-1.0"] = RULESET_VERSION
    seed: StrictInt
    inputs: dict[str, Any]
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class ComponentScore(StrictModel):
    score: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=1)
    clauses: list[str]
    code: str


class JudgeResult(StrictModel):
    score: float = Field(ge=0, le=1)
    status: Literal["complete", "pending"] = "complete"
    code: str = "JUDGE_GRADED"


class ScoreReport(StrictModel):
    task_id: str
    status: Literal["complete", "pending"]
    score: float | None = Field(default=None, ge=0, le=1)
    components: dict[str, ComponentScore]
    gate: str | None = None
    error_code: str | None = None
    ruleset_version: Literal["ASTER-1.0"] = RULESET_VERSION
    reward_version: Literal["1.0"] = REWARD_VERSION
