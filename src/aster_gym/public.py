"""Oracle-free API DTOs: no dependency on Task, Answer, or internal run rows."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)


class PublicTask(PublicModel):
    id: str
    prompt: str
    context_files: dict[str, str]
    ruleset_version: str
    schema_version: str
    tools_available: list[str]


class PublicVersions(PublicModel):
    ruleset: str
    reward: str
    schema_version: str = Field(alias="schema")


class PublicComponent(PublicModel):
    score: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=1)
    clauses: list[str]
    code: str


class PublicScore(PublicModel):
    task_id: str
    status: Literal["complete", "pending"]
    score: float | None = Field(ge=0, le=1)
    gate: str | None
    error_code: str | None
    ruleset_version: str
    reward_version: str
    components: dict[str, PublicComponent]


class PublicSubmittedAnswer(PublicModel):
    task_id: str
    answer: str | dict[str, Any]


class PublicTranscript(PublicModel):
    tasks: list[PublicTask]
    submitted_answers: list[PublicSubmittedAnswer]
    scores: list[PublicScore]


class PublicIssuedRun(PublicModel):
    run_id: str
    run_token: str
    seed_fingerprint: str
    versions: PublicVersions
    tier_mix: dict[str, int]
    tasks: list[PublicTask]


class PublicRun(PublicModel):
    run_id: str
    status: Literal["issued", "scoring", "pending", "complete"]
    versions: PublicVersions
    tier_mix: dict[str, int]
    seed_fingerprint: str
    scores: list[PublicScore]
    retryable: bool
    transcript: PublicTranscript
