"""Public submission sandbox. Oracle-bearing objects never form API responses."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from aster_gym.generator import generate_taskset, public_task
from aster_gym.public import PublicIssuedRun, PublicRun, PublicScore
from aster_gym.schemas import Task
from aster_gym.scoring import make_judge, score
from aster_gym.store import RunStore, StoreError, digest
from aster_gym.versions import REWARD_VERSION, RULESET_VERSION, SCHEMA_VERSION

logger = logging.getLogger(__name__)
_DEFAULT_JUDGE = object()


class SubmittedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(min_length=1, max_length=200)
    answer: str | dict[str, Any]


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=1, max_length=64)
    answers: list[SubmittedAnswer] = Field(min_length=1, max_length=10)


def _report_public(report: Any) -> dict[str, Any]:
    """Allowlist; diagnostics contain rule references, never corrected values."""
    source = report.model_dump(mode="json") if hasattr(report, "model_dump") else report
    keys = ("task_id", "status", "score", "gate", "error_code", "ruleset_version", "reward_version")
    result = {key: source.get(key) for key in keys}
    result["components"] = {
        key: {field: component.get(field) for field in ("score", "weight", "clauses", "code")}
        for key, component in source.get("components", {}).items()
    }
    return PublicScore.model_validate(result).model_dump(mode="json")


def _run_public(row: dict[str, Any]) -> dict[str, Any]:
    internal_tasks = [Task.model_validate(task) for task in json.loads(row["tasks_json"])]
    stored_metadata = json.loads(row["versions_json"])
    reports = json.loads(row["reports_json"])
    tier_mix = stored_metadata.get("tier_mix") or {
        str(tier): sum(task.difficulty == tier for task in internal_tasks) for tier in (1, 2, 3)
    }
    public = {
        "run_id": row["id"], "status": row["state"],
        "versions": {key: stored_metadata.get(key) for key in ("ruleset", "reward", "schema")},
        "tier_mix": tier_mix,
        "seed_fingerprint": hashlib.sha256(row["seed"].encode()).hexdigest(),
        "scores": reports,
        "retryable": row["state"] in ("pending", "scoring"),
        "transcript": {
            "tasks": [public_task(task, mode="single") for task in internal_tasks],
            "submitted_answers": json.loads(row["answers_json"]) if row["answers_json"] else [],
            "scores": reports,
        },
    }
    return PublicRun.model_validate(public).model_dump(mode="json")


class QuotaJudge:
    def __init__(self, judge: Any, store: RunStore, caller: str) -> None:
        self.judge, self.store, self.caller = judge, store, caller

    async def grade(self, task: Task, answer: Any) -> Any:
        self.store.reserve_judge(
            self.caller, int(os.getenv("ASTER_CALLER_JUDGE_LIMIT", "100")),
            int(os.getenv("ASTER_GLOBAL_JUDGE_LIMIT", "500")),
        )
        return await self.judge.grade(task, answer)


def create_app(database_url: str | None = None, judge: Any = _DEFAULT_JUDGE) -> FastAPI:
    resolved_url = database_url or os.environ.get("DATABASE_URL") or "sqlite:///data/sandbox.sqlite"
    store = RunStore(resolved_url)
    selected_judge = make_judge() if judge is _DEFAULT_JUDGE else judge

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            try:
                if selected_judge is not None and hasattr(selected_judge, "aclose"):
                    await selected_judge.aclose()
            finally:
                store.engine.dispose()

    app = FastAPI(title="Aster Payroll Gym", version="0.1.0", lifespan=lifespan)
    app.state.store, app.state.judge = store, selected_judge

    @app.middleware("http")
    async def request_bounds(request: Request, call_next: Any) -> Any:
        if request.method == "POST":
            limit = 64 * 1024
            try:
                chunks: list[bytes] = []
                length = 0
                async for chunk in request.stream():
                    length += len(chunk)
                    if length > limit:
                        return JSONResponse({"error": "REQUEST_TOO_LARGE"}, status_code=413)
                    chunks.append(chunk)
                request._body = b"".join(chunks)
            except Exception:
                return JSONResponse({"error": "REQUEST_READ_FAILED"}, status_code=400)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(StoreError)
    async def store_error(_: Request, exc: StoreError) -> JSONResponse:
        return JSONResponse({"error": exc.code}, status_code=exc.status)

    @app.exception_handler(Exception)
    async def safe_error(_: Request, exc: Exception) -> JSONResponse:
        logger.error(json.dumps({"event": "api_error", "type": type(exc).__name__}))
        return JSONResponse({"error": "INTERNAL_ERROR"}, status_code=500)

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        if not store.healthy():
            raise HTTPException(503, "STORAGE_UNAVAILABLE")
        return {"status": "ok", "ruleset_version": RULESET_VERSION, "reward_version": REWARD_VERSION}

    @app.get("/tasks", response_model=PublicIssuedRun)
    def tasks(
        request: Request, tier: str = Query("all", pattern="^(all|1|2|3)$"),
        n: int = Query(3, ge=1, le=10),
    ) -> dict[str, Any]:
        # Trust only the direct socket identity. Proxy headers require explicit server configuration.
        caller = digest(os.getenv("ASTER_RATE_SALT", "") + ":" + (
            request.client.host if request.client else "unknown"
        ))
        seed = secrets.randbits(63)
        generated = generate_taskset(n=n, seed=seed, tier=tier, split="sandbox")
        versions = {"ruleset": RULESET_VERSION, "reward": REWARD_VERSION, "schema": SCHEMA_VERSION}
        tier_mix = {str(t): sum(task.difficulty == t for task in generated) for t in (1, 2, 3)}
        run_id, token = store.create_run(
            [task.model_dump(mode="json") for task in generated], seed,
            {**versions, "tier_mix": tier_mix}, caller,
        )
        issued = {
            "run_id": run_id, "run_token": token, "seed_fingerprint": digest(str(seed)),
            "versions": versions, "tier_mix": tier_mix,
            "tasks": [public_task(task, mode="single") for task in generated],
        }
        return PublicIssuedRun.model_validate(issued).model_dump(mode="json")

    @app.post("/submit", response_model=PublicRun)
    async def submit(payload: Submission, x_run_token: str = Header("", max_length=128)) -> Any:
        row = store.get_run(payload.run_id, x_run_token)
        tasks_internal = [Task.model_validate(task) for task in json.loads(row["tasks_json"])]
        expected_ids = {task.id for task in tasks_internal}
        provided_ids = [answer.task_id for answer in payload.answers]
        if len(set(provided_ids)) != len(provided_ids) or set(provided_ids) != expected_ids:
            raise HTTPException(422, "Submit exactly one answer for every issued task")
        answers = sorted([answer.model_dump(mode="json") for answer in payload.answers], key=lambda x: x["task_id"])
        try:
            row, claimed = store.claim_submission(payload.run_id, x_run_token, answers)
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, "INVALID_JSON_VALUE") from exc
        if not claimed:
            response = _run_public(row)
            return JSONResponse(response, status_code=200 if row["state"] == "complete" else 202)
        raw_by_id = {answer["task_id"]: answer["answer"] for answer in answers}
        previous = {item["task_id"]: item for item in json.loads(row["reports_json"])}
        judge_for_run = QuotaJudge(selected_judge, store, row["caller_hash"]) if selected_judge else None
        reports: list[dict[str, Any]] = []
        for task in tasks_internal:
            if previous.get(task.id, {}).get("status") == "complete":
                reports.append(previous[task.id])
                continue
            try:
                report = await asyncio.wait_for(score(task, raw_by_id[task.id], judge=judge_for_run), timeout=30)
                reports.append(_report_public(report))
            except Exception as exc:
                logger.error(json.dumps({"event": "score_error", "run_id": row["id"], "type": type(exc).__name__}))
                reports.append({
                    "task_id": task.id, "status": "pending", "score": None, "components": {},
                    "gate": None, "error_code": "SCORING_UNAVAILABLE",
                    "ruleset_version": RULESET_VERSION, "reward_version": REWARD_VERSION,
                })
            # Persist progress after every task; retry resumes only pending work.
            store.save_reports(row["id"], reports, complete=False, progress=True)
        complete = all(report["status"] == "complete" for report in reports)
        store.save_reports(row["id"], reports, complete=complete)
        return JSONResponse(_run_public(store.get_run(row["id"], x_run_token)), status_code=200 if complete else 202)

    @app.get("/runs/{run_id}", response_model=PublicRun)
    def get_run(run_id: str, x_run_token: str = Header("", max_length=128)) -> dict[str, Any]:
        return _run_public(store.get_run(run_id, x_run_token))

    return app


app = create_app()
