"""Transactional, lightweight run storage; no model dependencies."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.engine import Connection


class StoreError(Exception):
    """Safe API error whose message never contains internal task data."""

    def __init__(self, code: str, status: int = 409):
        self.code, self.status = code, status
        super().__init__(code)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class RunStore:
    def __init__(self, database_url: str = "sqlite:///data/sandbox.sqlite") -> None:
        if database_url.startswith("postgres://"):
            database_url = "postgresql+psycopg://" + database_url[len("postgres://"):]
        elif database_url.startswith("postgresql://"):
            database_url = "postgresql+psycopg://" + database_url[len("postgresql://"):]
        self.sqlite = database_url.startswith("sqlite")
        if self.sqlite and "///" in database_url:
            database_file = database_url.split("///", 1)[1].split("?", 1)[0]
            if database_file and database_file != ":memory:":
                Path(database_file).parent.mkdir(parents=True, exist_ok=True)
        options: dict[str, Any] = {"pool_pre_ping": True}
        if self.sqlite:
            options["connect_args"] = {"check_same_thread": False, "timeout": 15}
            if database_url.endswith(":memory:"):
                from sqlalchemy.pool import StaticPool
                options["poolclass"] = StaticPool
        self.engine = create_engine(database_url, **options)
        metadata = MetaData()
        self.runs = Table(
            "aster_runs", metadata,
            Column("id", String(64), primary_key=True),
            Column("token_hash", String(64), nullable=False),
            Column("caller_hash", String(64), nullable=False),
            Column("seed", Text, nullable=False),
            Column("tasks_json", Text, nullable=False),
            Column("versions_json", Text, nullable=False),
            Column("created", Float, nullable=False),
            Column("state", String(16), nullable=False),
            Column("answers_hash", String(64)),
            Column("answers_json", Text),
            Column("reports_json", Text, nullable=False, default="[]"),
            Column("lease_until", Float, nullable=False, default=0),
        )
        self.events = Table(
            "aster_quota_events", metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("kind", String(16), nullable=False),
            Column("caller", String(64), nullable=False),
            Column("created", Float, nullable=False, index=True),
        )
        self.controls = Table("aster_controls", metadata, Column("id", String(32), primary_key=True))
        metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            # Database-native upsert avoids a startup race across server workers.
            if self.sqlite:
                connection.execute(text("INSERT OR IGNORE INTO aster_controls (id) VALUES ('quotas')"))
            else:
                connection.execute(text("INSERT INTO aster_controls (id) VALUES ('quotas') ON CONFLICT DO NOTHING"))

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        with self.engine.connect() as connection:
            if self.sqlite:
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                connection.begin()
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _lock_quotas(self, connection: Connection) -> None:
        statement = select(self.controls).where(self.controls.c.id == "quotas")
        if not self.sqlite:
            statement = statement.with_for_update()
        connection.execute(statement).first()

    def _quota(self, connection: Connection, kind: str, caller: str, since: float, limit: int) -> None:
        count = connection.scalar(select(func.count()).select_from(self.events).where(
            self.events.c.kind == kind, self.events.c.created >= since,
            *([] if caller == "*" else [self.events.c.caller == caller]),
        ))
        if int(count or 0) >= limit:
            raise StoreError("RATE_LIMIT" if kind == "creation" else "JUDGE_QUOTA", 429)

    def create_run(
        self, tasks: list[dict[str, Any]], seed: int, versions: dict[str, Any],
        caller: str, creation_limit: int = 5,
    ) -> tuple[str, str]:
        now = time.time()
        run_id, token = secrets.token_urlsafe(18), secrets.token_urlsafe(32)
        with self.transaction() as connection:
            self._lock_quotas(connection)
            self._quota(connection, "creation", caller, now - 60, creation_limit)
            connection.execute(delete(self.events).where(self.events.c.created < now - 86400))
            connection.execute(insert(self.events).values(kind="creation", caller=caller, created=now))
            connection.execute(insert(self.runs).values(
                id=run_id, token_hash=digest(token), caller_hash=caller, seed=str(seed),
                tasks_json=json.dumps(tasks), versions_json=json.dumps(versions), created=now,
                state="issued", reports_json="[]", lease_until=0,
            ))
        return run_id, token

    def _get(self, connection: Connection, run_id: str, token: str, lock: bool = False) -> dict[str, Any]:
        statement = select(self.runs).where(self.runs.c.id == run_id)
        if lock and not self.sqlite:
            statement = statement.with_for_update()
        row = connection.execute(statement).mappings().first()
        if row is None or not hmac.compare_digest(row["token_hash"], digest(token)):
            raise StoreError("RUN_NOT_FOUND", 404)
        return dict(row)

    def get_run(self, run_id: str, token: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            return self._get(connection, run_id, token)

    def claim_submission(
        self, run_id: str, token: str, answers: list[dict[str, Any]], lease_seconds: int = 180,
    ) -> tuple[dict[str, Any], bool]:
        canonical = json.dumps(answers, sort_keys=True, separators=(",", ":"), allow_nan=False)
        answer_hash = digest(canonical)
        with self.transaction() as connection:
            row = self._get(connection, run_id, token, lock=True)
            if row["answers_hash"] is not None and row["answers_hash"] != answer_hash:
                raise StoreError("SUBMISSION_CONFLICT")
            if row["state"] == "complete" or (
                row["state"] == "scoring" and row["lease_until"] > time.time()
            ):
                return row, False
            connection.execute(update(self.runs).where(self.runs.c.id == run_id).values(
                answers_hash=answer_hash, answers_json=canonical, state="scoring",
                lease_until=time.time() + lease_seconds,
            ))
            row.update(answers_hash=answer_hash, answers_json=canonical, state="scoring")
            return row, True

    def save_reports(
        self, run_id: str, reports: list[dict[str, Any]], complete: bool, progress: bool = False,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(update(self.runs).where(self.runs.c.id == run_id).values(
                reports_json=json.dumps(reports, allow_nan=False),
                state="scoring" if progress else ("complete" if complete else "pending"),
                lease_until=time.time() + 180 if progress else 0,
            ))

    def reserve_judge(self, caller: str, caller_limit: int = 100, global_limit: int = 500) -> None:
        now = time.time()
        with self.transaction() as connection:
            self._lock_quotas(connection)
            self._quota(connection, "judge", caller, now - 86400, caller_limit)
            self._quota(connection, "judge", "*", now - 86400, global_limit)
            connection.execute(insert(self.events).values(kind="judge", caller=caller, created=now))

    def healthy(self) -> bool:
        with self.engine.connect() as connection:
            return connection.scalar(select(func.count()).select_from(self.controls)) == 1
