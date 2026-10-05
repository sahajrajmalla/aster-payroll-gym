"""Small synthetic API fixtures; these are not model evaluation evidence."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from aster_gym.api import create_app
from aster_gym.judge import RemoteJudge
from aster_gym.store import RunStore, StoreError


@pytest.fixture
def client(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'sandbox.sqlite'}", judge=None)
    with TestClient(app) as instance:
        yield instance


def issued(client, tier="2", n=1):
    response = client.get(f"/tasks?tier={tier}&n={n}")
    assert response.status_code == 200
    return response.json()


def headers(run):
    return {"X-Run-Token": run["run_token"]}


def payload(run, answer="{}"):
    return {"run_id": run["run_id"], "answers": [
        {"task_id": task["id"], "answer": answer} for task in run["tasks"]]}


def keys_recursive(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from keys_recursive(child)
    elif isinstance(value, list):
        for child in value:
            yield from keys_recursive(child)


def test_public_tasks_and_roundtrip_never_serialize_oracle(client):
    run = issued(client, n=3)
    forbidden = {"ground_truth", "expected", "inputs", "seed", "difficulty", "tags", "token_hash", "tasks_json"}
    assert not forbidden.intersection(keys_recursive(run))
    assert len(run["seed_fingerprint"]) == 64
    result = client.post("/submit", json=payload(run), headers=headers(run))
    assert result.status_code == 200
    assert all(item["score"] == 0 for item in result.json()["scores"])
    assert not forbidden.intersection(keys_recursive(result.json()))
    assert client.get("/runs/" + run["run_id"], headers=headers(run)).json() == result.json()
    assert client.get("/healthz").json()["status"] == "ok"


def test_exact_answer_uses_shared_scorer(client):
    run = issued(client)
    internal = client.app.state.store.get_run(run["run_id"], run["run_token"])
    task = json.loads(internal["tasks_json"])[0]
    answer = task["ground_truth"]  # Test fixture only; API caller cannot access this.
    result = client.post("/submit", json=payload(run, answer), headers=headers(run))
    assert result.status_code == 200
    assert result.json()["scores"][0]["score"] == 1
    assert "expected" not in result.text
    transcript = result.json()["transcript"]
    assert transcript["tasks"] == run["tasks"]
    assert transcript["submitted_answers"] == payload(run, answer)["answers"]
    # Caller amounts may coincide with expected amounts; only the submitted copy is echoed.
    assert transcript["submitted_answers"][0]["answer"]["result"] == answer["result"]
    assert not {"ground_truth", "inputs", "seed", "tags"}.intersection(keys_recursive(transcript))
    assert result.json()["tier_mix"] == {"1": 0, "2": 1, "3": 0}


def test_idempotent_retries_and_conflicting_submission(client):
    run = issued(client)
    one = client.post("/submit", json=payload(run), headers=headers(run))
    two = client.post("/submit", json=payload(run), headers=headers(run))
    assert one.json() == two.json()
    conflict = client.post("/submit", json=payload(run, "[]"), headers=headers(run))
    assert conflict.status_code == 409
    assert conflict.json() == {"error": "SUBMISSION_CONFLICT"}


def test_pending_judge_is_not_zero_or_simplified_rubric(client):
    run = issued(client, tier="3")
    task = json.loads(client.app.state.store.get_run(run["run_id"], run["run_token"])["tasks_json"])[0]
    answer = task["ground_truth"]
    result = client.post("/submit", json=payload(run, answer), headers=headers(run))
    assert result.status_code == 202
    assert result.json()["scores"][0]["score"] is None
    assert result.json()["retryable"]
    assert client.post("/submit", json=payload(run, answer), headers=headers(run)).status_code == 202


def test_membership_unknown_token_body_and_task_limits(client):
    assert client.get("/tasks?n=11").status_code == 422
    assert client.get("/tasks?tier=4").status_code == 422
    run = issued(client, n=2)
    assert client.get("/runs/" + run["run_id"]).status_code == 404
    assert client.get("/runs/unknown", headers=headers(run)).status_code == 404
    missing = payload(run)
    missing["answers"].pop()
    assert client.post("/submit", json=missing, headers=headers(run)).status_code == 422
    duplicate = payload(run)
    duplicate["answers"][1] = duplicate["answers"][0]
    assert client.post("/submit", json=duplicate, headers=headers(run)).status_code == 422
    response = client.post("/submit", content=b"x" * 65537, headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_creation_limit_persistent(client):
    for i in range(5):
        assert client.get("/tasks", headers={"X-Forwarded-For": f"192.0.2.{i}"}).status_code == 200
    assert client.get("/tasks").status_code == 429


def test_transactional_claim_one_concurrent_owner(tmp_path):
    store = RunStore(f"sqlite:///{tmp_path / 'race.sqlite'}")
    run_id, token = store.create_run([], 777, {}, "test")
    answers = [{"task_id": "x", "answer": "{}"}]
    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(lambda _: store.claim_submission(run_id, token, answers)[1], range(4)))
    assert sum(claims) == 1
    row = store.get_run(run_id, token)
    assert row["seed"] == "777"
    store.save_reports(run_id, [{"task_id": "x", "status": "complete", "score": 0}], complete=True)
    reopened = RunStore(f"sqlite:///{tmp_path / 'race.sqlite'}")
    assert reopened.get_run(run_id, token)["state"] == "complete"


def test_atomic_judge_quotas(tmp_path):
    store = RunStore(f"sqlite:///{tmp_path / 'quota.sqlite'}")
    store.reserve_judge("one", caller_limit=1, global_limit=2)
    with pytest.raises(StoreError, match="JUDGE_QUOTA"):
        store.reserve_judge("one", caller_limit=1, global_limit=2)
    store.reserve_judge("two", caller_limit=1, global_limit=2)
    with pytest.raises(StoreError, match="JUDGE_QUOTA"):
        store.reserve_judge("three", caller_limit=1, global_limit=2)


def test_judge_and_database_close_on_lifespan_exit(tmp_path):
    class CloseableJudge:
        closed = False

        async def aclose(self):
            self.closed = True

    judge = CloseableJudge()
    app = create_app(f"sqlite:///{tmp_path / 'close.sqlite'}", judge=judge)
    with TestClient(app) as instance:
        assert instance.get("/healthz").status_code == 200
        assert not judge.closed
    assert judge.closed


def test_issued_tier_mix_and_transcript_persist_before_submission(client):
    run = issued(client, tier="all", n=3)
    current = client.get("/runs/" + run["run_id"], headers=headers(run)).json()
    assert current["tier_mix"] == {"1": 1, "2": 1, "3": 1}
    assert current["transcript"]["tasks"] == run["tasks"]
    assert current["transcript"]["submitted_answers"] == []
    internal = client.app.state.store.get_run(run["run_id"], run["run_token"])
    assert json.loads(internal["versions_json"])["tier_mix"] == run["tier_mix"]


@pytest.mark.parametrize("failure", [False, True])
def test_health_storage_outage_is_safe_and_retryable(client, monkeypatch, failure):
    def unavailable():
        if failure:
            raise OperationalError("private database hostname", {}, Exception("private connection detail"))
        return False

    monkeypatch.setattr(client.app.state.store, "healthy", unavailable)
    response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json() == {"detail": "STORAGE_UNAVAILABLE"}
    assert "private" not in response.text


@pytest.mark.parametrize("tier", ["2", "3"])
def test_judge_configuration_changes_freeze_blocker_scoring_only(tmp_path, tier):
    database = f"sqlite:///{tmp_path / 'judge-settings.sqlite'}"
    with TestClient(create_app(database, judge=None)) as first:
        run = issued(first, tier=tier)
        private = first.app.state.store.get_run(run["run_id"], run["run_token"])
        answer = json.loads(private["tasks_json"])[0]["ground_truth"]
        assert json.loads(private["versions_json"])["judge"] is None
        assert "judge" not in run
    configured = RemoteJudge(model="fixture", base_url="https://example.invalid", api_key="fixture-secret")
    with TestClient(create_app(database, judge=configured)) as second:
        response = second.post("/submit", json=payload(run, answer), headers=headers(run))
        if tier == "3":
            assert response.status_code == 409
            assert response.json()["detail"].startswith("JUDGE_CONFIGURATION_CHANGED")
            assert configured.calls == 0
        else:
            assert response.status_code == 200 and response.json()["scores"][0]["score"] == 1
        assert "fixture-secret" not in response.text
