"""Saved-artifact fixtures validate display only, never evidence of model quality."""

import json
import re
import shutil
import subprocess

import pytest

from aster_gym.reporting import build_dashboard, collect_results, summarize_records


def test_statistics_use_rollout_mean_sd_and_keep_operational_failures():
    records = [
        {"task_id": f"t{task}", "rollout": rollout, "tier": task + 1, "status": "complete",
         "score": score, "components": {"exact": {"score": score}}, "cost_usd": 0, "latency_s": 1}
        for rollout, score in enumerate([0, 0.5, 1]) for task in range(2)
    ]
    summary = summarize_records(records)
    assert summary["mean"] == 0.5
    assert summary["sd"] == 0.5
    assert summary["complete"] == 6
    assert summary["latency_mean_s"] == 1
    assert summary["components"]["exact"] == 0.5
    records.append({"task_id": "failure", "status": "provider_error", "score": None})
    assert summarize_records(records)["operational_failures"] == 1


def test_pending_dashboard_has_no_fake_curve_data(tmp_path):
    destination = build_dashboard(tmp_path / "missing", tmp_path / "site")
    page = destination.read_text()
    assert "Colab execution pending" in page
    data = json.loads((destination.parent / "dashboard-data.json").read_text())
    assert data["runs"] == []
    assert "No leaderboard scores have been invented" in page


def test_fixtures_excluded_and_transcript_markup_not_executable(tmp_path):
    run = tmp_path / "results" / "fixture"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"model": "fixture", "evidence_kind": "fixture"}))
    (run / "scores.json").write_text(json.dumps({"records": [{"status": "complete", "score": 1}]}))
    attack = "</script><script>alert('attack')</script>"
    (run / "transcript.jsonl").write_text(json.dumps({"messages": [{"content": attack}]}) + "\n")
    data = collect_results(tmp_path / "results")
    assert not data["runs"][0]["eligible"]
    page = build_dashboard(tmp_path / "results", tmp_path / "site").read_text()
    assert attack not in page
    assert "textContent" in page


def test_corrupt_run_is_disclosed(tmp_path):
    run = tmp_path / "results" / "broken"
    run.mkdir(parents=True)
    (run / "config.json").write_text("broken")
    data = collect_results(tmp_path / "results")
    assert data["warnings"] and data["runs"] == []


def write_evaluation(directory, records):
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"model": "fixture", "evidence_kind": "model_run",
        "task_ids": ["a", "b"], "task_count": 2, "rollouts": 3, "status": "partial"}))
    (directory / "scores.json").write_text(json.dumps({"records": records}))


def test_truncated_artifact_uses_frozen_coverage_and_stays_unranked(tmp_path):
    row = {"task_id": "a", "rollout": 0, "tier": 1, "status": "complete", "score": 1}
    write_evaluation(tmp_path / "run", [row])
    run = collect_results(tmp_path)["runs"][0]
    assert run["eligible"]
    assert run["summary"]["total"] == 6
    assert run["summary"]["coverage"] == pytest.approx(1 / 6)
    assert run["summary"]["mean"] is None
    assert not run["summary"]["rankable"]


@pytest.mark.parametrize("bad", [2, float("nan"), True])
def test_invalid_saved_score_cannot_enter_dashboard(tmp_path, bad):
    write_evaluation(tmp_path / "run", [{"task_id": "a", "rollout": 0, "status": "complete", "score": bad}])
    data = collect_results(tmp_path)
    assert not data["runs"] and data["warnings"]


def test_missing_manifest_and_training_rows_cannot_rank(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"evidence_kind": "model_run"}))
    (run / "scores.json").write_text(json.dumps({"records": [{"status": "complete", "score": 1}]}))
    data = collect_results(tmp_path)
    assert data["warnings"] and not data["runs"][0]["summary"]["rankable"]
    (run / "config.json").write_text(json.dumps({"evidence_kind": "model_run", "run_type": "training"}))
    assert not collect_results(tmp_path)["runs"][0]["eligible"]


def test_durable_accounting_is_separate_from_confirmed_spend(tmp_path):
    write_evaluation(tmp_path / "run", [])
    (tmp_path / "run" / "budget.json").write_text(json.dumps({"accounted_usd": .2,
        "confirmed_usd": .05, "reserved_usd": .1}))
    summary = collect_results(tmp_path)["runs"][0]["summary"]
    assert summary["cost_usd"] == pytest.approx(.3)
    assert summary["confirmed_cost_usd"] == .05


def test_dashboard_script_parses_when_node_available(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional for a lightweight dashboard JavaScript syntax check")
    page = build_dashboard(tmp_path / "missing", tmp_path / "site").read_text()
    for script in re.findall(r"<script>([\s\S]*?)</script>", page):
        result = subprocess.run([node, "--check", "-"], input=script, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("changed", [
    {"judge": None}, {"judge": {"model": "judge", "prompt_hash": "changed"}},
    {"temperature": .8}, {"max_tokens": 128}, {"seed": 7002},
    {"rules_hash": "changed"}, {"reward_version": "2.0"},
])
def test_incompatible_runs_cannot_share_a_dashboard_ranking(tmp_path, changed):
    """Same tasks and scores do not make different judge/sampling contracts comparable."""
    base = {"taskset_hash": "frozen", "mode": "single", "rules_hash": "rules",
            "reward_version": "1.0", "judge": {"model": "judge", "prompt_hash": "original"},
            "temperature": .7, "max_tokens": 256, "seed": 7001}
    rows = [{"task_id": task, "rollout": rollout, "tier": 1,
             "status": "complete", "score": .5} for task in ("a", "b") for rollout in range(3)]
    for name, settings in (("a", base), ("b", base), ("c", {**base, **changed})):
        directory = tmp_path / "results" / name
        write_evaluation(directory, rows)
        path = directory / "config.json"
        config = json.loads(path.read_text())
        config.update(settings, model=name, status="complete")
        path.write_text(json.dumps(config))
    runs = collect_results(tmp_path / "results")["runs"]
    keys = {run["config"]["model"]: run["comparison_key"] for run in runs}
    assert keys["a"] == keys["b"] and keys["a"] != keys["c"]
    page = build_dashboard(tmp_path / "results", tmp_path / "site").read_text()
    assert "const key=r.comparison_key" in page
