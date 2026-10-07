"""Synthetic ephemeral fixtures check analyses, never become submission evidence."""

import json

import pytest

from aster_gym.evidence import (
    analyze_results,
    freeze_transfer,
    judge_reliability,
    rank_agreement,
    verify_transfer_freeze,
)
from aster_gym.generator import read_taskset


def test_judge_agreement_and_uncached_repeat_stability():
    rows = [{"human_label": 2, "reviewer": "test reviewer", "ratings": [
        {"status": "complete", "label": label} for label in (2, 2, 1)]},
        {"human_label": 0, "reviewer": "test reviewer", "ratings": [
            {"status": "complete", "label": 0} for _ in range(3)]}]
    result = judge_reliability(rows)
    assert result["human_exact_agreement"] == pytest.approx(5 / 6)
    assert result["three_repeat_disagreement_fraction"] == 0.5
    assert result["status"] == "partial_or_pending"
    assert judge_reliability([])["human_exact_agreement"] is None
    with pytest.raises(ValueError):
        judge_reliability([{"human_label": True, "reviewer": "tester"}])


def test_transfer_ranks_handle_ties_and_small_sample():
    result = rank_agreement({"a": .9, "b": .7, "c": .1}, {"a": .1, "b": .7, "c": .9})
    assert result["spearman_rho"] == -1
    assert len(result["pairwise_ranking_reversals"]) == 3
    tied = rank_agreement({"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 0, "c": 1})
    assert tied["spearman_rho"] is None
    assert tied["generated_ranks"] == {"a": 2, "b": 2, "c": 2}


def test_transfer_requires_human_review_and_exact_frozen_inputs(tmp_path):
    tasks = read_taskset("data/transfer.jsonl")
    packet = json.loads(open("reviews/task-review-packet.json").read())
    source = tmp_path / "review.json"
    output = tmp_path / "freeze.json"
    source.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="HUMAN_REVIEW_PENDING"):
        freeze_transfer(tasks, source, output)
    for row in packet["transfer_reviews"]:
        row.update(reviewed=True, reviewer="ephemeral test reviewer", human_result=row["reference_result"])
    source.write_text(json.dumps(packet))
    freeze_transfer(tasks, source, output)
    marker = verify_transfer_freeze(tasks, output)
    assert len(marker["task_ids"]) == 5
    assert freeze_transfer(tasks, source, output) == output
    tasks[0] = tasks[0].model_copy(update={"prompt": "Changed after freeze"})
    with pytest.raises(ValueError, match="FREEZE_MISMATCH"):
        verify_transfer_freeze(tasks, output)


def test_empty_analysis_has_explicit_pending_states(tmp_path):
    path = analyze_results(tmp_path / "results", tmp_path / "reviews")
    value = json.loads(path.read_text())
    assert value["transfer"]["status"] == "pending"
    assert value["judge_reliability"]["human_exact_agreement"] is None
    assert value["human_failure_review"]["reviewed_failures"] == 0


def test_analysis_joins_blind_labels_to_matching_measured_ratings(tmp_path):
    reviews, results = tmp_path / "reviews", tmp_path / "results"
    reviews.mkdir()
    results.mkdir()
    packet = json.loads(open("reviews/judge-review-packet.json").read())
    for row in packet["examples"]:
        row.update(human_label=2, reviewer="ephemeral test reviewer")
    (reviews / "judge-review-packet.json").write_text(json.dumps(packet))
    measurement = json.loads(json.dumps(packet))
    for row in measurement["examples"]:
        row.update(human_label=None, reviewer=None)
        row["ratings"] = [{"repeat": repeat, "status": "complete", "label": 2} for repeat in range(3)]
    measurement["examples"][0]["ratings"][0].update(
        status="pending", label=None, error_code="JUDGE_INVALID_OUTPUT")
    source = results / "judge-stability-study.json"
    source.write_text(json.dumps(measurement))
    value = json.loads(analyze_results(results, reviews).read_text())["judge_reliability"]
    assert value["status"] == "complete"
    assert value["human_comparisons"] == 44
    assert value["human_exact_agreement"] == 1
    assert value["invalid_output_fraction"] == pytest.approx(1 / 45)
    assert value["three_repeat_outcome_disagreement_fraction"] == pytest.approx(1 / 15)
    measurement["examples"][0]["candidate"]["explanation"] = "Different candidate"
    source.write_text(json.dumps(measurement))
    with pytest.raises(ValueError, match="MEASUREMENT_MISMATCH"):
        analyze_results(results, reviews)


@pytest.mark.parametrize("changed_split", ["evaluation", "transfer"])
def test_transfer_rankings_require_matching_prompts_within_each_cohort(tmp_path, monkeypatch, changed_split):
    """A scaffold change must not masquerade as measured transfer of a model ranking."""
    runs = []
    for split, count in (("evaluation", 30), ("transfer", 5)):
        for model, score in (("a", .9), ("b", .7), ("c", .1)):
            ids = [f"{split}-{index}" for index in range(count)]
            runs.append({"config": {"evidence_kind": "model_run", "status": "complete", "mode": "single",
                "split": split, "task_count": count, "taskset_hash": split, "model": model,
                "run_id": f"{split}-{model}", "prompt_hashes": dict.fromkeys(ids, "original")},
                "summary": {"complete": count * 3, "total": count * 3,
                            "replicate_means": [score] * 3, "mean": score},
                "records": [{"task_id": task, "score": score} for task in ids for _ in range(3)]})
    monkeypatch.setattr("aster_gym.evidence.collect_results", lambda _: {"runs": runs})
    output = analyze_results(tmp_path / "results", tmp_path / "reviews")
    assert json.loads(output.read_text())["transfer"][0]["spearman_rho"] == 1
    changed = next(run for run in runs if run["config"]["split"] == changed_split)
    changed["config"]["prompt_hashes"][f"{changed_split}-0"] = "changed"
    output = analyze_results(tmp_path / "results", tmp_path / "reviews")
    assert json.loads(output.read_text())["transfer"]["status"] == "pending"
