"""Lightweight analyses and human-review gates; no model library or inference."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .reporting import collect_results
from .schemas import Task
from .versions import REWARD_VERSION, RULESET_VERSION, rules_hash, stable_hash


def freeze_transfer(tasks: list[Task], packet_path: str | Path,
                    output: str | Path = "reviews/transfer-freeze.json") -> Path:
    """Require signed review rows, then freeze the exact tasks before outcomes."""
    packet = json.loads(Path(packet_path).read_text())
    rows = packet.get("transfer_reviews", [])
    by_id = {row["task_id"]: row for row in rows}
    if len(tasks) != 5 or len(by_id) != 5 or set(by_id) != {t.id for t in tasks}:
        raise ValueError("TRANSFER_REVIEW_MEMBERSHIP")
    for task in tasks:
        row = by_id[task.id]
        if (row.get("reviewed") is not True or not str(row.get("reviewer") or "").strip()
                or row.get("human_result") is None or row.get("inputs") != task.inputs
                or row.get("prompt") != task.prompt):
            raise ValueError("TRANSFER_HUMAN_REVIEW_PENDING_OR_STALE")
    marker = {"status": "human_approved_frozen", "frozen_at": datetime.now(UTC).isoformat(),
              "taskset_hash": stable_hash([t.model_dump(mode="json") for t in tasks]),
              "task_ids": [t.id for t in tasks], "rules_hash": rules_hash(),
              "ruleset_version": RULESET_VERSION, "reward_version": REWARD_VERSION,
              "review_hash": stable_hash(rows),
              "reviewers": sorted({str(r["reviewer"]) for r in rows}),
              "presentation_origin": "Synthetic drafts approved by the named reviewers"}
    destination = Path(output)
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing.get("taskset_hash") != marker["taskset_hash"]:
            raise ValueError("TRANSFER_ALREADY_FROZEN_DIFFERENT_TASKS")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(marker, indent=2) + "\n")
    return destination


def verify_transfer_freeze(tasks: list[Task],
                           path: str | Path = "reviews/transfer-freeze.json") -> dict[str, Any]:
    marker_path = Path(path)
    if not marker_path.exists():
        raise ValueError("TRANSFER_HUMAN_APPROVAL_AND_FREEZE_REQUIRED")
    marker = json.loads(marker_path.read_text())
    expected = stable_hash([t.model_dump(mode="json") for t in tasks])
    if (marker.get("status") != "human_approved_frozen" or marker.get("taskset_hash") != expected
            or marker.get("rules_hash") != rules_hash()
            or marker.get("reward_version") != REWARD_VERSION
            or marker.get("ruleset_version") != RULESET_VERSION or not marker.get("reviewers")):
        raise ValueError("TRANSFER_FREEZE_MISMATCH")
    frozen_at = datetime.fromisoformat(marker["frozen_at"])
    if frozen_at.tzinfo is None or frozen_at > datetime.now(UTC):
        raise ValueError("TRANSFER_FREEZE_INVALID_DATE")
    return marker


def judge_reliability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact human agreement over completed ratings and three-repeat stability."""
    comparisons, consistent, completed_examples = [], 0, 0
    attempted_repeats, malformed_repeats = 0, 0
    outcome_examples, consistent_outcomes = 0, 0
    confusion: Counter[str] = Counter()
    for row in rows:
        attempted_repeats += sum(r.get("status") == "complete" or r.get("error_code") == "JUDGE_INVALID_OUTPUT"
                                 for r in row.get("ratings", []))
        malformed_repeats += sum(r.get("error_code") == "JUDGE_INVALID_OUTPUT" for r in row.get("ratings", []))
        labels = [r.get("label") for r in row.get("ratings", []) if r.get("status") == "complete"]
        if any(type(label) is not int or label not in (0, 1, 2) for label in labels):
            raise ValueError("INVALID_JUDGE_LABEL")
        if len(labels) == 3:
            completed_examples += 1
            consistent += len(set(labels)) == 1
        outcomes = [rating.get("label") if rating.get("status") == "complete" else "invalid_output"
                    for rating in row.get("ratings", []) if rating.get("status") == "complete"
                    or rating.get("error_code") == "JUDGE_INVALID_OUTPUT"]
        if len(outcomes) == 3:
            outcome_examples += 1
            consistent_outcomes += len(set(outcomes)) == 1
        human = row.get("human_label")
        if human is not None:
            if type(human) is not int or human not in (0, 1, 2) or not row.get("reviewer"):
                raise ValueError("INVALID_HUMAN_LABEL")
            for label in labels:
                comparisons.append(human == label)
                confusion[f"human_{human}_judge_{label}"] += 1
    human_reviewed = sum(row.get("human_label") is not None and bool(row.get("reviewer")) for row in rows)
    return {"status": "complete" if len(rows) >= 15 and human_reviewed == len(rows)
            and attempted_repeats == len(rows) * 3 else "partial_or_pending",
            "examples": len(rows), "completed_three_repeat_examples": completed_examples,
            "repeat_measurement_status": "complete" if len(rows) >= 15
            and attempted_repeats == len(rows) * 3 else "partial_or_pending",
            "attempted_repeats": attempted_repeats, "invalid_output_repeats": malformed_repeats,
            "invalid_output_fraction": malformed_repeats / attempted_repeats if attempted_repeats else None,
            "human_comparisons": len(comparisons),
            "human_exact_agreement": statistics.mean(comparisons) if comparisons else None,
            "three_repeat_disagreement_fraction": 1 - consistent / completed_examples
            if completed_examples else None,
            "three_repeat_outcome_disagreement_fraction": 1 - consistent_outcomes / outcome_examples
            if outcome_examples else None, "confusion_counts": dict(confusion),
            "note": "Agreement/stability condition on valid labels; malformed-output fraction is separate. "
            "Stability is not validity. Each repeat must bypass the production cache."}


def _ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda name: (-values[name], name))
    ranks: dict[str, float] = {}
    for name in ordered:
        tied = [i + 1 for i, other in enumerate(ordered) if values[other] == values[name]]
        ranks[name] = statistics.mean(tied)
    return ranks


def rank_agreement(generated: dict[str, float], transfer: dict[str, float]) -> dict[str, Any]:
    if set(generated) != set(transfer) or len(generated) < 3:
        raise ValueError("THREE_MATCHED_MODEL_CONFIGURATIONS_REQUIRED")
    a, b = _ranks(generated), _ranks(transfer)
    names = sorted(a)
    am, bm = statistics.mean(a.values()), statistics.mean(b.values())
    numerator = sum((a[n] - am) * (b[n] - bm) for n in names)
    denominator = math.sqrt(sum((a[n] - am) ** 2 for n in names) * sum((b[n] - bm) ** 2 for n in names))
    reversals = [[left, right] for i, left in enumerate(names) for right in names[i + 1:]
                 if (generated[left] - generated[right]) * (transfer[left] - transfer[right]) < 0]
    return {"generated_ranks": a, "transfer_ranks": b,
            "spearman_rho": numerator / denominator if denominator else None,
            "pairwise_ranking_reversals": reversals,
            "note": "Average ranks for ties; correlation is undefined for constant ranks. Five transfer tasks only."}


def analyze_results(results_dir: str | Path, review_dir: str | Path = "reviews") -> Path:
    root, reviews = Path(results_dir), Path(review_dir)
    data = collect_results(root)
    cohorts: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for run in data["runs"]:
        config, summary = run["config"], run["summary"]
        if (config.get("evidence_kind") != "model_run" or config.get("status") != "complete"
                or summary["complete"] != summary["total"] or config.get("mode") != "single"
                or len(summary["replicate_means"]) < 3):
            continue
        split = config.get("split")
        expected = 5 if split == "transfer" else 30
        if split not in {"evaluation", "eval", "transfer"} or config.get("task_count") != expected:
            continue
        cohort_key = stable_hash([config.get("rules_hash"), config.get("ruleset_version"),
                                  config.get("reward_version"), config.get("schema_version"), config.get("judge"),
                                  config.get("temperature"), config.get("max_tokens"),
                                  config.get("seed"), config.get("rollouts")])
        cohorts[cohort_key][split if split == "transfer" else "generated"].append(run)
    findings = []
    for grouped in cohorts.values():
        by_split: dict[str, dict[str, Any]] = {}
        ambiguous = False
        for split, runs in grouped.items():
            mapping = {r["config"]["model"]: r for r in runs}
            ambiguous |= len(mapping) != len(runs)
            by_split[split] = mapping
        if ambiguous or set(by_split) != {"generated", "transfer"}:
            continue
        matched = sorted(set(by_split["generated"]) & set(by_split["transfer"]))
        if len(matched) < 3:
            continue
        if any(len({stable_hash([by_split[s][n]["config"]["taskset_hash"],
                                by_split[s][n]["config"].get("prompt_hashes")]) for n in matched}) != 1
               for s in ("generated", "transfer")):
            continue
        generated = {n: by_split["generated"][n]["summary"]["mean"] for n in matched}
        transferred = {n: by_split["transfer"][n]["summary"]["mean"] for n in matched}
        task_scores: dict[str, dict[str, float]] = defaultdict(dict)
        for name in matched:
            for task_id in {r["task_id"] for r in by_split["transfer"][name]["records"]}:
                rows = [r["score"] for r in by_split["transfer"][name]["records"] if r["task_id"] == task_id]
                task_scores[task_id][name] = statistics.mean(rows)
        task_reversals = {task: rank_agreement(generated, scores)["pairwise_ranking_reversals"]
                          for task, scores in task_scores.items()}
        findings.append({"status": "complete", "models": matched, "generated_means": generated,
                         "transfer_means": transferred, **rank_agreement(generated, transferred),
                         "task_means": dict(task_scores), "task_level_ranking_reversals": task_reversals,
                         "runs": {s: {n: by_split[s][n]["config"]["run_id"] for n in matched}
                                  for s in ("generated", "transfer")}})
    judge_path = reviews / "judge-review-packet.json"
    if judge_path.exists():
        blind = json.loads(judge_path.read_text())
        judged_rows = blind.get("examples", [])
        measurement_path = root / "judge-stability-study.json"
        if measurement_path.exists():
            measured = json.loads(measurement_path.read_text())
            identity_keys = ("example_id", "task_id", "task_input_hash", "candidate")
            blind_ids = [{key: row.get(key) for key in identity_keys} for row in judged_rows]
            measured_ids = [{key: row.get(key) for key in identity_keys} for row in measured.get("examples", [])]
            if (stable_hash(blind_ids) != stable_hash(measured_ids)
                    or any(blind.get(key) != measured.get(key)
                           for key in ("rules_hash", "reward_version", "ruleset_version"))):
                raise ValueError("JUDGE_BLIND_REVIEW_MEASUREMENT_MISMATCH")
            judged_rows = [{**human, "ratings": rating.get("ratings", [])}
                           for human, rating in zip(judged_rows, measured["examples"], strict=True)]
        judge = judge_reliability(judged_rows)
    else:
        judge = {"status": "pending", "human_exact_agreement": None}
    failures_path = reviews / "failure-review-packet.json"
    failure_rows = json.loads(failures_path.read_text()).get("failures", []) if failures_path.exists() else []
    reviewed = [r for r in failure_rows if r.get("reviewed") is True and r.get("reviewer")]
    output = {"transfer": findings or {"status": "pending", "reason":
              "Needs three complete, matched 30-task generated and five-task frozen transfer model runs."},
              "judge_reliability": judge, "human_failure_review": {"reviewed_failures": len(reviewed),
              "category_counts": dict(Counter(r.get("category", "unclassified") for r in reviewed))}}
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "analysis.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return destination
