"""Resumable bounded evaluation; no local model implementation lives here."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import logging
import math
import os
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any

from aster_gym.generator import generate_taskset, task_prompt
from aster_gym.providers import AsyncOpenAIProvider, CostBudget, Pricing, ProviderError
from aster_gym.schemas import Task
from aster_gym.scoring import score
from aster_gym.tools import TOOL_DEFINITIONS, ToolSession, parse_tool_arguments
from aster_gym.versions import REWARD_VERSION, RULESET_VERSION, SCHEMA_VERSION, rules_hash, stable_hash

logger = logging.getLogger(__name__)


def _code_identity() -> dict[str, str]:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                           text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "uncommitted-workspace"
    root = Path(__file__).parent
    digest = stable_hash({str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*.py"))})
    return {"code_revision": revision, "implementation_hash": digest}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def summarize_records(records: list[dict[str, Any]], expected_tasks: int | None = None,
                      rollouts: int | None = None) -> dict[str, Any]:
    """Mean ± sample SD over complete replicate means; coverage is separate."""
    complete = [r for r in records if r.get("status") == "complete" and r.get("score") is not None]
    tasks = expected_tasks if expected_tasks is not None else len({r["task_id"] for r in records})
    rollout_count = rollouts if rollouts is not None else len({r["rollout"] for r in records})
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        groups[row["rollout"]].append(row)
    means = [statistics.mean(r["score"] for r in group) for group in groups.values() if len(group) == tasks]

    def summary(values: list[float]) -> dict[str, Any]:
        return {"n": len(values), "mean": statistics.mean(values) if values else None,
                "sd": statistics.stdev(values) if len(values) >= 2 else None}

    by_tier: dict[str, Any] = {}
    for tier in (1, 2, 3):
        selected = [r for r in complete if r["tier"] == tier]
        by_tier[str(tier)] = {**summary([r["score"] for r in selected]),
                             "pass_rate": statistics.mean(r["score"] >= .975 for r in selected) if selected else None,
                             "exact_score_rate": statistics.mean(r["score"] == 1.0 for r in selected) if selected else None}
    component_values: dict[str, list[float]] = defaultdict(list)
    per_task: dict[str, list[float]] = defaultdict(list)
    for record in complete:
        per_task[record["task_id"]].append(record["score"])
        for name, component in record.get("components", {}).items():
            component_values[name].append(component["score"])
    expected = tasks * rollout_count
    return {"aggregate": summary(means), "complete_replicates": len(means),
            "replicate_means": means, "sample_summary": summary([r["score"] for r in complete]),
            "expected_samples": expected, "completed_samples": len(complete),
            "coverage": len(complete) / expected if expected else 0.0,
            "status_counts": dict(Counter(r["status"] for r in records)), "by_tier": by_tier,
            "components": {name: summary(values) for name, values in component_values.items()},
            "per_task": {key: summary(values) for key, values in per_task.items()},
            "cost_usd": sum(r.get("cost_usd", 0) for r in records),
            "confirmed_cost_usd": sum(r.get("confirmed_cost_usd", 0) for r in records),
            "tokens": sum(r.get("tokens", 0) for r in records),
            "latency_s": summary([r.get("latency_s", 0) for r in records]),
            "retries": sum(max(0, len(r.get("attempts", [])) - r.get("model_turns", 1))
                           + r.get("judge_retries", 0) for r in records),
            "operational_failures": sum(r.get("status") not in {"complete", "pending"} for r in records),
            "statistics_note": "aggregate SD is sample SD across complete rollout-replicate means; by_tier/sample SD is across samples"}


def _exclusive_run(function):
    @wraps(function)
    async def wrapped(tasks: list[Task], output_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("RUN_ALREADY_ACTIVE") from None
            try:
                return await function(tasks, output_dir, **kwargs)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return wrapped


@_exclusive_run

async def run_evaluation(tasks: list[Task], output_dir: str | Path, *, model: str,
                         base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
                         api_key: str | None = None, rollouts: int = 3, mode: str = "single",
                         max_cost: float = 0.0, pricing: Pricing | None = None, concurrency: int = 2,
                         timeout_s: float = 120, max_turns: int = 6, judge: Any = None,
                         provider: Any = None, evidence_kind: str = "model_run", seed: int = 7001,
                         max_tokens: int = 512, temperature: float = 0.7) -> dict[str, Any]:
    if provider is None and pricing is None:
        raise ValueError("UNKNOWN_PRICING")
    if provider is None and not api_key:
        raise ValueError("API_KEY_REQUIRED")
    if not tasks or not 1 <= rollouts <= 20 or not 1 <= concurrency <= 8:
        raise ValueError("INVALID_EVALUATION_SIZE")
    if mode not in {"single", "tool"} or evidence_kind not in {"model_run", "fixture", "adversarial_baseline"}:
        raise ValueError("INVALID_EVALUATION_MODE")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("DUPLICATE_TASK_ID")
    if (not 1 <= max_turns <= 12 or not math.isfinite(timeout_s) or timeout_s <= 0
            or type(max_tokens) is not int or not 1 <= max_tokens <= 4096
            or not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise ValueError("INVALID_ROLLOUT_LIMITS")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prompts = {task.id: task_prompt(task, mode=mode) for task in tasks}
    split_names = {"training", "train", "validation", "evaluation", "eval", "transfer", "development", "seed"}
    common_tags = set.intersection(*(set(task.tags) for task in tasks))
    recognized_split = sorted(common_tags & split_names)
    transfer_freeze = None
    if recognized_split == ["transfer"] and evidence_kind == "model_run":
        from aster_gym.evidence import verify_transfer_freeze
        transfer_freeze = verify_transfer_freeze(tasks)
    config = {"split": recognized_split[0] if len(recognized_split) == 1 else "mixed", "model": model, "base_url": base_url, "seed": seed, "rollouts": rollouts,
              "mode": mode, "ruleset_version": RULESET_VERSION, "reward_version": REWARD_VERSION,
              "schema_version": SCHEMA_VERSION, "rules_hash": rules_hash(),
              "taskset_hash": stable_hash([task.model_dump(mode="json") for task in tasks]),
              "input_hashes": [task.input_hash for task in tasks],
              "task_ids": [task.id for task in tasks], "task_count": len(tasks),
              "prompt_hashes": {key: stable_hash(value) for key, value in prompts.items()},
              "evidence_kind": evidence_kind, "max_cost_usd": max_cost,
              "pricing": asdict(pricing) if pricing else None, "max_turns": max_turns,
              "timeout_s": timeout_s, "max_tokens": max_tokens, "temperature": temperature,
              "transfer_freeze": transfer_freeze,
              **_code_identity(),
              "judge": {"model": getattr(judge, "model", None), "prompt_hash": getattr(judge, "prompt_hash", None)}}
    config["run_id"] = stable_hash(config)[:20]
    config_path = output / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing["run_id"] != config["run_id"]:
            raise ValueError("RESUME_CONFIGURATION_MISMATCH")
    config["status"] = "running"
    atomic_json(config_path, config)
    journal = output / ".records"
    journal.mkdir(exist_ok=True)
    budget_path = output / "budget.json"
    saved_budget = json.loads(budget_path.read_text()) if budget_path.exists() else {}
    spent = saved_budget.get("accounted_usd", 0.0) + saved_budget.get("reserved_usd", 0.0)
    budget = CostBudget(max_cost, spent=spent)
    budget.actual_spend = saved_budget.get("confirmed_usd", 0.0)
    budget.on_change = lambda snapshot: atomic_json(budget_path, snapshot)
    owned_provider = provider is None
    if provider is None:
        if not api_key:
            raise ValueError("API_KEY_REQUIRED")
        provider = AsyncOpenAIProvider(model=model, base_url=base_url, api_key=api_key, budget=budget,
                                       pricing=pricing, timeout_s=timeout_s)
    else:
        provider.budget = budget
    if judge is not None:
        if hasattr(judge, "bind_budget"):
            judge.bind_budget(budget)
        elif evidence_kind == "model_run":
            raise ValueError("JUDGE_MUST_SHARE_COST_BUDGET")
    semaphore = asyncio.Semaphore(concurrency)
    judge_lock = asyncio.Lock()
    records: dict[str, dict[str, Any]] = {}
    transcripts: dict[str, dict[str, Any]] = {}
    for path in journal.glob("*.json"):
        saved = json.loads(path.read_text())
        row, trace = saved["record"], saved["transcript"]
        if (row.get("task_id") not in prompts or type(row.get("rollout")) is not int
                or not 0 <= row["rollout"] < rollouts
                or path.stem != stable_hash([row["task_id"], row["rollout"]])[:24]
                or (trace.get("task_id"), trace.get("rollout")) != (row["task_id"], row["rollout"])):
            raise ValueError("INVALID_RESUME_JOURNAL")
        records[path.stem] = saved["record"]
        transcripts[path.stem] = saved["transcript"]

    async def run_one(task: Task, rollout: int) -> None:
        key = stable_hash([task.id, rollout])[:24]
        previous = records.get(key)
        if previous and previous["status"] == "complete":
            return  # Never resample an incorrect but successfully delivered answer.
        async with semaphore:
            began = time.monotonic()
            old = transcripts.get(key)
            # Pending judges resume scoring the same final answer. Infrastructure
            # failures retry the request; their costs/attempts remain in history.
            scoring_only = previous is not None and previous["status"] == "pending"
            messages: list[dict[str, Any]] = old["messages"] if scoring_only and old else [
                {"role": "user", "content": prompts[task.id]}]
            attempts = list(previous.get("attempts", [])) if previous else []
            record: dict[str, Any] = {"task_id": task.id, "tier": task.difficulty, "rollout": rollout,
                                      "status": "provider_error", "score": None, "components": {},
                                      "latency_s": previous.get("latency_s", 0.0) if previous else 0.0,
                                      "tokens": previous.get("tokens", 0) if previous else 0,
                                      "cost_usd": previous.get("cost_usd", 0.0) if previous else 0.0,
                                      "mode": mode, "attempts": attempts,
                                      "model_turns": previous.get("model_turns", 0) if previous else 0,
                                      "tool_calls": previous.get("tool_calls", 0) if previous else 0}
            session = ToolSession(task)
            reference_reads = old.get("reference_reads", 0) if scoring_only and old else 0
            judge_calls: list[dict[str, Any]] = []
            grading_started = False
            try:
                async with asyncio.timeout(timeout_s):
                    if not scoring_only:
                        for turn in range(1 if mode == "single" else max_turns):
                            record["model_turns"] += 1
                            result = await provider.chat(messages, tools=TOOL_DEFINITIONS if mode == "tool" else None,
                                                         max_tokens=max_tokens, temperature=temperature)
                            attempts.extend([{**a, "turn": turn + 1} for a in result.attempts])
                            record["cost_usd"] += result.cost_usd
                            record["tokens"] += result.prompt_tokens + result.completion_tokens
                            messages.append(result.message)
                            calls = result.message.get("tool_calls", [])
                            if not calls:
                                break
                            if mode != "tool":
                                raise ProviderError("UNEXPECTED_TOOL_CALLS")
                            for call in calls:
                                function = call.get("function", {})
                                arguments = parse_tool_arguments(function.get("arguments", ""))
                                name = function.get("name", "")
                                response = session.call(name, arguments)
                                record["tool_calls"] += 1
                                if name in {"read_document", "lookup_rules"} and "error_code" not in response:
                                    reference_reads += 1
                                messages.append({"role": "tool", "tool_call_id": call.get("id", "invalid"),
                                                 "content": json.dumps(response)})
                        else:
                            raise ProviderError("MAX_TURNS")
                    if mode == "tool" and reference_reads == 0:
                        record.update(status="complete", score=0.0, gate="TOOLS_NOT_USED")
                    else:
                        raw = next((m.get("content", "") for m in reversed(messages) if m["role"] == "assistant"), "")
                        async with judge_lock:
                            ledger_start = len(getattr(judge, "ledger", []))
                            grading_started = True
                            try:
                                report = await score(task, raw, judge=judge)
                            finally:
                                judge_calls = list(getattr(judge, "ledger", [])[ledger_start:])
                        data = report.model_dump(mode="json")
                        record.update({key: data.get(key) for key in ("status", "score", "components", "gate", "error_code")})
            except ProviderError as exc:
                if exc.code in {"MAX_TURNS", "UNEXPECTED_TOOL_CALLS"}:
                    record.update(status="complete", score=0.0, gate=exc.code)
                else:
                    record["status"] = "cost_exhausted" if exc.code == "COST_EXHAUSTED" else "provider_error"
                record["error_code"] = exc.code
                attempts.extend(exc.attempts)
                record["cost_usd"] += sum(a.get("cost_usd", 0.0) for a in exc.attempts)
            except TimeoutError as exc:
                record.update(status="pending" if grading_started else "timeout",
                              error_code="JUDGE_TIMEOUT" if grading_started else "ROLLOUT_TIMEOUT")
                interrupted = getattr(exc.__cause__, "attempts", [])
                if interrupted and grading_started:
                    if not any(entry.get("attempts") == interrupted for entry in judge_calls):
                        judge_calls.append({"status": "pending", "code": "JUDGE_CANCELLED",
                                            "attempts": interrupted,
                                            "cost_usd": sum(a.get("cost_usd", 0.0) for a in interrupted)})
                else:
                    attempts.extend(interrupted)
                    record["cost_usd"] += sum(a.get("cost_usd", 0.0) for a in interrupted)
            except Exception:
                logger.exception("evaluation_failure", extra={"task_id": task.id, "rollout": rollout})
                record.update(status="internal_error", error_code="EVALUATION_ERROR")
            record["latency_s"] += time.monotonic() - began
            new_judge_cost = sum(entry.get("cost_usd", 0.0) for entry in judge_calls)
            record["tokens"] += sum(entry.get("prompt_tokens", 0) + entry.get("completion_tokens", 0)
                                    for entry in judge_calls)
            record["judge_cost_usd"] = (previous.get("judge_cost_usd", 0.0) if previous else 0.0) + new_judge_cost
            record["cost_usd"] += new_judge_cost
            judge_calls = (old.get("judge_calls", []) if old else []) + judge_calls
            record["judge_retries"] = sum(max(0, len(entry.get("attempts", [])) - 1) for entry in judge_calls)
            record["confirmed_cost_usd"] = sum(
                a.get("cost_usd", 0.0) for a in attempts if a.get("cost_kind") == "confirmed"
            ) + sum(a.get("cost_usd", 0.0) for entry in judge_calls
                    for a in entry.get("attempts", []) if a.get("cost_kind") == "confirmed")
            record["conservative_cost_usd"] = max(0.0, record["cost_usd"] - record["confirmed_cost_usd"])
            transcript = {"task_id": task.id, "rollout": rollout, "messages": messages,
                          "attempts": attempts, "status": record["status"], "reference_reads": reference_reads,
                          "judge_calls": judge_calls}
            if old and not scoring_only:
                transcript["prior_attempts"] = old.get("prior_attempts", []) + [
                    {"messages": old["messages"], "status": old["status"]}]
            records[key], transcripts[key] = record, transcript
            atomic_json(journal / (key + ".json"), {"record": record, "transcript": transcript})
            atomic_json(budget_path, budget.snapshot())

    try:
        await asyncio.gather(*(run_one(task, rollout) for rollout in range(rollouts) for task in tasks))
    finally:
        if owned_provider:
            await provider.aclose()
        ordered = sorted(records.values(), key=lambda r: (r["rollout"], r["task_id"]))
        metrics = summarize_records(ordered, len(tasks), rollouts)
        metrics["budget"] = budget.snapshot()
        atomic_json(output / "scores.json", {"records": ordered})
        atomic_json(output / "metrics.json", metrics)
        temp = output / "transcript.jsonl.tmp"
        with temp.open("w") as handle:
            for entry in sorted(transcripts.values(), key=lambda r: (r["rollout"], r["task_id"])):
                handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temp, output / "transcript.jsonl")
        config["status"] = "complete" if metrics["coverage"] == 1.0 else "partial"
        atomic_json(config_path, config)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="https://generativelanguage.googleapis.com/v1beta/openai")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--tier", default="all")
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--mode", choices=["single", "tool"], default="single")
    parser.add_argument("--max-cost", type=float, default=0)
    parser.add_argument("--input-per-million", type=float, required=True)
    parser.add_argument("--output-per-million", type=float, required=True)
    parser.add_argument("--verified-free", action="store_true")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.tasks:
        content = args.tasks.read_text()
        data = json.loads(content) if args.tasks.suffix == ".json" else [json.loads(line) for line in content.splitlines()]
        tasks = [Task.model_validate(row) for row in (data.get("tasks", []) if isinstance(data, dict) else data)]
    else:
        tasks = generate_taskset(n=args.n, seed=args.seed, tier=args.tier, split="evaluation")
    from aster_gym.scoring import make_judge
    metrics = asyncio.run(run_evaluation(tasks, args.output, model=args.model, base_url=args.base_url,
                                         api_key=os.getenv("MODEL_API_KEY"), seed=args.seed,
                                         rollouts=args.rollouts, mode=args.mode, max_cost=args.max_cost,
                                         pricing=Pricing(args.input_per_million, args.output_per_million, args.verified_free),
                                         concurrency=args.concurrency, timeout_s=args.timeout, judge=make_judge()))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
