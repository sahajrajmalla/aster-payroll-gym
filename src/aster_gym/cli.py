"""CLI commands stay lightweight; cloud training refuses this Mac before imports."""

import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

from .generator import (
    generate_taskset,
    read_taskset,
    task_prompt,
    validate_splits,
    validate_taskset,
    write_taskset,
)
from .logging import configure_logging
from .scoring import adversarial_answers, make_judge, score
from .versions import REWARD_VERSION, RULESET_VERSION, SCHEMA_VERSION, rules_hash, stable_hash


def _read_config(path: str | None) -> dict:
    return json.loads(Path(path).read_text()) if path else {}


def _code_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted-workspace"


async def run_baselines(task_path: str, output_dir: str, n: int | None = None) -> None:
    tasks = read_taskset(task_path)
    if n is not None:
        if n < 1:
            raise ValueError("baseline task count must be positive")
        tasks = tasks[:n]
    for name, answer in adversarial_answers().items():
        destination = Path(output_dir) / ("baseline-" + name)
        destination.mkdir(parents=True, exist_ok=True)
        records, transcripts = [], []
        for task in tasks:
            for rollout in range(3):
                started = time.perf_counter()
                graded = await score(task, answer)
                records.append({"task_id": task.id, "tier": task.difficulty, "rollout": rollout,
                                **graded.model_dump(mode="json"), "latency_s": time.perf_counter() - started,
                                "tokens": 0, "cost_usd": 0, "mode": "single"})
                transcripts.append({"task_id": task.id, "rollout": rollout, "status": graded.status,
                    "attempts": [], "messages": [{"role": "user", "content": task_prompt(task)},
                                                 {"role": "assistant", "content": json.dumps(answer)}]})
        config = {"run_id": "baseline-" + name, "model": name, "seed": tasks[0].seed,
                  "ruleset_version": RULESET_VERSION, "reward_version": REWARD_VERSION,
                  "schema_version": SCHEMA_VERSION, "rules_hash": rules_hash(), "code_revision": _code_revision(),
                  "taskset_hash": stable_hash([t.model_dump(mode="json") for t in tasks]),
                  "prompt_hashes": {t.id: stable_hash(task_prompt(t)) for t in tasks},
                  "evidence_kind": "adversarial_baseline", "status": "complete", "rollouts": 3,
                  "split": "evaluation", "mode": "single", "temperature": None,
                  "task_count": len(tasks), "task_ids": [t.id for t in tasks],
                  "input_hashes": [t.input_hash for t in tasks],
                  "note": "Deterministic handwritten policy; repeated records are not stochastic LLM samples."}
        (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        # Keep each complete record on one line: auditable without enormous diffs.
        (destination / "scores.json").write_text('{"records":[\n' +
            ",\n".join(json.dumps(r, separators=(",", ":")) for r in records) + "\n]}\n")
        (destination / "transcript.jsonl").write_text("".join(json.dumps(t) + "\n" for t in transcripts))
        from .eval import summarize_records
        (destination / "metrics.json").write_text(json.dumps(
            summarize_records(records, expected_tasks=len(tasks), rollouts=3), indent=2) + "\n")
    print(json.dumps({"baselines": 3, "tasks": len(tasks), "model_calls": 0}))


async def _eval(args) -> None:
    from .eval import run_evaluation
    from .providers import Pricing
    config = _read_config(args.config)
    model = args.model or config.get("model")
    base_url = args.base_url or config.get("base_url") or os.environ.get("MODEL_BASE_URL")
    if not model or not base_url:
        raise ValueError("--model and --base-url (or config) are required")
    if urlparse(base_url).hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
        raise ValueError("Local inference endpoints are forbidden; use a remote provider or Colab")
    path = args.tasks or config.get("tasks", "data/evaluation.jsonl")
    tasks = read_taskset(path)
    if args.tier != "all":
        tasks = [t for t in tasks if t.difficulty == int(args.tier)]
    n = args.n or config.get("n")
    if n:
        tasks = tasks[:int(n)]
    preset: dict[str, Any] = next((m for m in config.get("models", []) if m.get("name") == model), {})
    pricing_data = preset.get("pricing", config.get("pricing"))
    pricing = Pricing(**pricing_data) if pricing_data else None
    judge = make_judge()
    try:
        metrics = await run_evaluation(tasks, args.output, model=model, base_url=base_url,
            api_key=os.environ.get(config.get("api_key_env", "MODEL_API_KEY")),
            rollouts=args.rollouts if args.rollouts is not None else config.get("rollouts", 3),
            mode=args.mode if args.mode is not None else config.get("mode", "single"),
            max_cost=args.max_cost if args.max_cost is not None else config.get("max_cost_usd", 0),
            pricing=pricing,
            concurrency=args.concurrency if args.concurrency is not None else config.get("concurrency", 1),
            timeout_s=args.timeout if args.timeout is not None else config.get("timeout_s", 120),
            max_turns=args.max_turns if args.max_turns is not None else config.get("max_turns", 6),
            max_tokens=args.max_tokens if args.max_tokens is not None else config.get("max_tokens", 512),
            temperature=args.temperature if args.temperature is not None else config.get("temperature", .7),
            judge=judge, seed=args.seed if args.seed is not None else config.get("seed", 7001))
        print(json.dumps(metrics, indent=2))
    finally:
        if judge is not None:
            await judge.aclose()


def main() -> None:
    load_dotenv()
    configure_logging()
    parser = argparse.ArgumentParser(description="Aster Payroll Gym — no local model workloads")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate schema, oracle, context and split separation")
    validate.add_argument("--data-dir", default="data")
    generate = commands.add_parser("generate")
    generate.add_argument("--n", type=int, default=12)
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--tier", default="all")
    generate.add_argument("--output", default="data/generated.jsonl")
    baseline = commands.add_parser("baselines")
    baseline.add_argument("--tasks", default="data/evaluation.jsonl")
    baseline.add_argument("--output", default="results")
    baseline.add_argument("--n", type=int)
    evaluate = commands.add_parser("eval")
    evaluate.add_argument("--config")
    evaluate.add_argument("--model")
    evaluate.add_argument("--base-url")
    evaluate.add_argument("--tasks")
    evaluate.add_argument("--n", type=int)
    evaluate.add_argument("--tier", choices=["all", "1", "2", "3"], default="all")
    evaluate.add_argument("--rollouts", type=int)
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--mode", choices=["single", "tool"])
    evaluate.add_argument("--max-cost", type=float)
    evaluate.add_argument("--concurrency", type=int)
    evaluate.add_argument("--timeout", type=float)
    evaluate.add_argument("--max-turns", type=int)
    evaluate.add_argument("--max-tokens", type=int)
    evaluate.add_argument("--temperature", type=float)
    evaluate.add_argument("--output", default="results/evaluation")
    dashboard = commands.add_parser("report")
    dashboard.add_argument("--results", default="results")
    dashboard.add_argument("--output", default="site")
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    imported = commands.add_parser("import-results")
    imported.add_argument("--bundle", required=True)
    imported.add_argument("--output", default="results/imported")
    exported = commands.add_parser("export-results")
    exported.add_argument("--run-dir", required=True)
    exported.add_argument("--output", required=True)
    analysis = commands.add_parser("analyze", help="Recompute transfer and human-review statistics from evidence")
    analysis.add_argument("--results", default="results")
    analysis.add_argument("--reviews", default="reviews")
    frozen = commands.add_parser("freeze-transfer", help="Freeze five human-approved tasks before model outcomes")
    frozen.add_argument("--tasks", default="data/transfer.jsonl")
    frozen.add_argument("--packet", default="reviews/task-review-packet.json")
    study = commands.add_parser("judge-study", help="Three uncached remote ratings for fifteen review examples")
    study.add_argument("--packet", default="reviews/judge-review-packet.json")
    commands.add_parser("train", help="Cloud-only; use the Colab notebook")
    commands.add_parser("infer", help="Cloud-only; use the Colab notebook")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            partitions = {name: read_taskset(Path(args.data_dir) / f"{name}.jsonl")
                          for name in ("seeds", "train", "validation", "evaluation", "transfer")}
            print(json.dumps({"partitions": {k: validate_taskset(v) for k, v in partitions.items()},
                              "splits": validate_splits(partitions)}, indent=2))
        elif args.command == "generate":
            tasks = generate_taskset(args.n, seed=args.seed, tier=args.tier)
            write_taskset(tasks, Path(args.output))
            print(json.dumps(validate_taskset(tasks)))
        elif args.command == "baselines":
            asyncio.run(run_baselines(args.tasks, args.output, args.n))
        elif args.command == "eval":
            asyncio.run(_eval(args))
        elif args.command == "report":
            from .reporting import build_dashboard, write_summary
            print(build_dashboard(args.results, args.output))
            write_summary(args.results, "docs/saved-results.md")
        elif args.command == "serve":
            import uvicorn
            uvicorn.run("aster_gym.api:app", host=args.host, port=args.port, log_config=None)
        elif args.command == "import-results":
            from .bundles import import_bundle
            print(import_bundle(args.bundle, args.output))
        elif args.command == "export-results":
            from .bundles import export_bundle
            print(export_bundle(args.run_dir, args.output))
        elif args.command == "analyze":
            from .evidence import analyze_results
            print(analyze_results(args.results, args.reviews))
        elif args.command == "freeze-transfer":
            from .evidence import freeze_transfer
            print(freeze_transfer(read_taskset(args.tasks), args.packet))
        elif args.command == "judge-study":
            from .judge_study import run_study
            print(json.dumps(asyncio.run(run_study(args.packet)), indent=2))
        elif args.command in {"train", "infer"}:
            from .cloud.guard import CloudOnlyError
            raise CloudOnlyError("Use notebooks/aster_colab.ipynb in a Colab GPU runtime. Local model workloads are forbidden.")
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
