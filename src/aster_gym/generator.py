"""Seeded business evidence generation; ground truth is always reference.solve."""

import calendar
import json
import random
from pathlib import Path
from typing import Any

from .reference import authoritative_schedules, solve
from .schemas import Task
from .versions import RULESET_VERSION, SCHEMA_VERSION, rules_text, stable_hash


def input_fingerprint(inputs: dict[str, Any]) -> str:
    """Normalize unordered evidence lists without weakening strict input types.

    Reordering salary records or schedule copies must not evade split overlap
    checks. Repeated records are retained because multiplicity can be a blocker.
    """
    normalized = dict(inputs)
    for key in ("salary_records", "schedules"):
        if isinstance(normalized.get(key), list):
            normalized[key] = sorted(normalized[key], key=stable_hash)
    return stable_hash(normalized)


def _context(inputs: dict[str, Any]) -> dict[str, str]:
    return {
        "request.json": json.dumps({k: inputs[k] for k in ("kind", "pay_date", "query_field") if k in inputs},
                                   sort_keys=True),
        "payroll-records.json": json.dumps({k: v for k, v in inputs.items()
                                           if k not in {"kind", "pay_date", "query_field", "schedules"}},
                                          sort_keys=True),
        "schedules.json": json.dumps(inputs.get("schedules", []), sort_keys=True),
        "rules.md": rules_text(),
    }


def task_from_inputs(inputs: dict[str, Any], *, seed: int, difficulty: int,
                     tags: list[str] | None = None, narrative: str | None = None) -> Task:
    """Inputs first, independent solution second; ID conveys no answer or trap type."""
    fingerprint = input_fingerprint(inputs)
    request = (f"Retrieve {inputs.get('query_field')} for pay date {inputs.get('pay_date')}."
               if inputs.get("kind") == "retrieval"
               else f"Prepare the monthly AST payslip for pay date {inputs.get('pay_date')}.")
    prompt = narrative or (
        request + " Use the authoritative evidence and rule clauses. If the rules do not determine "
        "the answer, identify the blockers and ask for the required clarification. Return only the "
        "R10 JSON object. Document references: request.json, payroll-records.json, schedules.json, rules.md."
    )
    return Task(id="task-" + stable_hash({"input": fingerprint, "seed": seed})[:20],
                difficulty=difficulty, prompt=prompt, context_files=_context(inputs),
                ground_truth=solve(inputs), tags=tags or [], seed=seed,
                inputs=inputs, input_hash=fingerprint)


def generate_taskset(n: int = 12, seed: int = 7, tier: str | int = "all",
                     split: str = "development", *, proration: bool = True,
                     ceiling_binding: bool = True, irrelevant_documents: int = 1,
                     missing_inputs: bool = True, interacting_rules: int = 3,
                     lookup_count: int = 2) -> list[Task]:
    """Generate at most 1000 light tasks. All difficulty knobs are explicit metadata."""
    if type(n) is not int or not 1 <= n <= 1000 or type(seed) is not int:
        raise ValueError("n must be 1..1000 and seed an integer")
    tier = str(tier)
    if tier not in {"all", "1", "2", "3"}:
        raise ValueError("tier must be all, 1, 2 or 3")
    if not 0 <= irrelevant_documents <= 5 or not 1 <= interacting_rules <= 3 or not 1 <= lookup_count <= 4:
        raise ValueError("difficulty knob out of range")
    rng = random.Random(seed)
    tasks: list[Task] = []
    seen: set[str] = set()
    for index in range(n):
        difficulty = index % 3 + 1 if tier == "all" else int(tier)
        for _ in range(100):
            year = rng.choice([2029, 2030]) if difficulty == 1 else 2030
            month = rng.randint(1, 12)
            day = rng.randint(1, 28)
            pay_date = f"{year}-{month:02d}-{day:02d}"
            days = calendar.monthrange(year, month)[1]
            salary = rng.randint(65_000, 720_000) if interacting_rules >= 2 else rng.randint(20_000, 80_000)
            inputs: dict[str, Any] = {
                "kind": "retrieval" if difficulty == 1 else "payslip", "pay_date": pay_date,
                "salary_records": [{"effective": f"{year}-01-01", "signed": True,
                                    "monthly_salary_cents": salary}],
                "period_days": days, "paid_days": rng.randint(1, days) if proration else days,
                "bonus_cents": rng.choice([0, rng.randint(100, 50_000)]) if interacting_rules >= 2 else 0,
                "ytd_pensionable_cents": rng.choice([0, 1_200_000, rng.randint(1_000_000, 1_199_999)])
                if ceiling_binding else 0,
                "ytd_year": year, "schedules": authoritative_schedules(),
            }
            if difficulty == 1:
                inputs["query_field"] = rng.choice(["annual_ceiling_cents", "allowance_cents"])
            tags = [split]
            if difficulty == 3:
                trap = (index // 3 if tier == "all" else index) % 3
                if trap == 0 and missing_inputs:
                    del inputs["ytd_pensionable_cents"]
                    tags += ["required_input_absent", "MISSING_INPUT"]
                elif trap == 1 or (trap == 0 and not missing_inputs):
                    inputs["salary_records"].append({"effective": f"{year}-01-01", "signed": True,
                                                     "monthly_salary_cents": salary + 1234})
                    tags += ["evidence_conflict", "SALARY_CONFLICT"]
                else:
                    # R8 is silent about 2031 rates; extrapolation is forbidden.
                    inputs["pay_date"] = f"2031-{month:02d}-{day:02d}"
                    inputs["period_days"] = calendar.monthrange(2031, month)[1]
                    inputs["paid_days"] = min(inputs["paid_days"], inputs["period_days"])
                    inputs["ytd_year"] = 2031
                    tags += ["rules_silent", "NO_SCHEDULE"]
            task = task_from_inputs(inputs, seed=seed, difficulty=difficulty, tags=tags)
            if task.input_hash not in seen:
                break
        else:
            raise ValueError("unable to generate unique inputs")
        seen.add(task.input_hash)
        for distractor in range(irrelevant_documents):
            task.context_files[f"unrelated-{distractor}.txt"] = (
                "An unsigned planning note: proposed future salary is not authoritative. "
                f"Reference code {rng.randint(100000, 999999)}."
            )
        # lookup_count controls evidence fragmentation, not a hidden reward shortcut.
        if lookup_count >= 3:
            records = json.loads(task.context_files.pop("payroll-records.json"))
            for field, value in records.items():
                task.context_files[f"record-{field}.json"] = json.dumps({field: value}, sort_keys=True)
            task.prompt = task.prompt.replace("payroll-records.json", ", ".join(
                k for k in task.context_files if k.startswith("record-")))
        tasks.append(task)
    return tasks


def task_prompt(task: Task, mode: str = "single") -> str:
    if mode not in {"single", "tool"}:
        raise ValueError("unknown environment mode")
    contract = ("\nReturn exactly these six JSON properties: decision, result, issue_codes, "
                "missing_fields, citations, explanation. Money is integer cents. "
                "For needs_information result must be null. Do not include markdown or extra keys.")
    if mode == "tool":
        return task.prompt + contract + "\nAvailable document IDs: " + ", ".join(sorted(task.context_files))
    documents = "\n".join(f"\n--- {key} ---\n{value}" for key, value in sorted(task.context_files.items()))
    return task.prompt + contract + documents


def public_task(task: Task, mode: str = "single") -> dict[str, Any]:
    """Allowlist only; no private labels, seeds, oracle fields, or internal input dict."""
    return {"id": task.id, "prompt": task_prompt(task, mode),
            "context_files": dict(task.context_files) if mode == "single" else {},
            "ruleset_version": RULESET_VERSION, "schema_version": SCHEMA_VERSION,
            "tools_available": ["read_document", "lookup_rules", "calculate"] if mode == "tool" else []}


def write_taskset(tasks: list[Task], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(t.model_dump_json() + "\n" for t in tasks), encoding="utf-8")


def read_taskset(path: str | Path) -> list[Task]:
    return [Task.model_validate_json(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_taskset(tasks: list[Task]) -> dict[str, Any]:
    if not tasks or len({t.id for t in tasks}) != len(tasks) or len({t.input_hash for t in tasks}) != len(tasks):
        raise ValueError("empty or duplicate taskset")
    for task in tasks:
        if input_fingerprint(task.inputs) != task.input_hash:
            raise ValueError("input fingerprint mismatch")
        if solve(task.inputs).model_dump() != task.ground_truth.model_dump():
            raise ValueError("reference disagreement")
        if task.context_files != _context(task.inputs):
            # Additional distractors/fragments are allowed; canonical reference data must still match.
            expected = _context(task.inputs)
            if any(task.context_files.get(k) != v for k, v in expected.items()
                   if k != "payroll-records.json"):
                raise ValueError("context evidence differs from reference inputs")
            if "payroll-records.json" in task.context_files:
                if task.context_files["payroll-records.json"] != expected["payroll-records.json"]:
                    raise ValueError("payroll context differs from reference inputs")
            else:
                merged = {}
                for key, value in task.context_files.items():
                    if key.startswith("record-"):
                        merged.update(json.loads(value))
                if merged != json.loads(expected["payroll-records.json"]):
                    raise ValueError("fragmented context differs from reference inputs")
    return {"tasks": len(tasks), "tiers": {str(tier): sum(t.difficulty == tier for t in tasks)
                                             for tier in (1, 2, 3)},
            "taskset_hash": stable_hash([t.input_hash for t in tasks])}


def validate_splits(partitions: dict[str, list[Task]]) -> dict[str, Any]:
    seeds: dict[int, str] = {}
    fingerprints: dict[str, str] = {}
    for split, tasks in partitions.items():
        validate_taskset(tasks)
        for task in tasks:
            if task.seed in seeds and seeds[task.seed] != split:
                raise ValueError("SPLIT_SEED_LEAKAGE")
            if task.input_hash in fingerprints and fingerprints[task.input_hash] != split:
                raise ValueError("SPLIT_INPUT_LEAKAGE")
            seeds[task.seed] = split
            fingerprints[task.input_hash] = split
    return {"status": "disjoint", "partitions": {k: len(v) for k, v in partitions.items()}}
