# Shared implementation contract

This engineering contract coordinates modules; `rules/aster-payroll-v1.md` is the
authoritative domain specification. Source rule text was committed to the working
tree before calculator implementation. No human review or actual model runs are
claimed until separately recorded.

## Core interfaces (owned by core implementation)

`Task`: strict Pydantic model, `id: str`, `slice: str`, `difficulty: int`,
`prompt: str`, `context_files: dict[str,str]`, `ground_truth: Answer`,
`verifier: str`, `tags: list[str]`, `schema_version: str`, `ruleset_version: str`,
`seed: int`, `inputs: dict`, `input_hash: str`.
Inputs contain `kind: retrieval|payslip`, `pay_date`, `query_field` if retrieval,
`salary_records`, `paid_days`, `period_days`, `bonus_cents`,
`ytd_pensionable_cents`, `ytd_year`, and `schedules`. Schedules are authoritative
rule-defined dictionaries. Public contexts give all necessary evidence, without
oracle results. `.model_dump(mode='json')` serializes internal artifacts only.

`Answer`: decision `answer|needs_information`, `result: dict[str,int]|None`,
`issue_codes: list[str]`, `missing_fields: list[str]`, `citations: list[str]`,
`explanation: str`. All six fields required; no extra properties. Amounts cents.

`generator.generate_taskset(n=12, seed=7, tier='all', split='development', **knobs)`
returns `list[Task]`. `reference.solve(inputs: dict) -> Answer` computes oracle.
`generator.public_task(task, mode='single') -> dict` returns allowlisted public DTO
keys `id`, `prompt`, `context_files`, `ruleset_version`, `schema_version`,
`tools_available`. No difficulty, tags, seed, inputs, expected, or ground_truth.
`generator.task_prompt(task, mode='single') -> str` constructs model-facing text.

`scoring.score(task: Task, raw: str|dict|Answer, judge=None) -> ScoreReport` is async.
`judge` implements `async grade(task, answer) -> JudgeResult` (defined in schemas).
If an applicable judge is missing/unavailable: status `pending`, score `None`.
Reports: `task_id`, `status: complete|pending`, `score: float|None`,
`components: dict[str, ComponentScore]`, `gate: str|None`, `error_code: str|None`,
`ruleset_version`, `reward_version`. ComponentScore: `score`, `weight`,
`clauses: list[str]`, `code: str`. Public reports contain no expected values.
`scoring.make_judge()` returns remote judge or None from environment.

`versions`: RULESET_VERSION='ASTER-1.0', REWARD_VERSION='1.0', SCHEMA_VERSION='1.0';
`rules_text()`, `rules_hash()`, `stable_hash(value)`.

## Ownership

Root: schemas, generator, reference, parser, scoring, judge, core tests, seed
data, CLI composition, broad docs and requirements. Harness agent: tools.py,
environment.py, eval.py, providers.py and corresponding tests/configs. Cloud
agent: cloud/, bundles.py, notebooks/, configs/train.json and corresponding tests.
Service agent: api.py, store.py, reporting.py, scripts/quickstart.py, deployment
files, API/dashboard tests and deployment docs. Avoid editing another area until
coordinated. All code must remain lightweight by default.

## Results interchange

Each results run directory contains config.json, transcript.jsonl, metrics.json,
scores.json. Config includes run_id, model, seed, ruleset_version, reward_version,
schema_version, taskset_hash, prompt_hashes, evidence_kind (`model_run`,
`adversarial_baseline`, `fixture`), status. Scores JSON object contains `records`:
each has task_id, tier, rollout, status, score, components, latency_s, tokens,
cost_usd, mode. Metrics derive only from these records. A transcript JSONL record
has task_id, rollout, messages, attempts, status; each message preserves role and
tool calls. Training results additionally use training_metrics.jsonl and
heldout.json with actual step/curve data. No fabricated outputs.
