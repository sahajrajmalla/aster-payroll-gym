# Aster Payroll Gym

A synthetic monthly-payroll environment with deterministic Python ground truth
and one clause-auditable scorer shared by evaluation, reinforcement learning and
an external submission API. The fictional rulebook defines correctness.

- [Repository](https://github.com/sahajrajmalla/aster-payroll-gym)
- [Dashboard](https://sahajrajmalla.com.np/aster-payroll-gym/)
- [Colab notebook](https://colab.research.google.com/github/sahajrajmalla/aster-payroll-gym/blob/main/notebooks/aster_colab.ipynb)
- [Sandbox](https://aster-payroll-gym.onrender.com) · [API contract](https://aster-payroll-gym.onrender.com/docs)
- Loom: pending; see [submission checklist](docs/submission-checklist.md).

Implementation and lightweight QA pass the checks recorded in
[QA evidence](docs/final-qa.json), including tests, lint, types and clean startup.
Deterministic baselines and a genuine 90-sample Gemini 3.5 evaluation are recorded;
Gemini 3.8 remains rate-limited/partial. Public sandbox Tier-2/3 submissions and
hosted restart persistence passed; Qwen/RL and independent reviews remain pending.
Missing evidence is never replaced with fixtures.
The Colab notebook now defaults to no-API mode: Qwen inference can run on its GPU,
while training, remote comparisons and judge study are explicitly skipped.
Judge-dependent scores remain pending. This mode produces partial evidence and
does not complete all assignment requirements; see [Colab guide](docs/colab.md).
The [owner-supplied October 7 status](docs/colab-run-status-2026-10-07.json)
records a failed GPU preflight and deliberate API skips, not a successful experiment.

## Start here

Use [the short self-run guide](docs/setup.md) for essential files and copy-paste commands.
Read [the project explanation](docs/project-understanding.md), then
[the submission walkthrough](docs/submission-walkthrough.md).
For presentation, use [the interview guide](docs/interview-preparation.md),
[one-page reminder](docs/interview-cheatsheet.md) and
[ten-minute Loom script](docs/loom-script.md).

## Lightweight setup

Python 3.11 and uv are required:

```bash
uv sync --locked --extra server
uv run --extra server aster-gym validate
uv run --extra server pytest -q
uv run --extra server ruff check .
uv run --extra server mypy src/aster_gym scripts
uv run --extra server aster-gym report --output site
uv run --extra server aster-gym serve
```

The API runs at `http://127.0.0.1:8000`, with `/healthz` and `/docs`.
The dashboard is `site/index.html`. The server extra adds only Postgres support;
the default dependencies contain no training stack.
**Training, local inference and weight downloads are forbidden on this computer.**
Never install `--extra cloud` locally. Cloud entrypoints require Colab and CUDA
before model imports. Create `.env` from `.env.example` only if it does not already
exist; preserve existing secrets. Keep keys out of Git and video recordings.

## Domain and data

ASTER-1.0 covers signed salary authority, proration, bonuses, a capped contribution,
allowances, marginal tax and net pay. All amounts are integer fictional AST cents.

- Tier 1: retrieve an effective schedule field.
- Tier 2: calculate interacting payroll lines.
- Tier 3: identify absent YTD earnings, conflicting authoritative salary records,
  or a missing effective schedule; request the specific needed evidence.

Frozen data: 12 seeds, 30 training, 15 validation, 120 evaluation and five transfer
tasks. Disjoint seed namespaces and normalized input fingerprints prevent split
overlap. Repeated Tier-1 rule templates remain a documented limitation.
Track B changes presentation while retaining rules; human approval and freeze
must precede transfer evaluation.

The generator supports unlimited grading within these fictional rules. It cannot
authenticate real documents, interpret other jurisdictions or resolve discretionary
employment disputes.

## Scoring and experiments

Normal weights: correctness 60%, fields 25%, action 10%, format 5%.
Blocked-task weights: diagnosis 55%, fields 25%, action 10%, explanation 5%, format 5%.
Invalid answers, unsafe computation and unjustified abstention score zero.
Substantive errors cannot exceed 0.20; formatting alone earns no reward.
A remote judge assesses only an otherwise correct blocking explanation. Outages
remain pending. Pass threshold: 0.975. See [reward specification](docs/reward-spec.md).

The evaluation runner stores transcripts, component scores, retry/usage/cost
accounting and mean±sample SD of complete rollout means. Compare three policies
on the same 30 tasks with three rollouts; run the small model on all 120 tasks.
Actual deterministic baselines are reported separately from model runs.

All Qwen inference and GRPO training run in Colab. Defaults: Qwen2.5-0.5B-Instruct,
LoRA, four completions per prompt, 80 steps per beta and beta 0.001/0.10. The
reference is frozen; validation selects beta before untouched held-out comparison.
For four sampled answers, `A_i = (R_i − mean(R)) / (std(R) + ε)`.
The implemented token-masked, sequence-normalized objective is:

```text
J = E[(1/G) Σ_i (1/|o_i|) Σ_t
      {min(ρ_it A_i, clip(ρ_it, 0.8, 1.2) A_i) − β k3_it}]
ρ_it = πθ(o_it | context) / πold(o_it | context)
d_it = log πref(o_it | context) − log πθ(o_it | context)
k3_it = exp(d_it) − d_it − 1
```

GRPO avoids a separate critic and makes the shared task reward the direct learning
signal. Equal-reward groups have zero advantage; KL can still contribute a gradient.
The adapter-disabled frozen backbone supplies the initial reference; its initial
equivalence and parameter immutability are checked.
Full settings, reference checks and recovery are in [Colab guide](docs/colab.md).
Keep `SKIP_API_PHASES=True` if you have no API keys. With a successful GPU
preflight, enable `START_INFERENCE` to evaluate Qwen; leave the training switches
off. To run the full workflow later, set `SKIP_API_PHASES=False`, configure verified
judge access and use a fresh output directory. Do not mix the two scoring identities.

Download only the compact cloud results bundle, then run:

```bash
uv run aster-gym import-results --bundle /path/to/colab-results.zip
uv run aster-gym analyze
uv run aster-gym report --output site
```

Import checks versions, checksums, paths, sizes and split separation. Checkpoints
stay in cloud storage. Partial/failed runs stay visibly partial/failed.

## Independent sandbox

The [public sandbox](https://aster-payroll-gym.onrender.com) passed correct external
Tier-2/3 submissions, each scoring **1.0**, and both saved runs survived a confirmed
hosted restart. Eleven public contract checks also passed.
[Acceptance evidence](docs/render-sandbox-acceptance.json) ·
[Tier 2](docs/render-tier2-proof.json) · [Tier 3](docs/render-tier3-proof.json).

Run a correct public submission without cloning:

```bash
curl -fsSL https://raw.githubusercontent.com/sahajrajmalla/aster-payroll-gym/8c96d28abe5fe6f5bd504d96d091ee6920f040d0/scripts/sandbox_smoke.py | python3 - https://aster-payroll-gym.onrender.com
```

For Tier 3, append `--tier 3 --receipt tmp/tier3-private.json --proof tmp/tier3-proof.json`.
See [deployment guide](docs/deployment.md) for restart verification.
The separate `python scripts/quickstart.py https://aster-payroll-gym.onrender.com`
command deliberately submits an invalid answer.
The acceptance client solves from public evidence. `/tasks`, `/submit` and
`/runs/{run_id}` share production scoring. Public responses exclude expected answers. Runs are persisted,
token protected, rate limited and idempotent.

## Required documentation

[Rules](rules/aster-payroll-v1.md) · [architecture](docs/architecture.md) ·
[schema](docs/task-schema.md) · [evaluation](docs/evaluation.md) ·
[RL report](docs/rl-report.md) · [failure analysis](docs/failure-analysis.md) ·
[edge cases](docs/edge-cases.md) · [trade-offs](docs/assumptions-and-tradeoffs.md) ·
[transfer](docs/advanced-track.md) · [human review](docs/human-review.md) ·
[traceability](docs/traceability.md) · [tool disclosure](docs/ai-usage.md).

Use the [submission checklist](docs/submission-checklist.md) as the final authority
for readiness. Passing code checks does not establish model improvement.
