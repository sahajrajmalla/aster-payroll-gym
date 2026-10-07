# Aster Payroll Gym

A synthetic payroll gym with a separate written rulebook, deterministic Python
ground truth and one auditable scorer shared by evaluation, cloud RL and a public API.

[Repository](https://github.com/sahajrajmalla/aster-payroll-gym) ·
[Dashboard](https://sahajrajmalla.com.np/aster-payroll-gym/) ·
[Sandbox](https://aster-payroll-gym.onrender.com/docs) ·
[Colab](https://colab.research.google.com/github/sahajrajmalla/aster-payroll-gym/blob/main/notebooks/aster_colab.ipynb)

**Implementation checked; experimental submission incomplete.** The Render/Neon
sandbox passed correct Tier-2/3 submissions and persistence after restart. Qwen
completed 486 genuine Colab samples; none passed. A fresh Gemini run completed its
three rollouts on all 120 evaluation tasks: 360 answers, reward .12049 ± .00732,
32 passes and rewards from 0 to 1. The judge produced 45 uncached responses:
one malformed output and 2 unstable examples among 14 complete triples. Human
validity labels, transfer approvals/results, compatible frontier comparison and
actual RL sweeps remain pending. See [checklist](docs/submission-checklist.md),
[measured evidence](docs/evaluation.md) and [QA](docs/final-qa.json).

## Try the public sandbox without cloning

Python 3 is enough. This pinned client fetches a Tier-2 task, independently solves
it from public documents, submits it and retrieves its score:

```sh
curl -fsSL https://raw.githubusercontent.com/sahajrajmalla/aster-payroll-gym/8c96d28abe5fe6f5bd504d96d091ee6920f040d0/scripts/sandbox_smoke.py | python3 - https://aster-payroll-gym.onrender.com
```

Expect `submission_verified`, `complete` and score 1.0. Render Free may need a cold
start; the client retries bounded network failures. Ten tasks/run, caller quotas,
bounded payloads and judge-call limits bound abuse and cost. Run tokens protect
saved submissions; caller API keys and expected answers are never exposed.

## Run locally

Python 3.11 and uv are required. No model packages are installed by these commands.

```sh
uv sync --locked --extra server
uv run --extra server aster-gym validate
uv run --extra server pytest -q
uv run --extra server ruff check .
uv run --extra server mypy src/aster_gym scripts
uv run --extra server aster-gym report --output site
uv run --extra server aster-gym serve
```

Open `http://127.0.0.1:8000/docs`. Use [setup](docs/setup.md) for local dashboard,
public sandbox and result-import commands. Preserve existing `.env`; keep credentials
and private run receipts out of Git and recordings. Install the `cloud` extra only
in Colab. Model entrypoints require hosted Colab and CUDA before importing models.

## Task and reward design

ASTER-1.0 defines document authority, proration, bonuses, capped contributions,
allowances, marginal tax and net pay in integer fictional AST cents.

- Tier 1: retrieve an effective schedule field.
- Tier 2: calculate interacting payroll lines.
- Tier 3: request missing YTD earnings, resolve equal-authority salary conflicts,
  or request a schedule covering the payment date.

Frozen datasets contain 12 seeds, 30 training, 15 validation, 120 evaluation and
five transfer tasks. Seed namespaces and normalized fingerprints prevent split
overlap. Track B changes presentation while preserving rules; human approval and
freeze precede transfer evaluation. No real employee data is used.
Seeded tasks can be graded indefinitely without per-task human labels. This works
for the fictional rulebook; explanation usefulness and unwritten real-world
judgment are not computable by the payroll reference.

Normal weights: correctness 60%, fields 25%, action 10%, format 5%.
Blocked-task weights: diagnosis 55%, fields 25%, action 10%, explanation judge 5%,
format 5%. Invalid answers, unsafe computation and unjustified abstention score
zero. Substantive errors cannot exceed .20; formatting alone earns no reward.
Passing requires .975. The judge assesses only an otherwise correct blocking
explanation. Judge outages remain pending, without a substitute score.

The three tools are `read_document`, `lookup_rules` and bounded `calculate`.
The sandbox exposes `/tasks`, `/submit`, `/runs/{run_id}` and `/healthz`; public
responses exclude expected answers. Neon persists token-protected, idempotent runs.
Ten tasks per run, payload limits and caller quotas bound service use.

## Evaluation and cloud learning

Evaluation saves full transcripts, component scores, retries, tokens, cost and
latency. The main comparison uses the same 30 tasks and three rollouts per model;
Qwen additionally runs all 120 evaluation tasks and 12 tool tasks. Reported SD is
sample SD across complete rollout means. Programmatic baselines are separate.

Cloud defaults: Qwen2.5-0.5B-Instruct, LoRA rank 8, four grouped completions,
80 steps per beta (.001/.10). Validation selects beta before untouched held-out
comparison. GRPO uses group-relative advantages without a separate critic:

```text
A_i = (R_i − mean(R)) / (std(R) + ε)
J = E[(1/G) Σ_i (1/|o_i|) Σ_t
      {min(ρ_it A_i, clip(ρ_it, .8, 1.2) A_i) − β k3_it}]
ρ_it = πθ / πold; d_it = log πref − log πθ
k3_it = exp(d_it) − d_it − 1
```

The adapter-disabled frozen backbone is the initial reference; equivalence and
immutability are checked. Logs include reward, KL/penalty, entropy, length, tier
pass rates, components and equal-reward groups. Checkpoints stay in cloud storage.
No-key mode permits GPU inference but skips RL and judge study; eligible scores
remain pending. See [Colab instructions](docs/colab.md).

## Reviewer navigation

Start with [project explanation](docs/project-understanding.md),
[setup](docs/setup.md), [submission checklist](docs/submission-checklist.md) and
[Loom script](docs/loom-script.md).

Required references: [rules](rules/aster-payroll-v1.md),
[architecture](docs/architecture.md), [schema](docs/task-schema.md),
[reward specification](docs/reward-spec.md), [evaluation](docs/evaluation.md),
[RL report](docs/rl-report.md), [failure analysis](docs/failure-analysis.md),
[edge cases](docs/edge-cases.md), [trade-offs](docs/assumptions-and-tradeoffs.md),
[transfer](docs/advanced-track.md), [human review](docs/human-review.md),
[deployment/API](docs/deployment.md), [traceability](docs/traceability.md) and
[tool disclosure](docs/ai-usage.md). Loom URL remains pending recording.
