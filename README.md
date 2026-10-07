# Aster Payroll Gym

A synthetic payroll gym featuring a separate written rulebook, a deterministic Python
ground truth, and an auditable scorer shared across evaluation, cloud reinforcement learning (RL), and a public API.

[Repository](https://github.com/sahajrajmalla/aster-payroll-gym) ·
[Dashboard](https://sahajrajmalla.com.np/aster-payroll-gym/) ·
[Sandbox](https://aster-payroll-gym.onrender.com/docs) ·
[Colab](https://colab.research.google.com/github/sahajrajmalla/aster-payroll-gym/blob/main/notebooks/aster_colab.ipynb)

**Implementation verified; experimental submission pending completion.** The Render/Neon
sandbox successfully processed Tier-2/3 submissions and maintained persistence following a restart. Qwen
completed 486 authentic Colab samples, though zero achieved a passing score. A subsequent Gemini execution completed
three rollouts across all 120 evaluation tasks, yielding 360 answers, a reward of .12049 ± .00732,
32 passing instances, and rewards ranging from 0 to 1. The evaluation judge returned 45 uncached responses,
which included one malformed output and 2 unstable examples out of 14 complete triples. Human
validity labels, transfer approvals and results, compatible frontier comparisons, and
empirical RL sweeps remain pending. Refer to the [checklist](docs/submission-checklist.md),
[measured evidence](docs/evaluation.md), and [QA](docs/final-qa.json) for comprehensive details.

## Public Sandbox Access

The public sandbox is accessible without cloning the repository. A Python 3 environment is required. The following pinned client script fetches a Tier-2 task, independently resolves it utilizing public documents, submits the solution, and retrieves the corresponding score:

```sh
curl -fsSL https://raw.githubusercontent.com/sahajrajmalla/aster-payroll-gym/8c96d28abe5fe6f5bd504d96d091ee6920f040d0/scripts/sandbox_smoke.py | python3 - https://aster-payroll-gym.onrender.com
```

Successful execution yields a `submission_verified` and `complete` status alongside a score of 1.0. The Render Free instance may necessitate a cold start; the client automatically retries bounded network failures. System utilization and associated costs are governed by limits of ten tasks per run, caller quotas, bounded payloads, and judge-call limits. Saved submissions are secured by run tokens, ensuring caller API keys and expected answers remain unexposed.

## Local Execution

Local execution requires Python 3.11 and the `uv` package manager. The following commands initialize the environment without installing model packages:

```sh
uv sync --locked --extra server
uv run --extra server aster-gym validate
uv run --extra server pytest -q
uv run --extra server ruff check .
uv run --extra server mypy src/aster_gym scripts
uv run --extra server aster-gym report --output site
uv run --extra server aster-gym serve
```

The API documentation is accessible at `http://127.0.0.1:8000/docs`. Refer to the [setup documentation](docs/setup.md) for instructions regarding the local dashboard, public sandbox, and result-import commands. Existing `.env` files must be preserved, and credentials along with private run receipts must be excluded from version control and recordings. The `cloud` extra package should exclusively be installed within Colab environments. Model entrypoints require hosted Colab and CUDA availability prior to model initialization.

## Task and Reward Architecture

ASTER-1.0 defines document authority, proration, bonuses, capped contributions,
allowances, marginal tax, and net pay evaluated in integer fictional AST cents.

- Tier 1: Retrieves an effective schedule field.
- Tier 2: Calculates interacting payroll lines.
- Tier 3: Requests missing YTD earnings, resolves equal-authority salary conflicts,
  or requests a schedule covering the payment date.

Frozen datasets consist of 12 seeds, 30 training, 15 validation, 120 evaluation, and
five transfer tasks. Seed namespaces and normalized fingerprints prevent split
overlap. Track B modifications alter presentation while preserving underlying rules; human approval and
code freeze precede transfer evaluation. Synthetic data is strictly utilized; no real employee data is incorporated.
Seeded tasks can be evaluated indefinitely without requiring per-task human labels. This methodology is effective
for the fictional rulebook; explanation utility and unwritten real-world
judgment fall outside the computable scope of the payroll reference.

Normal evaluation weights: correctness 60%, fields 25%, action 10%, format 5%.
Blocked-task evaluation weights: diagnosis 55%, fields 25%, action 10%, explanation judge 5%,
format 5%. Invalid answers, unsafe computations, and unjustified abstentions yield a score of
zero. Substantive errors are capped at a .20 penalty; formatting compliance alone yields no reward.
A passing threshold requires a score of .975. The judge evaluates solely an otherwise correct blocking
explanation. Instances of judge outages remain pending and do not receive a substitute score.

The three available tools are `read_document`, `lookup_rules`, and a bounded `calculate` function.
The sandbox exposes the following endpoints: `/tasks`, `/submit`, `/runs/{run_id}`, and `/healthz`; public
responses deliberately omit expected answers. Neon ensures persistence for token-protected, idempotent runs.
Service utilization is regulated by a limit of ten tasks per run, payload constraints, and caller quotas.

## Evaluation and Cloud Learning

The evaluation system records full transcripts, component scores, retries, token usage, cost, and
latency. The primary comparative analysis employs the identical 30 tasks and three rollouts per model;
Qwen additionally processes all 120 evaluation tasks alongside 12 tool tasks. The reported standard deviation (SD) is
calculated as the sample SD across complete rollout means. Programmatic baselines are maintained separately.

Cloud configuration defaults: Qwen2.5-0.5B-Instruct, LoRA rank 8, four grouped completions,
and 80 steps per beta (.001/.10). The validation phase selects the optimal beta preceding the untouched held-out
comparison. GRPO implements group-relative advantages without requiring a separate critic:

```text
A_i = (R_i − mean(R)) / (std(R) + ε)
J = E[(1/G) Σ_i (1/|o_i|) Σ_t
      {min(ρ_it A_i, clip(ρ_it, .8, 1.2) A_i) − β k3_it}]
ρ_it = πθ / πold; d_it = log πref − log πθ
k3_it = exp(d_it) − d_it − 1
```

The adapter-disabled frozen backbone serves as the initial reference; equivalence and
immutability are rigorously verified. Logging metrics include reward, KL/penalty, entropy, length, tier
pass rates, components, and equal-reward groups. Checkpoints are securely stored in cloud storage.
No-key mode enables GPU inference while bypassing RL and judge study; eligible scores
remain in a pending state under this mode. Refer to the [Colab instructions](docs/colab.md) for further details.

## Documentation Reference

Initial configuration and project overview are detailed in the [setup](docs/setup.md) and [submission status](docs/submission-checklist.md) documents.
Further technical references include: [architecture](docs/architecture.md), [schemas](docs/task-schema.md),
[rules](rules/aster-payroll-v1.md), [reward](docs/reward-spec.md),
[evaluation](docs/evaluation.md), [RL](docs/rl-report.md),
[failures](docs/failure-analysis.md), [edge cases](docs/edge-cases.md),
[trade-offs](docs/assumptions-and-tradeoffs.md), [transfer](docs/advanced-track.md),
[deployment/API](docs/deployment.md), [Colab](docs/colab.md),
[review procedure](docs/human-review.md), and [requirements](docs/traceability.md).
