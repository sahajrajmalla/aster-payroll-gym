# Aster Payroll Gym — interview reminder

Use this for rehearsal. Actual experiment fields below are deliberately empty
until genuine records exist. Current engineering QA is green; full submission
evidence is pending.

## Say this in 30 seconds

> Aster Payroll Gym is a synthetic environment for evaluating and training AI
> agents on a narrow payroll task. A separate written rulebook defines correctness,
> Python supplies independent ground truth, and one clause-auditable reward scores
> evaluation, training and sandbox submissions. The hard part is distinguishing
> correct work from polished errors and justified clarification from blanket
> refusal. Training and small-model inference run only in Colab.

## Remember these facts

- **Domain:** fictional monthly payroll; integer AST cents; Decimal half-up.
- **Authority:** ASTER-1.0, clauses R1–R11. Prose wins over code.
- **Difficulty:** retrieval → interacting payroll calculations → evidence blockers.
- **Three traps:** absent YTD, equal-authority salary conflict, no effective schedule.
- **Data:** 12 seed / 30 train / 15 validation / 120 evaluation / 5 transfer.
- **Separation:** disjoint seeds and normalized inputs; repeated Tier-1 templates
  remain a disclosed limitation, even when dates differ.
- **Tools:** `read_document`, `lookup_rules`, bounded `calculate`.
- **Normal weights:** exact .60 / fields .25 / action .10 / format .05.
- **Blocker weights:** diagnosis .55 / fields .25 / action .10 / judge .05 / format .05.
- **Gates:** invalid, unsafe computation and unjustified abstention = 0.
  Substantive mismatch ≤ .20. Format alone gives no reward.
- **Judge:** communication quality of an otherwise correct blocker only;
  label 0 activates a .20 cap; outage = pending, never fabricated score.
- **Passing:** score ≥ .975. Perfect ordinary work = 1.0.
- **RL:** Qwen2.5-0.5B-Instruct, LoRA rank8/alpha16, GRPO group4, temperature.8,
  completion256, learning rate5e-5, defaults80 steps/beta, checkpoint20.
- **Reference:** frozen initial backbone with adapters disabled; initial logit probe
  and full frozen-parameter hashes become evidence only after cloud execution.
- **Beta:** .001 and .10. Choose with validation only; report untouched held-out
  initial/after results. Do not assume larger or smaller beta wins.
- **Five curves:** reward / KL / entropy / completion length / per-tier pass rate.
- **Advanced track:** B only, five authored transfer drafts; human review, freeze
  and findings pending. Measure rankings/failures with a small-sample caveat.
- **Sandbox:** `/tasks`, `/submit`, `/runs/{run_id}`, `/healthz`;
  explicit public DTOs omit private labels, seeds and trap tags.
- **Evidence today:** 178 passing tests, green CI, real deterministic baselines,
  positive local API smoke; actual model/RL/public-sandbox evidence pending.

## Explain this worked example

2030 full-month salary300,000 + bonus20,000 → gross320,000 cents.
YTD1,190,000 leaves10,000 ceiling → contribution500.
Allowance10,000 → taxable309,500.
Tax0 + 20,000 + 1,900 =21,900 → net297,600 cents =2,976.00 AST.
Missing YTD → `needs_information`, no monetary result, cite R4/R9/R10.

## Fill these from saved evidence before recording

- Three model names, same-cohort means±SD, coverage: **PENDING**.
- Selected beta and validation rationale: **PENDING**.
- Held-out before/after, coverage and uncertainty: **PENDING**.
- Judge human agreement and repeat stability: **PENDING**.
- One real failed model transcript ID and violated clause: **PENDING**.
- Reward-gaming search count, finding or honest negative result: **PENDING**.
- Transfer ranking agreement and task-level failures: **PENDING**.
- Public sandbox URL, successful independent round trip and restart test: **PENDING**.
- Actual cost, latency and limitations: **PENDING**.

## When a reviewer challenges you

Explain the written clause, show the actual transcript/score, and state the limit
of the evidence. If you do not know, say what you would inspect; do not invent a
finding. If a guard or test fails, preserve the failure and diagnose it openly.

Disclose AI assistance plainly. Credit Codex for assistance with implementation,
tests and documentation. State your actual role in understanding, independently
reviewing, executing cloud experiments, validating deployment and presenting
results only after you have performed those actions.

Full explanations: [project-understanding.md](project-understanding.md),
[interview-preparation.md](interview-preparation.md),
[submission-walkthrough.md](submission-walkthrough.md) and
[loom-script.md](loom-script.md).
