# Understand the project

## Simple explanation

Imagine a school for a payroll assistant. The rulebook is its textbook, synthetic
tasks are its questions, Python is the answer checker and the dashboard is its
report card. Some questions have a numerical answer; others need more evidence.
Guessing missing information is wrong, and refusing every question is also wrong.

Evaluation tests the assistant without changing it. Training changes it using
rewards. A final held-out test uses questions excluded from training and tuning.
The sandbox lets someone else submit an answer and retrieve its saved score.
A passing sandbox demonstration proves the service works, not that a model learned.

## Professional explanation

Aster Payroll Gym evaluates and trains agents on one fictional monthly-payroll
workflow. A separate versioned rulebook defines correctness; an independent Python
reference computes private answers. One clause-auditable scorer serves model
evaluation, cloud RL and the public sandbox. The assignment requires eight phases:
tasks, generation, environments, rewards, evaluation, dashboard, RL and sandbox,
plus one advanced track. This project selects Track B, transfer testing.

## Follow one answer

1. [Rulebook](../rules/aster-payroll-v1.md): R1–R11 define correct behaviour.
2. [Generator](../src/aster_gym/generator.py): creates synthetic evidence with split-safe seeds.
3. [Reference](../src/aster_gym/reference.py): computes expected answers independently.
4. [Tools](../src/aster_gym/tools.py): scoped document/rule reads and bounded arithmetic.
5. [Parser](../src/aster_gym/parser.py): validates one strict JSON answer.
6. [Scorer](../src/aster_gym/scoring.py): applies gates, partial credit and eligible judging.
7. [Evaluator](../src/aster_gym/eval.py) saves attempts; [API](../src/aster_gym/api.py)
   uses the same scorer and [storage](../src/aster_gym/store.py) persists private runs.

## Example to explain in your interview

These are teaching inputs, not model results: salary 300,000 cents, bonus 20,000,
30/30 paid days and prior pensionable earnings 1,190,000.

- Gross: 300,000 + 20,000 = 320,000 (R3).
- Remaining annual ceiling: 1,200,000 − 1,190,000 = 10,000.
  Contribution: 10,000 × 5% = 500 (R4).
- Taxable: 320,000 − 500 − 10,000 allowance = 309,500 (R5).
- Tax: 200,000 × 10% + 9,500 × 20% = 21,900 (R6).
- Net: 320,000 − 500 − 21,900 = 297,600 cents (R7).

Remove YTD earnings and the answer must become `needs_information`, with
`result: null`, the exact missing field and clauses R4/R9/R10. Inventing zero fails.

## Five interview answers

**Why payroll?** Dates, evidence authority, arithmetic and missing inputs interact,
while fictional rules allow deterministic checks without private data.

**Why trust the reward?** Python decides facts and blockers. Wrong substance is
capped at .20; invalid or unsafe answers score zero. The LLM judge affects only 5%
of eligible explanations and cannot override rejection. Outages remain pending.

**Why GRPO and KL?** Four answers compete within each prompt; group-relative rewards
avoid a separate critic. KL limits movement from the frozen initial policy. Equal
rewards yield zero advantage, so reward and update evidence must both be inspected.

**How is leakage prevented?** Disjoint seed namespaces, normalized fingerprints,
private expected-answer models and separate public API schemas.

**What is unfinished?** Consult the [checklist](submission-checklist.md). Code,
setup checks, inference and training are distinct. Claim improvement only after
real held-out comparisons, not because a training script exists.
