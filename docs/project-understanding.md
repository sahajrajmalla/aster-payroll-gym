# Understand Aster Payroll Gym

Read this first, then rehearse the [interview reminder](interview-cheatsheet.md).
Current completion evidence is in [the submission checklist](submission-checklist.md).

## Simple explanation

Imagine a school for an assistant learning payroll. It needs a rulebook, practice
questions, an answer checker, fair marks and a report card. Aster Payroll Gym is
that school. Aster is fictional; no real employee data or country's law is used.

Some questions ask the assistant to find a number. Others require calculations.
Some cannot be answered safely because information is missing or contradictory.
The assistant must explain precisely what it needs. Guessing is wrong, and refusing
every question is also wrong.

Python checks the answer using the written rules. We save each attempt so a
reviewer can see what happened. Training means changing the assistant using these
marks. Evaluation means testing it without changing it. The final test uses
questions excluded from training and tuning.

The assignment asked for this complete school: tasks, tools, scoring, model
comparisons, training, dashboard and a publicly accessible submission service.
Code and real deterministic baseline evidence exist. Model experiments and human
reviews need their own genuine records; a working training script is not a result.

## Professional explanation

> Aster Payroll Gym is a synthetic environment for evaluating and training agents
> on a narrow monthly-payroll workflow. A separate versioned rulebook defines
> correctness; an independent Python reference computes expected results. Tasks
> cover retrieval, interacting calculations and evidence gaps requiring
> clarification. One clause-auditable scorer serves evaluation, cloud RL and the
> external sandbox. The design prioritizes reward integrity, held-out evidence
> and reproducibility.

Payroll was chosen because effective dates, document authority, arithmetic and
missing evidence interact naturally. Fictional rules make correctness checkable.
Success here does not establish competence with actual payroll law.

The eight phases are task schemas/seeds, deterministic generation, environments,
reward design, evaluation, dashboard, cloud RL and sandbox. The sole advanced
track is transfer testing: five reviewed tasks change presentation, not rules.

## Follow an answer through the code

1. [Rules R1–R11](../rules/aster-payroll-v1.md) define authority. If code disagrees,
   the rulebook wins.
2. [Generator](../src/aster_gym/generator.py) creates evidence; disjoint seeds and
   normalized fingerprints detect split overlap.
3. [Reference](../src/aster_gym/reference.py) computes private answers without
   calling a model or importing the generator.
4. [Tools](../src/aster_gym/tools.py) provide scoped document/rule reads and bounded
   arithmetic. Tool mode obtains required reference evidence through tools.
5. [Parser](../src/aster_gym/parser.py) checks one strict six-key JSON answer.
6. [Scorer](../src/aster_gym/scoring.py) applies clauses, gates, partial credit and
   eligible explanation judging.
7. [Evaluation](../src/aster_gym/eval.py) saves responses, scores, usage and costs;
   [sandbox](../src/aster_gym/api.py) uses the same scorer with public-only responses.

## Worked example

All amounts below are integer cents. Payment: 2030-04-30. Latest applicable signed
salary: 300,000. Paid days: 30/30. Bonus: 20,000. Prior 2030 pensionable earnings:
1,190,000. These are teaching inputs, not a model result.

- R3 gross: `300,000 × 30/30 + 20,000 = 320,000`.
- R4 remaining ceiling: `1,200,000 − 1,190,000 = 10,000`.
  Contribution: `min(320,000, 10,000) × 5% = 500`.
- R5 taxable: `320,000 − 500 − 10,000 = 309,500`.
- R6 tax: `100,000 × 0% + 200,000 × 10% + 9,500 × 20% = 21,900`.
- R7 net: `320,000 − 500 − 21,900 = 297,600`, or **2,976.00 AST**.

Removing YTD earnings makes the contribution ceiling unknown. Return
`needs_information`, `result: null`, issue `MISSING_INPUT`, missing field
`ytd_pensionable_cents`, citations R4/R9/R10 and a specific request for prior
2030 earnings. Do not invent zero. A newer unsigned salary also cannot override
an applicable signed one.

## Why the reward deserves trust

Ordinary weights are .60 correctness, .25 fields, .10 action and .05 format.
Blocked-task weights are .55 diagnosis, .25 fields, .10 action, .05 explanation
and .05 format. Invalid output, unsafe trap computation and unjustified refusal
score zero; rejected substantive answers cannot exceed .20. Formatting alone
cannot pass. Passing requires at least .975.

Python decides arithmetic and blockers. The constrained judge evaluates only the
usefulness of an otherwise correct blocking explanation. It cannot override
Python; outages remain pending. Its reliability requires independent labels and
three uncached repeat judgments.

Training uses four completions per prompt, relative group rewards and a KL penalty
against a frozen initial policy. Validation selects between two beta values;
untouched evaluation measures before/after. Inspect reward, KL, entropy, length,
tier pass rates and actual outputs together. Rising reward alone is insufficient.

For deeper detail use [reward spec](reward-spec.md), [RL report](rl-report.md),
[glossary](glossary.md) and [submission walkthrough](submission-walkthrough.md).
