# Loom recording guide: a calm, evidence-based ten-minute walkthrough

**Current status: implementation and lightweight QA are complete; final evidence is not.**
The public dashboard currently contains real *programmatic adversarial baselines*,
not genuine model evaluations. Colab training, measured model comparisons, human
reviews, transfer findings and the public Render/Neon sandbox remain pending.
Rehearse now. Record the final submission after those requirements are met. Do not
read a placeholder as a result or call a baseline transcript a model failure.

The assignment asks for an **8–12 minute video**, including **at least three minutes
on reward design**, **at least one minute reading a genuine failed model transcript
aloud**, and a **live independently hosted sandbox request**. The schedule below is
ten minutes. Leave enough time for the actual failure; that requirement cannot be
replaced by a summary of tests.

## A sixty-second introduction you can practise today

> “I built Aster Payroll Gym: a small training and evaluation environment for an
> agent that handles fictional payroll tasks. The goal is to test whether the
> agent can apply a written rule correctly, use tools when needed, and recognise
> when missing or conflicting evidence makes a calculation impossible.
>
> The important design choice is that Python calculates the expected answer from
> separate, versioned rules. An LLM never creates the ground truth. A shared
> scorer grades evaluation, cloud training and the public sandbox, so these paths
> cannot quietly use different definitions of success.
>
> The system includes three difficulty tiers, adversarial baselines, saved
> transcripts, a dashboard, a cloud-only reinforcement-learning workflow and a
> submission API. I used AI assistance and documented it. Implementation and QA
> are complete; experimental claims must come from the recorded runs.”

The last sentence is accurate now. For the final video, replace it with the actual
completed experiment status and disclose anything still missing. Never imply that
you personally typed every line. Own the decisions, understand the code and explain
the evidence honestly.

## Preflight: complete this before pressing Record

1. Complete `docs/human-review.md`: independent task calculations, rule-fidelity
   checks, judge labels, transfer approval and review of 5–10 actual failures.
2. Freeze transfer tasks **before** evaluating them. Execute genuine model runs and
   both Colab KL sweeps; import the JSON bundle and regenerate reports. A failed or
   partial run stays failed or partial. Do not manufacture missing curves.
3. Publish the updated dashboard. Deploy the sandbox using `docs/deployment.md`;
   test a correct Tier-2/3 submission from an external client and verify persistence
   after restart. The localhost smoke is useful QA, not hosted deployment evidence.
4. Fill the required evidence notes below. Select one actual model failure with its
   task, rollout and run ID. Check that its reward and deciding clause are visible.
5. Open these screens, in order: README; written rules; reward specification;
   dashboard; selected failure transcript; RL/held-out and transfer reports;
   sandbox client; submission checklist. Increase font size and close unrelated tabs.
6. Warm the public sandbox by visiting `/healthz`. Free hosting may take time
   to wake after sleeping. Hide `.env`, API keys, connection strings and
   `run_token`. They are not useful reviewer evidence.

Required recording notes — replace every blank with a saved-record reference:

- Main comparison: three model names, the same 30 tasks and three rollouts; means,
  sample SDs, coverage, taskset/reward versions and actual spend.
- Failure: run/task/rollout IDs, exact output, correct diagnosis from the rule,
  gate/components, and what the example teaches.
- Judge: 15 independent labels, three uncached ratings per example, agreement,
  repeated instability and one disagreement if present.
- Rule audit: reviewed count and disagreements before/after corrections.
- RL: actual GPU/steps, both beta runs, validation selection, initial/after held-out
  mean±SD, reference proof, five curve observations and reward-gaming review count.
- Transfer: frozen task hash/date, three rankings, Spearman agreement, any task-level
  reversals and one concrete failure or success.
- Public sandbox URL; actual round-trip/restart checks; four final submission links.

These are **required evidence slots**, not proposed findings. The saved artifacts
must support every number you say. If a slot remains empty, describe it as pending
and understand that the final submission still has an unmet requirement.

## The timed script

### 0:00–1:00 — task and domain

Screen: README, then the beginning of `rules/aster-payroll-v1.md`.

> “The assignment was to build a gym where an AI can practise tool-shaped work,
> receive an auditable reward and be evaluated before and after training. I chose
> one fictional payroll jurisdiction because it combines precise arithmetic with
> document authority and decisions about missing information.
>
> Tier 1 retrieves a schedule value. Tier 2 combines proration, contribution,
> allowance, marginal tax and net pay. Tier 3 tests whether the agent notices
> missing year-to-date earnings, equally authoritative salary conflicts, or no
> effective schedule. Those cases require a specific request for information.
>
> This is a controlled synthetic benchmark. It does not establish competence with
> real payroll law. The rule text is the authority, and Python is its executable
> reference.”

Point briefly to the corpus: 12 seed, 30 training, 15 validation, 120 evaluation
and five transfer tasks. Do not describe seed separation as proof that every
template is novel: schedule-retrieval tasks share a small set of rule patterns.

### 1:00–4:30 — reward design: do not rush this section

Screens: R1/R4/R8/R9, `docs/reward-spec.md`, baseline dashboard and judge/audit report.

> “The first part of the reward is a deterministic verifier. All money uses integer
> cents and Decimal half-up rounding. The reference decides both the amounts and
> whether a calculation is justified. Each score component records clause IDs and
> a diagnostic code, so I can trace a penalty to a written rule.
>
> For ordinary answers, exact correctness carries 60 percent, field accuracy
> 25 percent, appropriate action 10 percent and format five percent. Correctness
> includes the required substantive content and relevant citations. Field credit
> supplies a smaller signal when some financial fields are right.
>
> For blocked work, structured diagnosis carries 55 percent, field or blocker
> accuracy 25 percent, appropriate action 10 percent, explanation quality five
> percent and format five percent. The disposition comes from the evidence; the
> tier label itself never tells the agent to refuse.
>
> These weights sit behind hard gates. Invalid JSON scores zero. Inventing a
> calculation when a required input is missing scores zero. Refusing a solvable
> task also scores zero. Incorrect substantive answers cannot earn more than
> 0.20, and nice formatting alone cannot earn a positive reward. An agent should
> not improve its score by becoming confidently wrong or by always refusing.”

Spend about 40 seconds showing one solvable case and one blocker. Explain which
clauses decide the action, rather than reading code line by line.

> “Python controls arithmetic and rejection. The LLM judge only assesses whether
> an otherwise correct blocking explanation communicates the issue and a useful
> next action. It returns a constrained zero, one or two label. It cannot override
> a wrong calculation. A correct trap with an adequate explanation can score
> 0.975; an excellent explanation can score one. I use 0.975 as the pass threshold.
>
> A missing judge response is pending, not a fabricated zero or a different rubric.
> The judge prompt, model and scoring versions are part of its cache identity.
> I checked reliability with independent human labels and repeated uncached
> judgments. Here are the measured agreement and instability results.”

Read the **actual** judge numbers and explain any disagreement. If not executed,
say “That measurement is pending,” and do not present it as completed validation.

> “I also attacked the reward with format-only, constant-payslip and always-abstain
> policies. These are handwritten baselines. Their repeated SD is zero because
> the outputs are identical, not because an AI is perfectly stable. They test
> whether superficial strategies receive credit. The rule-fidelity audit is a
> separate human check that the calculator actually matches the written authority.”

Show the saved baseline rows and **actual** human audit count. Explain that cache
stability, automated tests and agreement with one human are different evidence.

### 4:30–5:45 — read one genuine model failure aloud

Screen: dashboard heatmap → selected saved model transcript → score components.

Use at least a full minute on this example. Read the relevant task evidence and
the model's actual answer aloud, not just the summary. State the run/task/rollout ID.

> “This is an actual response from [model], not a test fixture or adversarial
> baseline. The task evidence says [read the relevant lines]. The model answered
> [read its exact substantive answer]. Clause [ID] requires [explain the rule].
> The scorer therefore recorded [actual gate or component] and [actual score].
>
> This example shows [specific arithmetic, authority, calibration or tool failure].
> The reward [caught it / missed this aspect]. My next change would be [bounded
> remedy supported by the example]. One failure illustrates a mechanism; it does
> not establish its frequency. The failure report records the reviewed denominator.”

Replace brackets before recording. If a real failure has not been collected, this
section cannot be satisfied with the existing baseline transcript.

### 5:45–7:30 — training, held-out evidence and reward gaming

Screen: five actual dashboard curves, then `docs/rl-report.md`.

> “Training runs only in Colab on a CUDA GPU. This laptop never trains, loads model
> weights or performs small-model inference. The policy is Qwen2.5-0.5B-Instruct
> with LoRA. GRPO samples four completions per prompt and compares their rewards
> within the group. A KL penalty discourages excessive movement from the frozen
> initial policy. I compare beta 0.001 and 0.10, starting from the same policy.
>
> These plots show reward, KL divergence, entropy, completion length and pass rate
> by tier. I interpret them together: reward going up is insufficient if the model
> collapses to repeated answers or improves only its formatting. Equal-reward
> groups also carry no reward preference, so I log their frequency.
>
> The reference proof checks initial equivalence and unchanged reference weights.
> I selected beta using validation only. Then I compared the initial and selected
> policies on untouched evaluation tasks, with three rollouts each.”

Read the **actual** steps, selected beta, before/after mean±SD and curve observations.
Report negative or inconclusive results directly. Include how many outputs were
examined for reward gaming and what was found. Do not claim “no gaming possible”
when the result is “none found in the reviewed sample.”

### 7:30–9:00 — public sandbox round trip

Screen: external client using the public Render URL; no localhost server.

> “This is the independently hosted sandbox. The client fetches source documents,
> submits its own answer and retrieves the persisted run. The response contains
> scores, component diagnostics and clauses; it does not reveal expected answers.
> The same scorer is used here, in evaluation and in training.”

Show a correct Tier-2/3 submission and retrieval. Then briefly show the malformed
answer quickstart scoring zero, if time permits. State limits: ten tasks per run,
bounded payloads and caller quotas. Show idempotent same-answer retry if practised;
do not change the answer within a locked run. Explain a pending judge response
honestly rather than retrying until an apparently good score appears.

### 9:00–10:00 — transfer, limitations and reproduction

Screens: transfer report, cost/latency section, README commands and checklist.

> “Track B keeps the rules unchanged while changing the task presentation. Five
> reviewed tasks were frozen before model outcomes. The generated-cohort ranking
> was [actual ranking], and transfer was [actual ranking]. Their agreement was
> [actual statistic]. [Describe one observed reversal or failure.] Five tasks are
> a diagnostic sample, not proof of broad generalisation.
>
> The reports distinguish model mistakes from provider failures and pending
> scores. They show actual confirmed spend separately from conservative accounting
> for uncertain calls. The important limits are the synthetic rule set, repeated
> template structure, small transfer sample and [actual remaining limitation].
>
> A clean clone can run lightweight validation and tests. Model work belongs in
> the Colab notebook, and only checked JSON evidence comes back. The repository,
> dashboard, sandbox and this video are linked in the submission. AI assistance
> and incomplete evidence are disclosed.”

## Exact commands and API shapes for the rehearsal

Safe local commands from the repository root; none invokes a model:

```sh
uv run aster-gym validate
uv run pytest -q
uv run aster-gym analyze
uv run aster-gym report --output site
python scripts/quickstart.py https://YOUR-ACTUAL-SANDBOX.onrender.com
```

The last command requires the deployed URL. It intentionally submits `{}` and
demonstrates rejection, not correct professional work. For a correct demo, your
independent client or an external agent must answer from the issued **public** prompt
and documents. Do not import `aster_gym.reference` into the sandbox client.

Requests, in order:

```text
GET /healthz
GET /tasks?tier=2&n=1
POST /submit
  Content-Type: application/json
  X-Run-Token: <run_token from task issuance; keep private>
GET /runs/<run_id>
  X-Run-Token: <same token>
```

`POST /submit` requires the exact issued IDs and one answer for **every** issued task:

```json
{
  "run_id": "<issued run_id>",
  "answers": [
    {"task_id": "<issued task id>", "answer": {}}
  ]
}
```

The empty answer above is deliberately invalid. A real answer has exactly
`decision`, `result`, `issue_codes`, `missing_fields`, `citations` and `explanation`.
An answered payslip uses five integer-cent result fields. A blocked task uses
`decision: needs_information`, `result: null`, complete blocker/missing sets and
a useful explanation of at most 400 characters. Use the issued answer instructions
and `docs/task-schema.md`, not invented field names.

An issued token protects that run. Identical retries are allowed; changed answers
return 409. HTTP 202 means scoring is pending, not successful or incorrect. Preserve
the first payload and resubmit it unchanged after the judge recovers. Do not hammer
the API: run creation is limited to five per minute per direct client address.

## Rehearsal and recovery

Practise the reward explanation first; it is the project's strongest design story.
Say “a small part of the score” before “partial credit,” and “distance from the
starting policy” before “KL divergence.” Pause after showing a number. If asked
something uncertain, say which artifact you would inspect rather than guessing.

Do one timed rehearsal with no recording. Then one private recording. Watch it once
for secret exposure, unreadable text, unsupported claims and the two time minima.
Use your own words; the script is a guide, not something to recite rigidly.

If the public service fails, explain the exact observed failure and repair it before
the final recording. A saved successful run may illustrate earlier evidence but
must be labelled replayed. A localhost demo or offline recording does **not** meet
the independent live sandbox requirement. If GPU work fails, keep its failure
metadata and checkpoints in the cloud; describe the resulting partial evidence.
Never fill a missing curve or finding to make the video look complete.
