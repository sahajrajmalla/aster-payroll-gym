# Interview preparation: understanding and defending Aster Payroll Gym

Read this alongside the [beginner and professional guide](project-understanding.md)
and [Loom outline](loom-outline.md). You do not need to memorize every file. You
need to explain the problem, follow one task through the system, justify the
reward, and distinguish implemented machinery from measured results.

The present evidence is **implementation and lightweight QA**, including 178
passing tests and deterministic adversarial runs. Genuine model evaluations,
GPU training, judge reliability, independent human reviews, public sandbox
deployment, and Loom are pending. A published dashboard is not proof that an
experiment happened. Never fill that gap with a confident claim.

## 1. What was the assignment, and what did you build?

**Spoken answer:** “The assignment was to build a small AI training gym: repeatable
tasks, a trustworthy grader, model evaluation, a reinforcement-learning workflow,
and an external agent sandbox. This implementation uses fictional monthly payroll
because it combines arithmetic, document selection, and justified abstention.”

**Understand it:** The deliverable is the whole learning-and-measurement system,
not merely a payroll calculator or a chatbot. The eight phases connect task
schemas, generation, environments, rewards, evaluation, dashboard, RL, and sandbox.
Track B checks transfer to five unfamiliar presentations under unchanged rules.
The [traceability register](traceability.md) maps requirements to code, checks,
evidence, and remaining work.

## 2. Why this domain?

**Spoken answer:** “A narrow fictional jurisdiction gives us explicit rules and
deterministic answers. It is complex enough to require interacting rules, but
small enough to audit without relying on an LLM to decide financial correctness.”

**Understand it:** Real payroll law introduces changing legislation, exceptions,
and ambiguous authority. ASTER deliberately excludes those uncertainties. The
trade-off is limited external validity: passing this gym does not establish
readiness for real payroll. The written rules are separately versioned and win
if the code disagrees. Read [R1–R11](../rules/aster-payroll-v1.md) and
[assumptions and trade-offs](assumptions-and-tradeoffs.md).

## 3. How does one task travel through the system?

**Spoken answer:** “The generator creates evidence. A separate Python reference
computes its label. The model receives public evidence, returns strict JSON, and
the shared scorer grades it. Evaluation saves transcripts and metrics; the
dashboard displays saved evidence.”

**Understand it:** `Task` keeps private inputs and ground truth separate from
public documents. `parse_answer` validates the response before substantive
scoring. Evaluation, training, and sandbox use `scoring.score`; they do not own
different versions of correctness. The dashboard does not call a model.
Follow `task_from_inputs` in [generator.py](../src/aster_gym/generator.py),
`solve` in [reference.py](../src/aster_gym/reference.py), and
`score` in [scoring.py](../src/aster_gym/scoring.py).

## 4. Can you calculate a simple example yourself?

**Spoken answer:** “For a full January 2030 month, salary 300,000 cents, explicit
zero bonus, and zero 2030 year-to-date earnings before this payroll: gross is 300,000, contribution
15,000, taxable pay 275,000, tax 17,500, and net pay 267,500 cents.”

**Understand it:** R4 applies 5% because the annual earnings ceiling is available.
R5 subtracts contribution and the 10,000-cent allowance. R6 taxes only the
175,000-cent slice above the first 100,000 at 10%; no 20% slice applies. R7
subtracts contribution and tax from gross. Money uses integer cents and Decimal
ROUND_HALF_UP at specified stages, avoiding binary-float surprises. Recalculate
this from [R3–R7](../rules/aster-payroll-v1.md), then inspect `solve`.

## 5. What do the three tiers test?

**Spoken answer:** “Tier 1 retrieves an effective schedule field. Tier 2 combines
salary selection, proration, contribution, tax, and net pay. Tier 3 asks whether
the evidence actually permits a calculation.”

**Understand it:** The three primary traps are missing YTD earnings, conflicting
equally authoritative salary records, and no effective schedule. A 2031 date
cannot inherit 2030 rates. Conversely, an unsigned or stale distractor must not
cause refusal when valid authority exists. Tier labels are private metadata;
`solve` determines disposition from evidence, and scoring selects its rubric
from that disposition. See [task schema](task-schema.md), `generate_taskset`,
and R2, R8, R9, R11.

## 6. How trustworthy is the ground truth?

**Spoken answer:** “No LLM produces the labels. The independent Python reference
implements the written clauses using exact units and explicit authority rules.
Tests check hand-derived cases; independent human rule-fidelity review is still
required.”

**Understand it:** The generator calls the reference, but the reference imports
neither the generator nor a model library. That prevents labels being copied
from a model answer or generator-specific intermediates. Independence does not
make the reference infallible: a mistranscribed schedule could affect many labels.
The separate prose rules, boundary tests, and ten-task review worksheet address
that risk. Read [reference.py](../src/aster_gym/reference.py) and
[human review](human-review.md). Do not claim the pending audit is complete.

## 7. How are train and test separated?

**Spoken answer:** “There are frozen, disjoint seed namespaces and normalized
input fingerprints. Validation selects beta; the evaluation split is untouched
by optimizer updates and beta selection.”

**Understand it:** The corpus has 12 seeds, 30 training, 15 validation, 120
evaluation, and five authored transfer tasks. Fingerprints normalize unordered
evidence lists while preserving duplicate records that can change authority.
Retrieval hashes exclude irrelevant payroll amounts, preventing distractors from
disguising the same request. Different dates within four schedule-field groups
still share a narrow template; the report exposes this instead of claiming
independent rule diversity. Inspect `input_fingerprint` and `validate_splits` in
[generator.py](../src/aster_gym/generator.py), plus `split_manifest` in
[cloud/common.py](../src/aster_gym/cloud/common.py).

## 8. What is the difference between single-turn and tool mode?

**Spoken answer:** “Single-turn mode supplies documents in the prompt. Tool mode
supplies document IDs and makes the agent fetch evidence through `read_document`
or `lookup_rules`; `calculate` performs bounded arithmetic.”

**Understand it:** Tools are scoped to the task, with safe errors and limits.
The calculator interprets an allowlisted arithmetic AST rather than arbitrary
Python `eval`. The harness requires a successful reference read in tool mode
and bounds turns and calls. This catches unsupported guessing; it does not
prove an agent understood every document. The optional verifiers adapter has
per-rollout state, avoiding one global active task. Read
[tools.py](../src/aster_gym/tools.py), [environment.py](../src/aster_gym/environment.py),
and [evaluation.md](evaluation.md).

## 9. How does the sandbox protect expected answers?

**Spoken answer:** “Public response models allowlist task evidence and score
diagnostics. They never serialize the internal task or server-owned expected
answer. Runs require their token and have task, payload, and rate limits.”

**Understand it:** `/tasks`, `/submit`, `/runs/{id}`, and `/healthz` expose a
usable contract without an oracle endpoint. A caller's submitted answer may
appear in its transcript; legitimate rules may contain the same numeric value
as a retrieval answer. Leakage is therefore checked by provenance, not banning
every matching number. The repository is open and frozen task files contain
labels for audit; this is not secret-algorithm security. Inspect
[public.py](../src/aster_gym/public.py), [api.py](../src/aster_gym/api.py), and
[deployment.md](deployment.md).

## 10. Why are the reward weights and hard gates designed this way?

**Spoken answer:** “Correctness dominates. Ordinary work uses 60% exact,
25% fields, 10% appropriate action, and 5% format. Blocking work uses 55% exact
diagnosis, 25% fields, 10% action, 5% explanation judge, and 5% format.”

**Understand it:** These weights do not override gates. Invalid output, computing
money on unresolved work, and unjustified abstention earn zero. Rejected
substantive work is capped at `min(0.20, 0.25 × field_accuracy)`; format alone
earns nothing. Correct diagnosis with missing or unusable explanation caps at
0.20. This rewards useful progress without letting polished nonsense look
successful. Read [reward-spec.md](reward-spec.md) and `score` in
[scoring.py](../src/aster_gym/scoring.py).

## 11. Why include an LLM judge at all?

**Spoken answer:** “Python checks arithmetic and blocking diagnosis. The judge
assesses only whether an otherwise correct blocking explanation identifies the
issue clearly and gives a useful next action.”

**Understand it:** The constrained label is 0, 1, or 2. Adequate clarity yields
0.975 overall; excellent clarity yields 1.0; label zero triggers the 0.20 gate.
The judge cannot rescue wrong arithmetic or wrong blockers. Despite its 5%
weight, its gate can strongly penalize a correct trap, so false negatives are a
real risk. Fifteen human labels and three uncached judgments each measure
agreement and stability; these measurements remain pending. Read
[judge-v1.txt](../prompts/judge-v1.txt), [judge.py](../src/aster_gym/judge.py), and
[judge_study.py](../src/aster_gym/judge_study.py).

## 12. What happens when a provider or judge fails?

**Spoken answer:** “Infrastructure failure is recorded separately from a wrong
answer. A required unavailable judge leaves a pending score; it does not invent
a zero or silently use a cheaper rubric.”

**Understand it:** Eligible judgments use the same versioned cache across paths.
Evaluation retries transient failures within deadlines and budgets. Pending
judgments reuse the saved model answer, avoiding a fresh answer that changes
the experiment. Atomic journals preserve completed wrong answers too. Confirmed
spend and conservatively accounted spend are separate; unknown prices fail
closed. Training stops a batch if required reward is unavailable. Inspect
[providers.py](../src/aster_gym/providers.py), [eval.py](../src/aster_gym/eval.py),
and [judge.py](../src/aster_gym/judge.py).

## 13. How would you compare the models fairly?

**Spoken answer:** “Use the same frozen 30-task cohort, three stochastic rollouts
per task, the same mode and scorer, and save every transcript. Report mean and
sample SD of complete replicate means alongside coverage and costs.”

**Understand it:** The 120-task small-model run measures a larger distribution;
it must not be ranked directly against another model's 30-task run. Incomplete
replicates and operational failures remain visible, rather than becoming zero
reward. Three rollouts are a modest variability measurement, not a strong
confidence interval. Current baselines are handwritten policies with repeated
identical outputs; their zero SD is not model stability. See
[eval.py](../src/aster_gym/eval.py), [reporting.py](../src/aster_gym/reporting.py),
and [saved-results.md](saved-results.md).

## 14. What does GRPO actually do here?

**Spoken answer:** “For each prompt, the model samples four completions. The shared
scorer grades them. GRPO centers and scales rewards within that group and uses
those relative advantages to update the policy.”

**Understand it:** This is reinforcement learning from scored generated outputs,
not supervised copying of ground-truth strings. The implementation uses TRL's
clipped policy objective with group scaling and sequence-normalized token loss.
If every completion has equal reward, their reward advantages are zero; KL can
still contribute a gradient. A small model, sparse exact rewards, or truncated
JSON may make learning ineffective. Log identical-reward groups and inspect
actual completions. See [cloud/train.py](../src/aster_gym/cloud/train.py) and
[colab.md](colab.md). No learning outcome is claimed yet.

## 15. What are LoRA, the frozen reference, and KL doing?

**Spoken answer:** “LoRA trains small adapter matrices while freezing the model
backbone. Disabling fresh adapters provides the initial reference policy. KL
penalizes excessive departure from that reference.”

**Understand it:** The workflow checks enabled/disabled initial logits on a
probe and hashes all frozen parameters before and after training. These are
implemented checks whose GPU evidence is pending. The logged sampled estimator
is `k3 = exp(d) − d − 1`, where `d = log πref − log πθ`; it is not an exhaustive
exact KL calculation. `beta × KL` enters the optimization loss, not the shared
task reward. See `_hash_parameters`, `train_beta`, and the explicit GRPO
configuration in [cloud/train.py](../src/aster_gym/cloud/train.py).

## 16. Why two beta values, and how is the winner selected?

**Spoken answer:** “Beta 0.001 and 0.10 test weaker and stronger reference
regularization. Both start from the same policy. The highest validation mean
wins; a tie selects the larger beta before held-out testing.”

**Understand it:** Defaults are 80 optimizer steps per beta, temperature 0.8,
256 completion tokens, LoRA rank 8/alpha 16, and learning rate 5e-5. Checkpoints
are saved every 20 steps. A resource adjustment must be explicit and symmetric,
with a new configuration identity. Only after selection are initial and selected
policies compared on the evaluation split, three rollouts each. Do not select
beta from final-test performance. Follow `run_sweep` in
[cloud/train.py](../src/aster_gym/cloud/train.py) and [train.json](../configs/train.json).

## 17. How did you investigate reward gaming?

**Spoken answer:** “Deterministic attacks test format-only output, constant
payslips, and always abstaining. Parser and adversarial tests cover malformed
JSON, excess length, injection, and hidden-answer attacks. Actual trained-policy
gaming analysis awaits real completions.”

**Understand it:** Clause coverage is checked, but copying all valid citations
does not prove reasoning; extra citations do not earn extra reward. Ask whether
a high score can coexist with wrong content, unsupported refusal, or a useless
explanation. Preserve transcript IDs and original artifacts if a reward bug is
fixed, then rescore compared policies consistently under a new version. Negative
findings need inspected-output counts, not “no gaming exists.” Read
[failure-analysis.md](failure-analysis.md), [rl-report.md](rl-report.md), and
[tests](../tests).

## 18. What does the transfer test establish?

**Spoken answer:** “Five authored tasks change presentation while keeping the
rules. After independent human approval, their hash is frozen before model
calls. We compare model rankings and task-level reversals.”

**Understand it:** `freeze_transfer` requires genuine review rows; evaluation
checks the marker. Analysis uses matched three-model cohorts and average-rank
Spearman correlation, including ties. The drafts are AI-assisted and are not
already human-reviewed. Five tasks can reveal brittle document handling but
cannot support broad real-world generalization or a reliable population effect.
Do not report a correlation until complete records exist. Inspect
[advanced-track.md](advanced-track.md) and `freeze_transfer`,
`verify_transfer_freeze`, and `analyze_results` in
[evidence.py](../src/aster_gym/evidence.py).

## 19. Why cloud-only execution and safe result bundles?

**Spoken answer:** “The development computer must not train or load models.
Entry points verify recognized Colab before ML imports and require CUDA. Only
lightweight JSON evidence returns; checkpoints remain in cloud storage.”

**Understand it:** There is no CPU/MPS fallback or local override. Bundle import
rejects unsafe paths, symlinks, executable/pickle payloads, oversized archives,
bad checksums, incompatible versions, non-finite metrics, and split leakage.
It validates identities and recomputes summaries. Hashes detect mismatches;
they do not cryptographically attest that a particular model produced the
outputs. Preserve cloud logs and notebook evidence. Read
[cloud/guard.py](../src/aster_gym/cloud/guard.py),
[bundles.py](../src/aster_gym/bundles.py), and [colab.md](colab.md).

## 20. What did you personally do, and what would you improve?

**Spoken answer:** “Codex and AI subagents assisted with design, code, tests, and
documentation. The implementation is checked, but I will distinguish my own
reviews and executions from AI assistance. I will prioritize genuine held-out
evidence and fidelity audits before adding features.”

**Understand it:** Do not claim to have hand-written every module or successfully
trained a model. After actually performing reviews, execution, failure analysis,
deployment, and recording, explain those actions specifically. If asked about
unfamiliar code, locate the relevant function and trace it rather than bluffing.
The highest-value next work fills evidence gaps while keeping scope fixed. Read
[ai-usage.md](ai-usage.md), [submission-checklist.md](submission-checklist.md), and
[final-qa.json](final-qa.json).

## Practice before the interview

Use these exercises as understanding checks, not statements of completed work:

1. Explain the entire system in 60 seconds without opening a file.
2. Recalculate the example in question 4 from the prose rules. Then set YTD to
   1,190,000 cents: contribution becomes 500, taxable 289,500, tax 18,950,
   net 280,550. Explain why the cap concerns earnings, not contributions paid.
3. Remove YTD entirely. State the blocker, required clarification, and why a
   guessed number earns zero. Contrast that with a stale unsigned salary note.
4. Follow one saved baseline transcript through parser, gate, components, and
   dashboard. Label it a deterministic baseline, not a model failure.
5. Point to the public API models and show which internal fields are excluded.
6. Explain equal-reward groups, beta selection, and frozen-reference checks in
   plain language. State what still needs a real GPU run.
7. Once real model results exist, read one genuine failure aloud. Identify the
   violated clause, score, likely cause, and evidence-backed next step.

You are ready to explain the critical parts when you can answer: “What is being
measured?”, “Who decides correctness?”, “Can the reward be fooled?”, “Was this
actually run?”, and “What does this evidence fail to establish?” Practice the
answers in your own words. Confidence should come from tracing concrete evidence,
not memorizing terminology or promising selection for the job.
