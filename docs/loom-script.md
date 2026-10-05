# Ten-minute Loom script

Record after the [submission checklist](submission-checklist.md) has genuine
experiment, review and public-sandbox evidence. Target 10 minutes; required range
is 8–12. Spend at least three minutes on rewards and one minute reading a genuine
model failure. This outline allocates 3:30 and 1:15 respectively.

## Prepare the screen

Open the README, rulebook, reward spec, dashboard, one real failed transcript,
RL/transfer reports and public sandbox client. Use readable zoom and a quiet room.
Warm the free hosted service before recording. Close `.env`, provider consoles,
Colab Secrets, database credentials and tabs containing run tokens.

Fill these slots from saved results before recording:

- Three models: [names], comparison [means±SD], coverage [counts].
- Judge: [human agreement], [repeat instability], [review disagreements].
- Failure: [run/task/rollout ID], [response], [violated clause].
- RL: [actual steps], selected beta [value], validation [reason], held-out
  [initial→selected means±SD], [curve observations], [gaming sample/findings].
- Transfer: [generated/transfer rankings], [agreement], [one task failure/reversal].
- Public sandbox: [URL], [correct round trip], [persistence-after-restart evidence].
- Resources: [confirmed cost], [latency], [GPU/runtime], [remaining limitations].

Empty slots are pending evidence. Do not turn planned behavior into measured claims.

## 0:00–1:00 — problem and scope

Screen: README and a task.

> “Aster Payroll Gym tests an assistant on one fictional monthly-payroll workflow.
> The assistant must read authoritative documents, calculate correctly and ask for
> specific information when the evidence is insufficient. I chose this domain
> because rules, dates, arithmetic and evidence gaps interact, while a synthetic
> jurisdiction makes correctness deterministic and avoids private customer data.
> The project connects tasks, shared scoring, model evaluation, cloud training,
> a dashboard and an independent submission service.”

Show a retrieval question, a payslip calculation and one evidence blocker. Explain
that difficulty labels are not sent as answer hints. State actual completion status.

## 1:00–4:30 — reward design

Screens: rulebook R2/R4/R8/R9/R10, reward spec and baseline dashboard.

> “The written rulebook is authoritative. Python independently computes expected
> results; an LLM does not generate ground truth. Each scoring component returns
> clause references and diagnostics, so I can connect a penalty to its rule.”

Show the ceiling example: gross320,000, remaining ceiling10,000, contribution500,
taxable309,500, tax21,900, net297,600 cents. Then remove YTD.

> “Without prior pensionable earnings, the contribution is unresolved. The useful
> answer names that missing field and requests it. Guessing zero is wrong. Refusing
> a solvable task is also wrong.”

Explain weights deliberately, giving each screen time:

> “For ordinary work, correctness is 60%, field accuracy 25%, appropriate action
> 10% and format 5%. For blocked work, diagnosis is 55%, fields 25%, action 10%,
> explanation 5% and format 5%. Hard gates override the sum: invalid output,
> unsupported computation and unnecessary refusal receive zero. Substantive
> errors cannot exceed .20; passing requires .975. Formatting alone cannot pass.”

Show a malformed output and one substantive mismatch, then actual baseline scores.

> “Format-only, constant-answer and always-abstain policies test obvious shortcuts.
> These are deterministic attack policies, not model benchmarks. Their results
> support these particular defenses rather than proving every possible hack absent.”

Explain the constrained judge and actual review evidence:

> “Python first verifies blockers and citations. The remote judge only assesses
> whether a correct explanation communicates the issue and a useful next action.
> It cannot calculate payroll or override rejection. Identical eligibility and
> caching rules apply across evaluation, training and sandbox. Outages stay pending,
> so the rubric is not silently simplified.”

Read actual human agreement and three-repeat stability. Explain one disagreement,
if observed, and show the independent rule-fidelity review count. Note that stable
judgments can still be wrong. This section must reach at least three minutes.

## 4:30–5:45 — read one genuine failure

Screen: a saved model transcript with run/task/rollout identifiers.

Read its input and actual response aloud, then the score component and deciding
clause. Spend at least a minute on this example.

> “The model returned [actual response]. Clause [ID] requires [specific behavior].
> The scorer recorded [actual gate/component]. This failure shows [observed issue].
> The operational status is [complete/error/pending]; an outage is not a wrong
> answer. My response to the failure was [actual investigation or change].”

Use a real model error rather than a test fixture or deterministic baseline. If
all inspected model outputs passed, report that honestly and find a genuinely
failed configuration/run rather than manufacturing a failure.

## 5:45–7:30 — cloud RL and held-out evidence

Screens: five actual curves and before/after comparison.

> “All model work ran in Colab. Qwen2.5-0.5B-Instruct uses LoRA and GRPO: four
> sampled completions receive relative group advantages. A KL penalty discourages
> excessive movement from the frozen initial policy. Both beta runs start from
> the same policy; validation selects beta before untouched held-out evaluation.”

Read actual completed steps, beta choice and before/after mean±SD. Show initial
reference equivalence and unchanged parameter hashes. Explain reward, KL, entropy,
completion length and tier pass rate together; include equal-reward group fraction.

> “Rising reward alone is not enough. I inspected [count] outputs for [attacks]
> and observed [actual findings]. [Describe a negative or inconclusive result
> honestly.] Checkpoints remain in cloud storage; checked JSON evidence drives
> the dashboard.”

## 7:30–9:00 — live public sandbox

Screen: an independent client targeting the actual Render URL, never localhost.

> “This service runs independently of my laptop. I fetch a public task, solve from
> its evidence, submit my answer and retrieve its persisted score. The response
> includes components and clauses without expected answers. Scoring is shared
> with evaluation and training.”

Show requests in order:

```text
GET /healthz
GET /tasks?tier=2&n=1
POST /submit        X-Run-Token: issued token
GET /runs/run_id    X-Run-Token: same token
```

For the live correct submission, run:

```bash
python3 scripts/sandbox_smoke.py https://YOUR-SANDBOX.onrender.com
```

It solves only from public documents with independent Decimal arithmetic and prints
a token-free proof. Keep its private receipt off screen. Show the correct score
and the saved restart-persistence proof.
Mention ten tasks per run, bounded payloads and caller quotas. Identical submissions
are idempotent; changed answers receive409. A202 response is pending, not a pass.
The quickstart's empty answer is deliberately invalid and cannot replace this demo.

## 9:00–10:00 — transfer and reproduction

Screens: transfer findings, cost/latency and README.

> “Track B changes presentation while keeping rules fixed. Five reviewed tasks
> were frozen before outcomes. Generated ranking was [actual], transfer ranking
> was [actual], with agreement [actual]. [Explain one observed task-level issue.]
> Five tasks diagnose template dependence; they do not prove broad generalization.
> Limits include synthetic rules, repeated retrieval templates, small transfer
> sample and judge error. Costs and coverage are reported alongside scores.”

Show clean setup and the four submission links. State the highest-value next work,
such as additional held-out runs or independent fidelity checks, rather than new
features. Finish within the required time range.

## Rehearsal

Do one timed practice and one private recording. Check readable text, audio, hidden
secrets, correct links, genuine measurements and both timing minima. Use your own
words. If the live endpoint fails, fix it before the final recording; a saved run
must be labeled replayed and does not replace a live request.
