# Ten-minute Loom script

Record after the [submission checklist](submission-checklist.md) has genuine
experiment, review and public-sandbox evidence. Target 10 minutes; required range
is 8–12. Spend at least three minutes on rewards and one minute reading a genuine
model failure. This outline allocates 3:30 and 1:15 respectively.

## Prepare the screen

Use the Loom desktop app with screen recording and microphone; camera is optional.
Check a 15-second test for clear audio and readable text, then do a timed rehearsal.
The [Starter plan has a five-minute limit](https://support.atlassian.com/loom/docs/how-long-can-i-record),
so check that your recording account supports the required 8–12-minute video.
[Official recorder setup](https://support.atlassian.com/loom/docs/get-started-with-the-loom-desktop-app).

Open the README, rulebook, reward spec, dashboard, one real failed transcript,
RL/transfer reports and public sandbox client. Use readable zoom and a quiet room.
Warm the free hosted service before recording. Close `.env`, provider consoles,
Colab Secrets, database credentials and tabs containing run tokens.

Before recording, open the public `/healthz` endpoint and wait for `status: ok`.
Use a new receipt/proof filename for each fresh take. After recording, title the
video “Aster Payroll Gym — Niural AI Labs — [Your Name]”, review the full video,
and test the reviewer link in a private browser window.

Current evidence: Gemini 3.5 Flash-Lite completed 90/90 remote API samples with
reward **0.11444 ± 0.03845** across three rollout means. Gemini 3.8 Flash and the
tool-use run are partial; they do not establish a three-model ranking. The
independent failure packet is prepared but unreviewed. Qwen/RL, judge reliability,
transfer findings remain pending. The [2026-10-07 Colab attempt](colab-run-status-2026-10-07.json)
records failed GPU preflight and skipped API phases; it contains no model or
training results. Public Tier-2/3 submissions scored **1.0** and
survived a confirmed hosted restart; eleven API contract checks passed. These API calls
were remote; no open-weight model inference or training ran on this computer.

Fill the remaining slots from saved results before recording:

- Three models: [names], comparison [means±SD], coverage [counts].
- Judge: [human agreement], [repeat instability], [review disagreements].
- Failure: use the genuine `failure-02` example below; complete your own review.
- RL: [actual steps], selected beta [value], validation [reason], held-out
  [initial→selected means±SD], [curve observations], [gaming sample/findings].
- Transfer: [generated/transfer rankings], [agreement], [one task failure/reversal].
- Public sandbox: https://aster-payroll-gym.onrender.com; Tier-2/3 both **1.0**,
  [hosted restart and contract evidence](render-sandbox-acceptance.json).
- Resources: [confirmed cost], [latency], [GPU/runtime], [remaining limitations].

Empty slots are pending evidence. Do not turn planned behavior into measured claims.

## If submitting without API credentials

This is a partial submission, even if its sandbox and code checks pass. Do not
read unfinished result slots. The notebook's `SKIP_API_PHASES=True` mode permits
Qwen inference on Colab CUDA while recording training, new remote comparisons
and judge reliability as skipped. Keep the reward and genuine-failure sections
and live sandbox demonstration; replace unexecuted findings with this disclosure:

> “I could not provide judge API credentials for this Colab session. I therefore
> skipped the RL sweeps, new remote comparisons and judge reliability study.
> The reward has not been simplified: otherwise correct blocking explanations
> remain pending without the judge. Those required experiments are incomplete.”

If Qwen has not run, say “Qwen evaluation is also pending.” If its accepted results
have been imported, show their actual coverage and scores; distinguish complete
deterministic scores from pending explanation judgments. Never describe inference
as training. Existing saved Gemini results remain genuine historical evidence.
Do not claim three-model ranking or transfer ranking agreement without complete,
comparable records. In the RL segment, show configuration and the pending dashboard
instead of invented curves; explain the planned method and this limitation.

Until a successful preflight rerun produces genuine evidence, also say:

> “My Colab preflight exited with an error, so this attempt produced no Qwen or
> training results. The recorded status does not identify the cause. A code audit
> found and fixed a runtime-recognition issue involving the isolated environment;
> that fix still needs successful cloud verification. I have preserved the failed
> attempt and marked the experiments incomplete.”

If preflight later passes, replace this with the actual rerun status. Passing
preflight is setup evidence, not a model benchmark or training result.

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

Show the ceiling example: gross 320,000, remaining ceiling 10,000, contribution 500,
taxable 309,500, tax 21,900, net 297,600 cents. Then remove YTD.

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

Judge reliability and independent fidelity reviews are currently pending. After
they finish, read actual agreement and three-repeat stability, explain an observed
disagreement and show the review count. Stable judgments can still be wrong.
This section must reach at least three minutes.

## 4:30–5:45 — read one genuine failure

Screen: [failure packet](../reviews/failure-review-packet.json), row `failure-02`,
and its [saved transcript](../results/gemini-3.8-comparison/transcript.jsonl).
Run `c697a5f3d20e737d05ae`; task `task-0ec7d48247ff1b45b74b`; rollout **0**.
This is a completed Gemini 3.8 Flash answer, not a provider outage or fixture.

Read the task's request and this exact model explanation aloud:

> “Calculated payslip for 2030-01-18 under schedule ASTER-2030: gross is 522,564
> cents, contribution is 0 cents as YTD pensionable earnings met the 1,200,000
> ceiling, taxable pay is 512,564 cents after 10,000 allowance, tax is 64,513
> cents, and net pay is 458,051 cents.”

Then explain the evidence:

> “The reference gives gross 522,587, taxable pay 512,587, tax 62,517 and net
> 460,070 cents. Only the zero contribution matches. R1 requires exact rounded
> cents; R3 defines salary proration plus bonus; R5 defines taxable pay; R6 applies
> marginal bands; R7 defines net pay. Field accuracy is 0.20. The scorer records
> SUBSTANTIVE_MISMATCH and the CORRECTNESS_CAP gate, with final reward 0.05.
> Valid JSON and citations cannot rescue incorrect arithmetic. Calculator-assisted
> proration and a five-field check are proposed remedies, not measured improvements.
> My independent reading of this packet is still required before signing the review.”

Spend at least one minute reading the actual response and comparing these fields.
After you personally complete the review, update the final sentence truthfully.

## 5:45–7:30 — cloud RL and held-out evidence

Current status: cloud execution pending. All open-weight inference and training
must run in Colab with CUDA; there is no local CPU/MPS fallback. Remote Gemini API
evaluation is separate from Colab. Show five actual curves and before/after results
only after genuine cloud artifacts are imported.

> “The cloud workflow uses Qwen2.5-0.5B-Instruct with LoRA and GRPO: four
> sampled completions receive relative group advantages. A KL penalty discourages
> excessive movement from the frozen initial policy. Both planned beta runs start
> from the same policy; validation selects beta before untouched held-out evaluation.
> This explains the implementation; training results remain pending until executed.”

After execution, read actual completed steps, beta choice and before/after mean±SD. Show initial
reference equivalence and unchanged parameter hashes. Explain reward, KL, entropy,
completion length and tier pass rate together; include equal-reward group fraction.

> “Rising reward alone is not enough. I inspected [count] outputs for [attacks]
> and observed [actual findings]. [Describe a negative or inconclusive result
> honestly.] Checkpoints remain in cloud storage; checked JSON evidence drives
> the dashboard.”

## 7:30–9:00 — live public sandbox

The [Render sandbox](https://aster-payroll-gym.onrender.com) passed external
Tier-2/3 submissions at **1.0**, eleven contract checks and confirmed hosted
restart persistence. Show [recorded evidence](render-sandbox-acceptance.json), then
perform a fresh live request for the recording.
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
cd /Users/sahajrajmalla/Documents/NeurlAI
uv run --extra server python scripts/sandbox_smoke.py https://aster-payroll-gym.onrender.com --receipt tmp/loom-take1-private.json --proof tmp/loom-take1-proof.json
```

For another fresh recording, change both paths to `loom-take2` (or another unused
name). `--resume` reuses the earlier task; do not describe that as a new task fetch.
It solves only from public documents with independent Decimal arithmetic and prints
a token-free proof. Keep its private receipt off screen. Show the correct score
and the saved restart-persistence proof.
Mention ten tasks per run, bounded payloads and caller quotas. Identical submissions
are idempotent; changed answers receive 409. A 202 response is pending, not a pass.
The quickstart's empty answer is deliberately invalid and cannot replace this demo.

## 9:00–10:00 — transfer and reproduction

Transfer approval, freeze and measured findings are pending. After execution,
show the transfer findings, cost/latency and README.

> “Track B changes presentation while keeping rules fixed. [After approval and
> execution: Five reviewed tasks were frozen before outcomes.] Generated ranking was [actual], transfer ranking
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
