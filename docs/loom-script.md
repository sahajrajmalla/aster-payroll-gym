# Ten-minute Loom recording guide

Record 8–12 minutes, including >=3 minutes on rewards, >=1 minute reading a genuine
failure and a live public sandbox request. Use your own words; the script below
is a speaking guide. Check your account permits a video longer than five minutes.

## Before recording

Open README, rulebook, reward spec, dashboard, failure packet and public API.
Use readable zoom; record a 15-second microphone test. Hide `.env`, Colab Secrets,
provider consoles and private run receipts. Warm the sandbox at
https://aster-payroll-gym.onrender.com/healthz. Rehearse once with a timer.
Consult [the checklist](submission-checklist.md) immediately before recording;
read only measured findings. Unexecuted experiments must be described as pending.

## 0:00–1:00 — what the assignment asked for

> “The assignment was to build an AI training gym: tasks, tools, rewards, model
> evaluation, a dashboard, reinforcement learning and a public sandbox. I chose
> one fictional monthly-payroll workflow because dates, document authority,
> arithmetic and missing evidence interact, with deterministic ground truth.
> Tier 1 retrieves a rule, Tier 2 calculates pay and Tier 3 asks for missing or
> conflicting evidence. No real employee data is used.”

Show the README and one task. State the actual completion status.

## 1:00–4:30 — reward design (at least three minutes)

Show the rulebook and reward specification; pause to point at each component.

> “The written rulebook is authoritative. Python computes expected answers;
> an LLM does not generate ground truth. Every component returns diagnostic codes
> and clause references so a penalty can be audited.”

Walk through the [worked example](project-understanding.md): gross 320,000,
contribution 500, taxable 309,500, tax 21,900, net 297,600 cents. Then remove YTD.

> “Without prior earnings the ceiling is unknown. The answer should request that
> field, not guess zero. Refusing a solvable task is also wrong.”

> “Ordinary weights are correctness 60%, fields 25%, action 10% and format 5%.
> Blocked tasks use diagnosis 55%, fields 25%, action 10%, explanation 5% and
> format 5%. Hard gates override the sum: invalid output, unsafe computation and
> unjustified refusal score zero. Incorrect substantive answers cannot exceed .20;
> passing requires .975. Valid formatting alone earns no reward.”

Show malformed-output and constant/always-abstain baseline evidence.

> “The remote judge only assesses the usefulness of an otherwise correct blocking
> explanation. It cannot calculate payroll or override Python rejection. Eligibility
> and caching are shared across evaluation, training and sandbox. An outage leaves
> scoring pending; the rubric is never silently simplified.”

Show the actual judge study and independent numerical audit:

> “I measured 45 judge responses. One was malformed, and two of fourteen valid
> triples disagreed. All nine injection ratings were zero. That measures stability,
> not human accuracy. A separate rational-arithmetic implementation agrees with
> all 27 review inputs; personal rule-fidelity checks remain a separate step.”

## 4:30–5:45 — read a genuine failure (at least one minute)

Open [failure packet](../reviews/failure-review-packet.json), `failure-02`, and
[actual transcript](../results/gemini-3.8-comparison/transcript.jsonl).
Run `c697a5f3d20e737d05ae`, task `task-0ec7d48247ff1b45b74b`, rollout 0.
Read the request and model response aloud:

> “Calculated payslip for 2030-01-18 under schedule ASTER-2030: gross is 522,564
> cents, contribution is 0 cents as YTD pensionable earnings met the 1,200,000
> ceiling, taxable pay is 512,564 cents after 10,000 allowance, tax is 64,513
> cents, and net pay is 458,051 cents.”

> “The reference gives gross 522,587, taxable 512,587, tax 62,517 and net
> 460,070. Only contribution matches. R1 requires rounded cents, R3 proration,
> R5 taxable pay, R6 marginal bands and R7 net pay. Field accuracy is .20;
> the substantive-error gate produces final reward .05. JSON and citations cannot
> rescue wrong arithmetic. Calculator-assisted proration is a proposed remedy,
> not a measured improvement.”

Read/classify this failure yourself before signing the review packet.

## 5:45–7:30 — evaluation and RL

Show actual model coverage, mean±SD, tiers, cost and latency. Existing complete
Gemini 3.5 comparison: 90/90 samples, reward .11444 ± .03845 across three rollout
means. Qwen initial-policy inference completed 90/90 samples with reward
.00139 ± .00241 and zero passes: 89 invalid contracts and one substantive-error
cap. Its judge and seed differ, so this is a separate experiment group, not a
controlled ranking against Gemini. The full Qwen evaluation has 360/360 scored
samples across 120 tasks, reward 0 ± 0; the tool cohort has 36/36, also 0 ± 0,
with no accepted tool calls. Qwen's all-zero rewards warn of flat group-relative advantages during training.
A separate Gemini run completed all 120 tasks with three rollouts: 360 answers,
reward .12049 ± .00732 and 32 passes. That supplies the required varied distribution.
The models still require a compatible common-cohort ranking before claiming superiority.
Gemini 3.8 and Gemini tool runs are partial. Setup checks are not model outcomes.

> “Cloud training uses Qwen0.5B, LoRA and GRPO with four grouped completions.
> Relative rewards provide advantages; KL limits movement from a frozen initial
> policy. Two beta runs start identically. Validation selects beta before untouched
> held-out evaluation. Logs include reward, KL, entropy, length and tier pass rate.”

If training remains skipped, say plainly:

> “The remote judge study ran successfully, but this Colab session still needs
> Notebook access to its judge key before mixed-tier training can start. RL sweeps
> are pending; no training improvement is claimed. GPU inference is a separate
> experiment.”

If completed, show actual five curves, reference proof, selected beta, before/after
results and counted reward-gaming observations. Never draw replacement curves.

## 7:30–9:00 — live public sandbox

> “The service runs independently of my laptop. I fetch a task, solve from public
> evidence, submit an answer and retrieve its persisted score. The shared scorer
> returns components and clauses without exposing expected answers.”

Run in a terminal, using fresh filenames for each take:

```sh
cd /Users/sahajrajmalla/Documents/NeurlAI
uv run --extra server python scripts/sandbox_smoke.py https://aster-payroll-gym.onrender.com --receipt tmp/loom-take1-private.json --proof tmp/loom-take1-proof.json
```

Show `submission_verified`, `complete` and score 1.0. This client uses independent
Decimal arithmetic, not an LLM. Explain `GET /tasks`, `POST /submit` and
`GET /runs/{run_id}` with the issued token. Keep private receipts off screen.
Show [restart proof](render-sandbox-acceptance.json): correct Tier-2/3 submissions
survived an actual hosted restart. Mention ten tasks per run, quotas, bounded
payloads and idempotency. Pending HTTP 202 is not a completed score.

## 9:00–10:00 — transfer, limits and reproduction

> “Track B changes presentation while preserving rules. Five tasks need independent
> approval and freezing before outcomes. Without comparable completed runs I cannot
> claim ranking agreement. Synthetic rules, repeated retrieval templates and a
> five-task transfer sample limit generalization.”

Show actual transfer findings if available, otherwise its pending report. Show
setup commands, repository/dashboard/sandbox links and honest outstanding work.
End within 8–12 minutes. Review the recording and test its share link in a private
browser window before adding it to the checklist and submission email.
