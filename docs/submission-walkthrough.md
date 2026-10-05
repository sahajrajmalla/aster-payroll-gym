# From implementation GREEN to your final submission

This is your action guide. The software is implemented and the lightweight checks
passed. The experiments and personal reviews still need to happen. **GREEN means
implementation readiness; it does not mean the complete assignment is ready to
submit.** You do not need to understand every Python line before starting, but you
should understand the written rules, reward gates, experiment separation, and what
the evidence actually proves.

Your computer remains a writing, testing, and results-viewing machine. **Run every
training job and every Qwen model inference in hosted Google Colab with a CUDA GPU.**
Never install the `cloud` extra on your laptop. Remote API calls use a lightweight
client; they do not run a model on the laptop.

## 1. Know what is already done

The [repository](https://github.com/sahajrajmalla/aster-payroll-gym) and
[dashboard](https://sahajrajmalla.com.np/aster-payroll-gym/) are public. The corpus
contains 12 seed, 30 training, 15 validation, 120 evaluation, and five transfer
tasks. Python computes their labels independently of the generator and of any LLM.
The shared scorer, three real tools, evaluation harness, cloud training workflow,
sandbox API, deployment configuration, tests, and documentation are implemented.

Recorded QA includes 178 lightweight tests, lint/types, a clean-clone check, green
GitHub CI, and a correct local Tier-2 API round trip. The dashboard's saved
adversarial baseline scores are real deterministic outputs. They are **not model
results**. No local training, inference, or model-weight downloads happened.

The following are still pending: your independent reviews, actual model evaluation,
judge reliability calls, GPU acceptance and RL runs, measured transfer findings,
hosted Render/Neon sandbox, and Loom. These are substantial required evidence, not
optional polishing. See [the checklist](submission-checklist.md).

## 2. Prepare your accounts and safe workspace

You need your GitHub account, Google Colab with GPU availability, Google Drive
storage, a remote model/judge provider account, free Neon and Render accounts, and
a Loom recording account. Verify actual account quotas and availability before
starting. Do not enable paid billing for this zero-dollar plan.

In the existing project directory, the safe local commands are:

```bash
cd /Users/sahajrajmalla/Documents/NeurlAI
uv sync --locked
uv run aster-gym validate
uv run pytest -q
```

Copy `.env.example` to `.env` only if `.env` does not already exist. Edit it privately.
Set `MODEL_API_KEY` and `JUDGE_API_KEY`; use the appropriate remote endpoints and a
judge model actually available on your account. Never put keys in GitHub, notebook
code cells, screenshots, Loom, messages, or exported evidence. Colab uses its
Secrets panel; Render uses service secrets. `.env` is excluded from Git.

`JUDGE_VERIFIED_FREE` starts false. Set it true only after confirming free access
and disabled billing. Remote model `pricing` in `configs/eval.json` starts `null`,
so evaluation refuses to proceed. After verification, set the selected model's
pricing to:

```json
{"input_per_million": 0, "output_per_million": 0, "verified_free": true}
```

These numbers are your checked pricing declaration, not proof of a provider's
terms. Keep `--max-cost 0`; unknown pricing must remain blocked. Verify the suggested
model IDs rather than assuming your account has access. If an ID must change,
record the actual configuration and use that same policy for its generated and
transfer comparisons. The frontier, efficient, and small roles must remain clear.

## 3. Complete the personal reviews before the experiments

First finish the assignment's background reading using
[background-reading.md](background-reading.md): review 10–15 example environments,
the verifiers documentation, the RL overview and the architecture reference.
Record which sources you personally read and what you learned. The prepared
source list is not evidence that you have already completed this reading.

Read [the rules](../rules/aster-payroll-v1.md), then open
[`reviews/task-review-packet.json`](../reviews/task-review-packet.json).

1. For all **12 seed rows**, hide `reference_result`, read inputs, and calculate the
   answer or blockers yourself. Enter `human_result`, your name in `reviewer`, and
   notes. Compare with the reference only afterward; then set `reviewed: true`.
2. Repeat for the **10 fidelity audit rows**. Record disagreements before any fix,
   their clauses, and results after correction. Automated passing tests do not
   count as your independent audit.
3. Read all **five transfer rows** and their unfamiliar presentations. Calculate
   their results independently, check clarity, and approve them. The existing
   drafts are AI-assisted. If you edit them, update the task and matching review
   prompt/context consistently, then validate. Do not claim human authorship just
   by changing `reviewed`.

Freeze the approved transfer tasks **before looking at transfer model outcomes**:

```bash
uv run aster-gym validate
uv run aster-gym freeze-transfer
```

This creates `reviews/transfer-freeze.json`, tying your approval to the exact task
hash and time. Commit/push the review packet and marker so Colab clones the same
approved version. Do not regenerate or edit the frozen experimental corpus after
seeing results. A discovered error requires a documented new version and new runs.

Next, independently label all **15 examples** in
[`reviews/judge-review-packet.json`](../reviews/judge-review-packet.json), before
obtaining judge predictions. Enter `human_label` and `reviewer`: 0 means misleading
or unusable, 1 means a specific understandable issue with an adequate next action,
and 2 means a clear concise issue with a concrete next action and no unsupported
claims. These are handwritten rubric examples, not generated model answers.

Then run the bounded remote reliability study:

```bash
uv run aster-gym judge-study
uv run aster-gym analyze
```

It requests three uncached ratings per example: 45 judgments if all succeed.
Inspect human agreement, confusion counts, and repeat disagreement. Low agreement
is a finding to explain and address, not a number to hide. Pending calls are not
successful ratings. See [human review](human-review.md) and [rewards](reward-spec.md).

## 4. Freeze the code and run the real model evaluations

Commit your reviewed configuration before running. Record the commit ID with
`git rev-parse HEAD`. Keep this implementation, rule set, prompts, task files,
sampling settings, and scorer version unchanged throughout the comparisons.
Source hashes protect resumption: a source change requires a fresh run directory
and appropriate reruns. Do not splice results across different reward versions.

Run the two remote models on the **same first 30 frozen evaluation tasks**, three
stochastic rollouts per task:

```bash
uv run aster-gym eval --config configs/eval.json --model gemini-3.8-flash \
  --n 30 --rollouts 3 --seed 7001 --max-cost 0 --output results/frontier
uv run aster-gym eval --config configs/eval.json --model gemini-3.5-flash-lite \
  --n 30 --rollouts 3 --seed 7001 --max-cost 0 --output results/efficient
```

Substitute verified IDs if necessary. Compare tool use on the **same first 12
tasks**; Colab supplies the matching small-model tool run:

```bash
uv run aster-gym eval --config configs/eval.json --model gemini-3.8-flash \
  --n 12 --mode tool --rollouts 3 --seed 7001 --max-cost 0 \
  --output results/frontier-tool
```

Run both remote policies on the **same five frozen transfer tasks**:

```bash
uv run aster-gym eval --config configs/eval.json --model gemini-3.8-flash \
  --tasks data/transfer.jsonl --rollouts 3 --seed 7001 --max-cost 0 \
  --output results/frontier-transfer
uv run aster-gym eval --config configs/eval.json --model gemini-3.5-flash-lite \
  --tasks data/transfer.jsonl --rollouts 3 --seed 7001 --max-cost 0 \
  --output results/efficient-transfer
```

Inspect `metrics.json`, `scores.json`, `transcript.jsonl`, `config.json`, and
`budget.json` in each directory. Report mean ± SD of complete replicate means,
tier/component performance, coverage, retries, latency, tokens, and confirmed
versus conservatively accounted spend. Provider timeouts and judge outages are
operational failures or pending scores, not incorrect arithmetic. Repeating the
identical command resumes the same directory without throwing away previous work.

The 120-task small-model distribution and 30-task comparison are separate cohorts.
Do not rank a 120-task score against a 30-task score. See [evaluation](evaluation.md).

## 5. Execute Qwen inference and RL only in Colab

Open the [actual Colab notebook](https://colab.research.google.com/github/sahajrajmalla/aster-payroll-gym/blob/main/notebooks/aster_colab.ipynb)
and save your own copy. Select **Runtime → Change runtime type → GPU**. Its
`REPO_URL` is already correct; you do not need to replace it. Make sure your review
commit is pushed before cloning, and verify the cloned commit matches it.

Run cells in order. Hosted-runtime guards run first. Dependency installation and
model downloads then happen **inside Colab**, never on your laptop. Mount Drive,
add `JUDGE_API_KEY` to Colab Secrets, enable access, verify the judge model, and set
`FREE_TIER_CONFIRMED = True` only after checking the account. The preflight checks
CUDA, pinned dependencies, optional `verifiers` environments, available storage,
and writable checkpoint location. Stop and preserve the error if it fails.

The notebook deliberately stops until you choose its explicit start flags:

1. Set `START_SMOKE = True` for the real three-step smoke. Inspect its metrics and
   frozen-reference proof. A smoke run is compatibility evidence, not a full RL
   result. Record equal-reward groups or unchanged adapters honestly.
2. Set `START_SWEEPS = True` for both beta values, **0.001 and 0.10**. Defaults are
   80 optimizer steps each, four completions per prompt, and checkpoints every 20
   steps. The training reference remains frozen. Validation chooses beta before
   initial and selected policies are evaluated on the untouched 120-task split,
   three rollouts per task.
3. Set `START_INFERENCE = True` for the initial small model's 30-task, 12-task tool,
   and approved five-task transfer comparisons. Use this initial policy for the
   three-model ranking; describe trained-policy comparisons separately.

The 120-task initial evaluation is produced by a successful full training workflow
as `heldout-before`. If training fails, collect the required initial distribution
independently **inside Colab**:

```bash
uv run --frozen --extra cloud --no-dev python -m aster_gym.cloud.evaluate \
  --config /content/aster-train.json --tasks data/evaluation.jsonl \
  --output /content/drive/MyDrive/aster-gym-results/small-model-eval \
  --start-inference
```

If resources are insufficient, use 40 steps for **both** betas, in a fresh output
directory, and document the change. Do not reduce one sweep alone. Use `RESUME =
True` only with the unchanged configuration and compatible saved checkpoint.
Judge quotas are bounded and can stop a run; increase them only within your
verified allowance. Never replace missing rewards with zeros to keep training going.

Preserve failed/partial status and cloud logs. Continue reviews, evaluation,
deployment, and documentation if training fails. A transparent partial submission
is possible, but **failed or smoke-only training does not fulfill the assignment's
required RL evidence**. See [Colab](colab.md) and [the RL report](rl-report.md).

## 6. Bring back evidence and write the measured reports

Use the notebook's export cell. Download only `aster-results.zip`, containing
allowlisted JSON/JSONL evidence and hashes. **Leave weights and checkpoints in
Drive.** A notebook error does not prevent running the export cell independently
to preserve failed/partial artifacts that were actually written.

On your laptop, import to a new destination:

```bash
uv run aster-gym import-results --bundle /path/to/aster-results.zip \
  --output results/colab-import
uv run aster-gym analyze
uv run aster-gym report --output site
```

The importer rejects unsafe files, incompatible scorer versions, bad checksums,
and split leakage. Do not bypass validation. Inspect the generated dashboard:
complete runs appear as evidence; absent/partial experiments remain visible.
`analyze` calculates matched three-model transfer rankings and Spearman agreement.

Read 5–10 genuine failed model transcripts and complete
`reviews/failure-review-packet.json`. Record IDs, violated clauses, categories,
counts, and remedies. Search actual training completions for reward gaming:
unnecessary abstention, clause copying, valid-JSON nonsense, length tricks, and
high-reward wrong answers. Report examples, or state how many outputs/attacks you
checked if no exploit appeared. Do not invent a failure or improvement.

Fill [evaluation](evaluation.md), [RL](rl-report.md), [failure analysis](failure-analysis.md),
and [transfer](advanced-track.md) with measured numbers and limitations. The five
RL views need actual reward, KL, entropy, completion length, and tier pass rates;
also inspect components, equal-reward groups, and before/after results. Update
[AI disclosure](ai-usage.md) with your actual tools, cloud runtime, costs, and work.

Commit/push safe synthetic result files and reports. Confirm green CI and the
**Evidence dashboard** GitHub Actions deployment. Refresh the public dashboard
and confirm it shows your new evidence. Do not publish runtime databases, run
tokens, keys, or cloud checkpoints.

## 7. Deploy and independently test the sandbox

Follow [deployment instructions](deployment.md). Create Neon Postgres, then
Render **New → Blueprint** connected to the repository's `render.yaml`. Supply
the pooled TLS database URL and judge key as secrets. Verify free eligibility
before accepting the blueprint's `JUDGE_VERIFIED_FREE=true`; configure the actual
available judge model/endpoint. The server contains no training stack.

Open `/healthz` and `/docs` on the real public URL. From another browser or device,
run the README's no-clone quickstart using that URL. Its intentionally malformed
answer should receive zero; that proves the API round trip, not good solving.
Also solve and submit a real Tier-2 answer and a Tier-3 blocking answer. Retrieve
each run using its private `X-Run-Token`; verify component/clauses and judge status.
Save URLs, timestamps, IDs, and safe response evidence. Keep tokens private.

Restart the Render service, then retrieve an existing run with the same token to
prove Neon persistence. Test that public tasks/reports contain no expected answers.
Allow time for free hosting to wake. Judge outages must remain retryable pending;
retry the identical submission rather than changing answers. Replace README's
sandbox placeholder only after this external test passes.

## 8. Record Loom and submit four working links

Rehearse using [the Loom outline](loom-outline.md). Record 8–12 minutes, with at
least three minutes explaining rewards, at least one minute reading a **genuine
model failure**, and a live request to the public sandbox. Explain what you know,
not equations you cannot defend. Show measured findings, uncertainty, cloud-only
execution, and honest AI usage. Hide all secrets and tokens before recording.

Use this final readiness check:

- [ ] Background reading and personal seed/fidelity/judge reviews are completed; transfer was frozen
      before outcomes, with true reviewer identities.
- [ ] Three models have three complete stochastic rollouts on the same 30 tasks;
      the small model covers 120 tasks; tool and five-task transfer findings exist.
- [ ] Judge agreement/stability, operational coverage, and actual spend are recorded.
- [ ] Both real RL sweeps, frozen-reference proof, five curves, validation choice,
      held-out before/after, and reward-gaming inspection are present—or limitations
      explicitly say which required RL evidence is missing.
- [ ] Genuine failures and counted taxonomy are documented; claims match transcripts.
- [ ] Repo CI is green; dashboard evidence works; public sandbox survives restart;
      a reviewer can complete the quickstart independently.
- [ ] Loom permissions work in a private browser; README contains four real links;
      AI disclosure and known limitations are accurate.

Confirm the assignment's deadline from **when you actually received it** and its
stated business-day rules, not the repository creation date. Ask the recruiter if
the receipt/deadline is ambiguous. Record submission time and total actual spend.

Submission draft—edit and send yourself through the requested channel:

> Subject: AI Product Operator Assignment - [Your Name]
>
> Hello [Recruiter's name],
>
> Please find my Aster Payroll Gym submission:
>
> Repository: https://github.com/sahajrajmalla/aster-payroll-gym
>
> Dashboard: https://sahajrajmalla.com.np/aster-payroll-gym/
>
> Sandbox: [tested public URL]
>
> Loom: [working recording URL]
>
> The project uses synthetic payroll tasks, an independent deterministic reference,
> and a shared clause-auditable reward. My strongest measured finding is [actual
> finding with a number]. The documented limitations are [actual limitations].
>
> Thank you,\
> Sahaj Raj Malla

Do not send placeholders as completed links or describe pending experiments as
successful. If a required phase remains incomplete, name it plainly. Being able to
show evidence and explain limits is stronger than making a claim you cannot defend.
