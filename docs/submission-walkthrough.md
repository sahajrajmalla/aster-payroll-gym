# Submission walkthrough

Use [the checklist](submission-checklist.md) for current status. Repository and
results dashboard are published, including the completed remote Flash-Lite run.
Code checks do not complete the remaining experiments
or independent human review. Keep all model work in Colab or a remote API.

## 1. Understand and review

Read [project explanation](project-understanding.md), [rules](../rules/aster-payroll-v1.md)
and [reward spec](reward-spec.md). Work out the example yourself.
Complete the [background reading](background-reading.md).

In `reviews/task-review-packet.json`, independently solve twelve seed rows and ten
rule-fidelity rows before comparing with `reference_result`. Enter your result,
name, notes and `reviewed: true` only after doing the work. Review/edit five transfer
presentations, keeping inputs and prompts synchronized. They must be frozen before
viewing model outcomes:

```bash
uv run aster-gym validate
uv run aster-gym freeze-transfer
```

In `reviews/judge-review-packet.json`, assign fifteen independent human labels:
0 unusable, 1 adequate, 2 excellent. Enter your reviewer name before remote ratings.
These examples are a rubric review set, not measured model outputs. Never mark
blank worksheets as completed.

## 2. Configure remote access safely

Create `.env` from `.env.example` only if absent; preserve existing database
credentials and keep it untracked. Add provider and judge keys using
local environment variables or cloud secret storage. Verify current model access,
free-tier entitlement and disabled billing. Only then supply explicit zero pricing
and `verified_free: true` for the selected remote entries in `configs/eval.json`,
and set `JUDGE_VERIFIED_FREE=true`. Unknown prices deliberately stop execution.

The checked-in model names are setup defaults, not proof your account can access
them. If a model changes, choose the final three configurations before comparison
and keep the shared task cohort and scoring versions fixed. Required policies
include a frontier model and a small model.

Run the reliability study only after your labels and judge configuration exist:

```bash
uv run aster-gym judge-study
```

It obtains three uncached ratings for each of fifteen examples. Preserve agreement,
confusion counts and repeat instability, including outages or partial completion.

## 3. Run genuine evaluation and cloud RL

For each configured remote model, substitute its exact name:

```bash
uv run aster-gym eval --config configs/eval.json --model MODEL_NAME   --n 30 --rollouts 3 --seed 7001 --max-cost 0 --output results/MODEL-comparison
uv run aster-gym eval --config configs/eval.json --model MODEL_NAME   --n 12 --mode tool --rollouts 3 --max-cost 0 --output results/MODEL-tool
uv run aster-gym eval --config configs/eval.json --model MODEL_NAME   --tasks data/transfer.jsonl --n 5 --rollouts 3 --max-cost 0   --output results/MODEL-transfer
```

The same frozen first 30 evaluation tasks form the three-policy comparison. The
small model also needs all 120 evaluation tasks and the five frozen transfer tasks.
Three genuine stochastic rollouts are required; repeated copied answers do not count.

Open the [Colab notebook](https://colab.research.google.com/github/sahajrajmalla/aster-payroll-gym/blob/main/notebooks/aster_colab.ipynb).
Select a GPU runtime, connect Drive, configure Secrets and follow [Colab guide](colab.md).
Check the repository commit and frozen review marker. Complete preflight before
explicitly enabling smoke training, two beta sweeps and inference. Defaults use
80 steps per beta; if resources require a smaller symmetric run, document it.

Select beta with validation only. Then evaluate the initial and selected policy
on untouched evaluation tasks. Keep checkpoint files in Drive. Preserve failures
and resumable metadata rather than replacing missing measurements.

## 4. Import and audit the evidence

Download only the compact JSON results bundle:

```bash
uv run aster-gym import-results --bundle /path/to/colab-results.zip
uv run aster-gym analyze
uv run aster-gym report --output site
```

The importer checks versions, paths, sizes, hashes and split separation. It never
loads weights. Complete [evaluation report](evaluation.md), [RL report](rl-report.md)
and [transfer report](advanced-track.md) from genuine records. Include mean±SD,
coverage, costs and operational failures. Partial runs stay partial.

Read 5–10 actual model failures and record identifiers/categories in
`reviews/failure-review-packet.json`. Inspect high-reward training outputs for gaming.
Record how many were checked, what happened and what remained inconclusive. Do not
claim improvement solely because training reward rose.

Regenerate the dashboard after accepted results, commit the lightweight artifacts
and confirm the published page has the correct cohort and pending states.

## 5. Verify the public sandbox

Follow [deployment guide](deployment.md) for Render/Neon and remote judge secrets.
The [public sandbox](https://aster-payroll-gym.onrender.com) passed correct external
Tier-2/3 submissions at **1.0** and retained both runs after a confirmed hosted
restart. Eleven public API contract checks also passed.
[Acceptance evidence](render-sandbox-acceptance.json) ·
[Tier 2](render-tier2-proof.json) · [Tier 3](render-tier3-proof.json).
Check [health](https://aster-payroll-gym.onrender.com/healthz).
From an independent client fetch Tier-2/3 tasks, derive a correct
answer from public evidence, submit it and retrieve the run. Do not import the
private reference into the client. Responses must omit expected answers.

The independent acceptance client performs the correct round trip without model
calls or private reference imports:

```bash
python3 scripts/sandbox_smoke.py https://aster-payroll-gym.onrender.com
```

Restart the Render service, then verify the same privately saved receipt:

```bash
python3 scripts/sandbox_smoke.py --verify-restart --restart-confirmed
```

Only confirm a restart you actually performed. The private receipt contains a run
token; do not publish it. The separate proof file excludes that token. Save HTTP/status evidence without
connection strings or tokens. The malformed-answer quickstart proves rejection;
it does not replace the required correct external submission.

## 6. Record and submit

Use [Loom script](loom-script.md): 8–12 minutes, ≥3 minutes on rewards,
≥1 minute reading a genuine failure, actual RL/transfer findings and a live public
sandbox request. Keep secrets off screen.

Submit repository, dashboard, sandbox and Loom links. Verify them in a private
browser window, run the final lightweight checks and tick only completed evidence
in [submission checklist](submission-checklist.md). Keep required reports, rule
artifact, traceability and the brief tool disclosure; do not substitute tests for
cloud measurements or independent reviews.
