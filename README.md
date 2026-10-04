# Aster Payroll Gym

A verifiable synthetic monthly-payroll environment with one shared, clause-auditable
reward function for evaluation, reinforcement learning, and external submissions.
The written rule set is authoritative; Python is a testable implementation of it.

**Status:** runnable local code and deterministic adversarial baseline evidence.
Real model evaluations, Colab training, human audits, public deployment and Loom
are pending. This repository never substitutes fixture outputs for those results.

| Submission link | Status |
| --- | --- |
| GitHub repository | Replace after pushing this repository |
| Dashboard | Replace after enabling GitHub Pages |
| Sandbox | Replace after Render + Neon setup |
| Loom | Replace after recording the 8–12 minute walkthrough |

## Two-minute sandbox quickstart

After deployment, substitute the public URL below. No clone, provider key, or model
download is needed. The trivial answer earns zero, demonstrating the round trip.

```bash
ASTER_SANDBOX_URL=https://YOUR-SANDBOX.onrender.com python3 - <<'PY'
import json, os, time, urllib.request
base = os.environ['ASTER_SANDBOX_URL'].rstrip('/')
def request(path, payload=None, token=None):
    headers = {'Content-Type':'application/json'}
    if token: headers['X-Run-Token'] = token
    data = json.dumps(payload).encode() if payload is not None else None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(base+path, data=data, headers=headers), timeout=90) as r:
                return json.load(r)
        except Exception:
            if attempt == 3: raise
            time.sleep(2)
batch = request('/tasks?tier=2&n=1')
result = request('/submit', {'run_id':batch['run_id'], 'answers':[
    {'task_id':t['id'], 'answer':'{}'} for t in batch['tasks']]}, batch['run_token'])
print(json.dumps(result, indent=2))
PY
```

Free hosting may need approximately a minute to wake. See `docs/deployment.md` for
exact API examples, limits, persistence and operational recovery.

## Safe local setup

Python 3.11 and uv are required. Default dependencies contain no ML training stack.

```bash
uv sync --locked
uv run aster-gym validate
uv run pytest -q
uv run aster-gym baselines
uv run aster-gym baselines --n 30 --output results/comparison-baselines
uv run aster-gym analyze
uv run aster-gym report --output site
uv run aster-gym serve
```

Open `site/index.html` for the saved-results dashboard. The API listens on
`http://127.0.0.1:8000`; `python3 scripts/quickstart.py --help` explains its client.
The API server and these tests perform small deterministic calculations only.

**No local training or model inference is permitted.** `aster-gym train`,
`aster-gym infer`, and cloud model entrypoints refuse local execution before model
imports or downloads. Never run `uv sync --extra cloud` on this computer. The
Colab notebook installs that optional stack inside its GPU runtime.

## Domain and data

One fictional jurisdiction, AST currency, monthly salary, proration, one capped
employee contribution, a monthly allowance, three marginal tax bands, and net pay.
No actual jurisdiction or customer data is used. This is professional computation,
not trivia or advice about real payroll law.

The generator creates evidence, and an independent Python implementation computes
ground truth. The prose rule artifact was written before the calculator. Every
calculation and reward identifies its relevant clauses; disagreements with the
written authority are bugs.

The frozen corpus contains 12 seed tasks, 30 training tasks, 15 validation tasks,
120 evaluation tasks and 5 authored transfer tasks. Seeds and normalized business
input fingerprints are checked across partitions. All seed/transfer human-review
claims remain pending in `reviews/task-review-packet.json`.

Tier 1 retrieves a schedule field. Tier 2 computes interacting payroll lines.
Tier 3 includes absent YTD evidence, equal-authority salary conflicts, or a pay date
outside the available rule schedules. An agent must request clarification rather
than confidently invent a result. Stale distractors on a solvable task are not
grounds for abstention.

This reference implementation can generate unlimited graded homework within the
fictional rules. It cannot establish whether real documents are truthful, interpret
unmodeled law, or resolve discretionary employment disputes. External validity is
deliberately limited and measured with the five-task transfer test.

## Evaluation and rewards

The parser accepts one strict JSON object. Correct cents and clause citations are
checked without an LLM. Graded field accuracy supplies a bounded learning signal.
Invalid output, unsafe computation on a trap, and unnecessary abstention score
zero. Incorrect substantive work cannot exceed 0.20. A constrained remote judge
assesses only a verified blocking explanation; outages remain pending, never zero.

Three deterministic adversarial policies have genuine programmatic artifacts:
format-only, constant-payslip, and always-abstain. They are not model evaluations.
See `docs/reward-spec.md` and `docs/saved-results.md`.

Remote evaluation uses an OpenAI-compatible API. Copy `.env.example` to `.env`, add
your own key outside version control, and verify the account's free-tier entitlement.
Populate the selected entry's pricing in `configs/eval.json`; unset prices fail
closed. Keep billing disabled for the zero-dollar plan.

```bash
uv run aster-gym eval --config configs/eval.json --model gemini-3.8-flash \
  --n 30 --rollouts 3 --seed 7001 --max-cost 0 --output results/frontier
uv run aster-gym eval --config configs/eval.json --model gemini-3.5-flash-lite \
  --n 30 --rollouts 3 --seed 7001 --max-cost 0 --output results/efficient
uv run aster-gym eval --config configs/eval.json --model gemini-3.8-flash \
  --n 12 --mode tool --rollouts 3 --max-cost 0 --output results/frontier-tool
```

Small-model inference runs exclusively in Colab. Compare models on the same frozen
cohort, mode and reward version. Three replicate means supply the reported sample
SD; task and operational variability remain separately visible. Identical dataset
seeds do not make provider responses deterministic.

## Colab and result handoff

Open `notebooks/aster_colab.ipynb` in Google Colab, select a GPU runtime, and follow its
setup, smoke, sweep, held-out evaluation and export cells. Enter the repository URL
and put keys in Colab Secrets. Checkpoints stay in Google Drive or cloud storage.
Do not send model weights back to this computer.

GRPO uses four sampled completions per prompt and a frozen initial reference:

`A_i = (r_i - mean(group rewards)) / (std(group rewards) + epsilon)`

The actual objective is the token-masked, sequence-normalized clipped policy
surrogate minus `beta * KL`, using
`k3 = exp(log pi_ref - log pi_theta) - (log pi_ref - log pi_theta) - 1`.
The sweeps use beta 0.001 and 0.10. LoRA freezes the backbone; disabling fresh
adapters produces the reference policy, whose initial equivalence and unchanged
weights are checked. Equal-reward groups have zero reward advantage; KL may still
produce a gradient. See `docs/colab.md` for the executable settings and logs.

After Colab execution, download only the compact JSON results bundle:

```bash
uv run aster-gym import-results --bundle /path/to/colab-results.zip
uv run aster-gym analyze
uv run aster-gym report --output site
```

Import validates hashes, schema, versions, finite metrics and split separation.
It never loads checkpoints. Failed runs may be imported as failed runs; partial
curves remain partial. Training failures do not block the other phases.

## Review and submission

Start with `docs/human-review.md`, then `docs/submission-checklist.md`. They provide
your task audit, judge labels, failure-review worksheet and timed Loom outline.
The traceability register is `docs/traceability.json`; the readable matrix is
`docs/traceability.md`. Architecture, setup, edge cases, trade-offs, evaluation,
RL, transfer, deployment and AI-disclosure documents are under `docs/`.

Implementation readiness and submission readiness are different. Public URLs,
three real model configurations, 100+ generated-task score distribution, actual
RL curves/held-out evidence, your hand reviews and Loom must be completed before
calling this a finished assignment. With two more days, prioritize additional
held-out runs and independent fidelity checks rather than additional features.
