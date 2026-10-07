# Cloud execution and result handoff

Open the [Colab notebook](https://colab.research.google.com/github/sahajrajmalla/aster-payroll-gym/blob/main/notebooks/aster_colab.ipynb).
**No training, open-weight inference or weight downloads run on the laptop.**
The notebook and entrypoints require hosted Colab; Qwen also requires CUDA.
There is no CPU/MPS fallback or local override. GPU acceptance and model findings
remain pending until genuine runs succeed.

## Before running

1. Complete the independent task reviews and fifteen judge labels. Freeze the five
   transfer tasks with `uv run --extra server aster-gym freeze-transfer`, then commit and push.
2. Save your own notebook copy. Set `REPO_REF` to that reviewed commit SHA. The
   notebook resolves and prints `CODE_REVISION`, checks out that exact revision,
   and refuses to overwrite tracked edits.
3. Select **Runtime → Change runtime type → GPU** and mount Drive. Allow at least
   5 GiB of cloud filesystem space; checkpoints remain on Drive.
4. Add `JUDGE_API_KEY` and `MODEL_API_KEY` in Colab Secrets and enable notebook
   access. Both may contain the same Google API key. Never paste keys into cells.
5. Verify model access, quotas, zero-dollar pricing and disabled billing in your
   account. Only then enable `FREE_TIER_CONFIRMED` for the judge and
   `REMOTE_FREE_TIER_CONFIRMED` for the two comparison models. Prices fail closed
   until confirmed. Keep the hard cost cap at zero.

The notebook installs the pinned `cloud` extra only in Colab. Preflight checks CUDA,
storage, dependency versions, both optional `verifiers` environments and frozen
data. A failed GPU preflight blocks Qwen/RL while remote evaluation can continue.

## Run the phases

Every start switch defaults to `False`. Enable a phase deliberately, then run its cell:

- `START_SMOKE`: three optimizer steps. Inspect the real metrics, failure file and
  `reference_proof.json`. Equal-reward groups and an unchanged adapter are an
  ineffective smoke, not evidence of learning.
- `START_SWEEPS`: set `SMOKE_REVIEWED=True` after inspection. Run both beta values,
  validation-only selection and held-out initial/selected comparisons.
- `START_INFERENCE`: Qwen runs three rollouts per task on the same 30-task cohort,
  all 120 evaluation tasks and 12 tool tasks. Five transfer tasks run only when
  the exact human-approved freeze marker verifies.
- `START_REMOTE_EVAL`: both configured remote models run three rollouts on the
  same 30-task, 12-tool-task and approved five-task transfer cohorts. Temperature,
  seed and completion budget match Qwen; the shared scorer records full evidence.
- `START_JUDGE_STUDY`: requires fifteen independent human labels and reviewer
  names. It saves three uncached production-scorer ratings per example in a Drive
  copy of the review packet. An interrupted study resumes without overwriting labels.

Failed commands write `run_status.json`; training also saves failure/checkpoint
evidence. They do not stop independent model evaluation or export. A command's
zero exit status is not a claim that its experimental evidence is complete.

## Configuration and resume

`configs/train.json` pins Qwen2.5-0.5B-Instruct to
`7ae557604adf67be50417f59c2c2f167def9a775`. Defaults: LoRA rank 8/alpha 16, four
completions, temperature 0.8, 256 completion tokens, learning rate `5e-5`, 80
optimizer steps per beta (0.001 and 0.10), checkpoint every 20 steps, three
evaluation rollouts. Microbatch 1 × accumulation 4 must remain divisible by four.

If resources are insufficient, reduce **both** runs to 40 steps and select a fresh
output directory. Record the adjustment. Resume with `RESUME=True` and unchanged
code, model, rules, prompts, splits and configuration. A failure before the first
checkpoint needs a new output directory. Never delete failed evidence.

`JUDGE_MAX_CALLS` starts at 100 per process. Adjust only within verified account
allowances. Provider errors and quota exhaustion leave scores pending; they are
never replaced with zero or a simpler rubric. Fix the cloud issue and resume.
The judge uses `low` reasoning effort and 512 bounded output tokens, including
reasoning. `JUDGE_MAX_TOKENS` accepts 64–2048; settings enter every judge identity
and cannot change silently during resume. Google's
[compatibility guide](https://ai.google.dev/gemini-api/docs/openai) documents `low`.
Unrelated endpoints omit effort unless explicitly configured. Real acceptance
still requires the cloud smoke and judge study; exhausted reasoning never earns a fabricated label.

The frozen adapter-disabled backbone is the reference. Initial logits must agree;
all reference parameters must remain frozen and their full hashes unchanged.
Every step logs reward, sampled `k3` KL, beta-times-KL, entropy, length, per-tier
pass rates, components, equal-reward fraction and actual completions. Missing-tier
metrics are `null`, not zero. See the RL report for the objective and interpretation.
Source compatibility was checked against the pinned
[TRL trainer](https://github.com/huggingface/trl/blob/v0.26.2/trl/trainer/grpo_trainer.py);
actual CUDA execution still requires the smoke test.

## Verified setup defaults

On 2026-10-05, Google's official docs listed stable
[Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash) and
[Gemini 3.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite),
with free standard input/output in the [pricing page](https://ai.google.dev/gemini-api/docs/pricing).
The frontier comparison role is a selection inference from Google's
[3.8 model-card benchmarks](https://deepmind.google/models/model-cards/gemini-3-8-flash/).
It is not a measured ranking in this gym. On 2026-10-06, the owner confirmed disabled
billing/free-tier access; authenticated metadata and actual API calls verified both
configured model IDs. Rate limits left some runs partial. Check current account
quotas before another run.

## Return results

The final cell exports only validated JSON/JSONL; weights, optimizer state, code
and pickle stay out. Partial runs remain partial. It downloads the reviewed judge
packet separately. Keep the notebook, full logs and checkpoints on Drive.

On the laptop, use a new import directory:

```bash
uv run --extra server aster-gym import-results --bundle aster-results.zip --output results/colab-import
# Copy the downloaded judge-review-packet.json into reviews/ after checking its labels.
uv run --extra server aster-gym analyze
uv run --extra server aster-gym report
```

Import checks versions, schemas, checksums, unsafe paths, archive limits, split
separation, transcript identities and required metrics; it recomputes summaries.
It never loads checkpoints. Update the evaluation, RL, failure and transfer reports
from genuine accepted records. Imported outputs are reviewable evidence, not
cryptographic proof of where a model ran.
