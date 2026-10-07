# Cloud execution and result handoff

Open the [Colab notebook](https://colab.research.google.com/github/sahajrajmalla/aster-payroll-gym/blob/main/notebooks/aster_colab.ipynb).
**No training, open-weight inference or weight downloads run on the laptop.**
The notebook and entrypoints require hosted Colab; Qwen also requires CUDA.
There is no CPU/MPS fallback or local override. The owner-supplied Tesla T4
preflight passed; model findings remain pending.

## Recorded attempt and preflight rerun

The [2026-10-07 attempt](colab-run-status-2026-10-07.json), revision
`91366b4b07b60392c637ef2d615b5df58e10f4f6`, records `gpu-preflight` exit 1 and
three deliberate `API_PHASES_SKIPPED` phases. It contains no Qwen or RL evidence.
No stderr was supplied; this status alone does not prove a missing GPU.

The code audit found that isolated `uv` Python can miss the native `google.colab`
package even on hosted Colab. The guard now checks bounded system package
locations as well, without changing Python's import path. Explicit start, hosted
runtime markers and CUDA remain mandatory; there is no local or CPU fallback.
Preflight now saves structured, sanitized diagnostics instead of only an exit code.

The [later rerun](colab-preflight-passed-2026-10-07.json) passed all preflight checks
at revision `2cc6eb1b7d90c9e5e96369508ad1965cb4efeccc`. Task preparation then
failed because the notebook passed string filenames to a writer expecting `Path`.
The writer now accepts both. Its regression test executes the exact notebook
command and checks that the 30/12-task cohorts retain the frozen inputs.

For the already-open notebook at the older revision, replace only `cohort_code`
in the preflight/cohort cell with the following, then rerun that cell:

```python
cohort_code = (
    "from pathlib import Path; "
    "from aster_gym.generator import read_taskset, write_taskset; "
    "tasks = read_taskset('data/evaluation.jsonl'); "
    "write_taskset(tasks[:30], Path('data/comparison.jsonl')); "
    "write_taskset(tasks[:12], Path('data/tool-comparison.jsonl'))"
)
```

This works without reinstalling or starting a model. A pending transfer approval
skips only transfer tasks. After task preparation finishes, enable
`START_INFERENCE=True` in the small-model evaluation cell; keep all API/training
switches off in no-key mode.

To retry:

1. Save a fresh copy of the updated notebook and select **Runtime → Change runtime
   type → GPU**. Run its setup cells so the clone uses the updated code revision.
2. Keep `SKIP_API_PHASES=True` and all start switches off. Mount Drive and run
   preflight. Preserve the earlier failed attempt.
3. Expect the preflight diagnostic to say `status: passed` and `GPU_READY=True`.
   If it fails, share the diagnostic and preflight cell output, with secrets removed.
4. Only after preflight passes, enable `START_INFERENCE=True` if you want Qwen
   evaluation. Export actual artifacts afterward. API-dependent phases remain
   skipped, and the submission remains partial.

## Continue without API keys

The notebook defaults to `SKIP_API_PHASES=True`. It does not load API secrets and
clears their environment variables. Use a fresh notebook copy with the updated
repository revision, select a GPU, mount Drive and run setup/preflight. Leave the
API switches off; enable only `START_INFERENCE=True` to run Qwen. Transfer tasks
still require independent approval and a matching freeze.

The updated notebook pins tested source revision
`03d5e082c451d84375f67b1dad8c0a8096140ab0`, uses explicit `Path` arguments and
records validation/cohort preparation. Inference requires both `GPU_READY` and
`DATA_READY`. An absent transfer review is recorded as pending without a traceback.
To evaluate approved transfer tasks later, select the exact commit containing
their signed freeze marker.

This mode writes to `/content/drive/MyDrive/aster-gym-results-no-api-<revision>`,
using the first twelve characters of the code revision. Earlier evidence stays
in its original directory; an unchanged rerun can resume the matching directory. It records
training, remote comparisons and judge reliability as skipped. The same scorer
still applies: eligible blocking explanations have `status: pending` and
`score: null` without a judge. Other outputs receive their deterministic score.
No weights are removed, no missing ratings become zero, and incomplete coverage
cannot establish a complete ranking. Qwen results exist only after execution.

Export and import the genuine evidence using the commands below. This can support
a clearly disclosed partial submission; it does not complete the required RL or
judge study. Training remains skipped until judge access is available.

## Full workflow when keys are available

1. Complete the independent task reviews and fifteen judge labels. Freeze the five
   transfer tasks with `uv run --extra server aster-gym freeze-transfer`, then commit and push.
2. Save your own notebook copy. Set `REPO_REF` to that reviewed commit SHA. The
   notebook resolves and prints `CODE_REVISION`, checks out that exact revision,
   and refuses to overwrite tracked edits.
3. Select **Runtime → Change runtime type → GPU** and mount Drive. Allow at least
   5 GiB of cloud filesystem space; checkpoints remain on Drive.
4. Set `SKIP_API_PHASES=False`. Add `JUDGE_API_KEY` and `MODEL_API_KEY` in Colab Secrets and enable notebook
   access. Both may contain the same Google API key. Never paste keys into cells.
5. Verify model access, quotas, zero-dollar pricing and disabled billing in your
   account. Only then enable `FREE_TIER_CONFIRMED` for the judge and
   `REMOTE_FREE_TIER_CONFIRMED` for the two comparison models. Prices fail closed
   until confirmed. Keep the hard cost cap at zero.

Full mode defaults to `/content/drive/MyDrive/aster-gym-results-<revision>`. Use a fresh
output directory when adding a judge or changing configuration; a no-judge run
cannot resume under a different judge identity. Preserve its original evidence.

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
  the exact human-approved freeze marker verifies. It needs GPU preflight but no
  API key; no-key judge-dependent scores stay pending.
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
