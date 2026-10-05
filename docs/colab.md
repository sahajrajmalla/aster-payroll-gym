# Colab-only execution and return of results

No training or open-weight inference has been run on the development computer. The
training implementation is runnable code, but GPU compatibility and experimental
outcomes remain **pending execution in Colab**. Local tests exercise guards,
configuration, and safe data import only. Do not interpret passing those tests as
successful training.

## Start safely

Open `notebooks/aster_colab.ipynb` in Google Colab and select a **GPU** runtime.
The first cell refuses to run outside recognized hosted Colab. It checks Linux,
`/content`, the installed `google.colab` package, and a Colab environment marker.
The training and inference entrypoints independently enforce the same boundary
before importing PyTorch. After that, CUDA must be available. There is no CPU,
MPS, local-force flag, or notebook-only bypass.

1. Open the published notebook and save your own Colab copy. Its `REPO_URL` already
   points to the public repository. Push your completed review/freeze commit first
   and confirm Colab clones that exact version.
2. Clone the repo in the notebook. Install the committed lockfile's `cloud` extra
   there. The laptop's default install never includes ML packages.
3. Mount Google Drive. Keep `output_dir` on Drive so checkpoint writes survive
   notebook disconnects. Check that there is several GB of available space.
4. Add `JUDGE_API_KEY` to Colab Secrets. Enable notebook access. Confirm that your
   judge model is available on a verified zero-dollar tier with billing disabled.
   Set `JUDGE_MODEL` accordingly and `JUDGE_VERIFIED_FREE=true` only after checking.
5. Run the guarded CUDA preflight and the explicit three-step smoke command.
6. Inspect the smoke status, real metrics, and reference proof. An unchanged
   adapter with all equal-reward groups is an ineffective smoke run, not an
   invented success. Preserve its evidence.
7. Explicitly start the two-beta sweep. Each beta begins from the pinned initial
   model. Validation chooses the winner before final evaluation.
8. Export only the compact JSON bundle and download it. Keep weights, optimizer,
   scheduler, tokenizer, and RNG checkpoints in Drive.

The model and tokenizer are pinned to `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`. This revision was resolved through the
[official model metadata](https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct).
No weights were downloaded to resolve it.

## Commands, only inside Colab

```bash
uv run --frozen --extra cloud --no-dev python -m aster_gym.cloud.train \
  --config configs/train.json --start-training --smoke
uv run --frozen --extra cloud --no-dev python -m aster_gym.cloud.train \
  --config configs/train.json --start-training
uv run --frozen --extra cloud --no-dev python -m aster_gym.cloud.train \
  --config configs/train.json --start-training --resume
uv run --frozen --extra cloud --no-dev python -m aster_gym.cloud.evaluate \
  --config configs/train.json --tasks data/evaluation.jsonl \
  --output /content/drive/MyDrive/aster-gym-results/small-model-eval --start-inference
```

Use the frozen 30-task common cohort and five transfer tasks for additional
small-model comparison runs. The `--tasks` flag chooses these JSONL files. The
cloud provider uses the exact same evaluator and scorer as the remote models.
There is no local model server or hidden inference fallback.

Create the common cohort without inference:

```bash
uv run --frozen --extra cloud --no-dev python - <<'PY'
from aster_gym.generator import read_taskset, write_taskset
write_taskset(read_taskset('data/evaluation.jsonl')[:30], 'data/comparison.jsonl')
write_taskset(read_taskset('data/evaluation.jsonl')[:12], 'data/tool-comparison.jsonl')
PY
```

Use `--tasks data/comparison.jsonl` for the small-model common comparison and
`--tasks data/tool-comparison.jsonl --mode tool` for its tool comparison. Before
the transfer run, finish your five review rows on the Mac, run
`uv run aster-gym freeze-transfer`, and commit/sync that marker to Colab. Transfer
scoring refuses to start without the exact approved task hash.

## Resource adjustments

`configs/train.json` controls output location, steps, microbatch, accumulation,
completion limit, checkpoint interval, and sampling. Defaults are 80 steps per
beta, batch 1 with accumulation 4, and four grouped completions. The effective
batch must remain divisible by four. Reduce **both beta runs** to 40 steps if
needed. If adjusting token length or any other setting, start both runs in a new
output directory and report the adjustment. Resuming a changed configuration,
rule hash, taskset, or prompt configuration is rejected.

The trainer writes an optimizer checkpoint every 20 steps and retains three.
A successful smoke writes checkpoints every step. `--resume` uses the latest
checkpoint and discards metric/completion rows beyond that durable step. A failed
run before its first checkpoint needs a new output directory. Never delete the
failed evidence just to make the results look cleaner.

The judge quota is a separate constraint: start conservatively; raise its call
limit only within your verified free allowance. Judge errors or quota exhaustion
stop a training batch without replacing rewards or training on `None`. Fix the
external issue and resume. Training failure does not block development or other
submission work.

## Implemented objective and evidence

The implementation explicitly selects TRL 0.26.2's `loss_type="grpo"`, grouped
reward scaling, one policy iteration, clipping epsilon 0.2, and positive beta.
For completion token `t`, let `r_t = exp(log πθ − log πold)` and `A` be the
completion's group-centered, group-standardized reward. It minimizes the negative
clipped surrogate `min(r_t A, clip(r_t, 0.8, 1.2) A)` plus `β KL_t`, averaging
over completion tokens and then sequences. Equal-reward groups have zero
advantages and are retained in the audit.

`KL_t = exp(log πref − log πθ) − (log πref − log πθ) − 1` is the sampled `k3`
estimator; `use_bias_correction_kl=False` is explicit. The reference is the frozen
adapter-disabled backbone. Initial enabled/disabled logits must agree on a probe;
all non-adapter parameters must be frozen; a full parameter digest must remain
unchanged after training. The saved proof also records whether adapter weights
actually changed. This does not claim every prompt's logits were exhaustively
compared; the frozen parameter hashes cover the whole reference.

Source compatibility was inspected against the [pinned trainer source](https://github.com/huggingface/trl/blob/v0.26.2/trl/trainer/grpo_trainer.py)
and [pinned configuration source](https://github.com/huggingface/trl/blob/v0.26.2/trl/trainer/grpo_config.py).
Actual GPU execution still needs the smoke test.

Logs contain reward, sampled KL, beta-times-KL, entropy, completion length,
pass rate by tier, reward-component means, fraction of equal-reward groups,
truncation, and full completion/score transcripts. Tier values are `null` when a
step's group did not contain that tier. Do not plot them as zeros. Training
completion summaries describe sampled training groups, not held-out performance.
Validation and held-out files use three independently sampled rollouts per task.

Beta selection uses highest validation mean, with the larger beta as the
predeclared tie-breaker. Only after selection does the runner evaluate the
initial and selected policies against the frozen evaluation split. It records
IDs, input fingerprints, seeds, taskset hashes, prompt hashes, and rule/reward
versions. Stop if a fingerprint or seed crosses split boundaries.

## Bring back results

The notebook calls `export_bundle` to produce a ZIP containing only allowlisted
JSON and JSONL evidence. Model states never enter that ZIP. Checkpoint metadata
contains cloud paths, byte counts, and SHA-256 hashes, not downloadable weights.

On your computer run the lightweight CLI:

```bash
uv run aster-gym import-results --bundle downloaded-results.zip --output results/colab-import
```

The import destination must be new. The importer validates paths, member count,
uncompressed size, compression ratio, symlinks, encryption, checksums, versions,
JSON finite numbers and identities, transcript/score correspondence, split
separation, and required training curves/proofs. No extraction of executable or
pickle payloads is permitted. Fixtures are rejected by default. Aggregate
metrics are recomputed from records using the same evaluator summarizer.

Imported self-reported model outputs are evidence, not a cryptographically
attested proof of where a model ran. Preserve the notebook and cloud logs, code
revision, and checkpoints for reviewer inspection. After import, regenerate the
dashboard and complete the RL, failure, and advanced-track reports with actual
numbers. Incomplete runs stay visibly incomplete.
