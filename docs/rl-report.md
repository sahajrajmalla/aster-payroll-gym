# RL training report

**Status: Colab preflight failed; RL evidence remains pending. No training or
model inference ran locally.**

The owner supplied a [Colab attempt record](colab-run-status-2026-10-07.json) on
2026-10-07, at revision `91366b4b07b60392c637ef2d615b5df58e10f4f6`.
`gpu-preflight` exited 1; training, remote comparison and judge reliability were
explicitly skipped with `API_PHASES_SKIPPED`. The record contains no Qwen outputs,
training steps, curves or held-out scores. Its failure status is valid evidence
of an unsuccessful attempt, not a completed experiment.

The record does not include stderr, so it cannot establish whether CUDA was absent.
A code audit found a probable preflight issue: the isolated `uv` environment may
not find the native Colab package. The guard now also recognizes that package in
bounded system locations without importing host dependencies or enabling local
execution. A new preflight diagnostic records sanitized check outcomes. The exact
cause of this historical failure remains unconfirmed; a successful cloud rerun is
still required. Follow [the rerun steps](colab.md#recorded-attempt-and-preflight-rerun).

The default notebook mode, `SKIP_API_PHASES=True`, intentionally skips RL because
judge credentials are unavailable. It can run Qwen evaluation on Colab CUDA with
the unchanged scorer; this is inference, not training evidence. Eligible blocking
explanations remain pending without a judge. No Qwen results are claimed before
its artifacts are executed and imported. A partial submission must disclose that
the required KL sweeps and held-out policy comparison are not completed.

The executable cloud workflow uses Qwen2.5-0.5B-Instruct, pinned revision, LoRA and
GRPO. See README for the implemented clipped objective and k3 KL estimator, and
docs/colab.md for configuration, GPU guard, four-completion groups, reference proof,
checkpointing and recovery. Both beta runs start from the same initial policy.

Required evidence to import: reward, KL (and beta×KL), policy entropy, completion
length, tier pass rates at logged steps; equal-reward group fraction; reference
weight hashes; changed adapters; full completion transcripts; cloud checkpoint
locations; validation-only beta choice; final initial-versus-trained held-out
scores with three stochastic rollouts and split fingerprints.

No measured improvement, collapse, spend or reward gaming is claimed yet. After
execution fill: steps actually completed per beta, chosen beta and justification,
before/after mean±SD by tier, raw versus penalized reward, entropy/KL/length trends,
degenerate group fraction and uncertainty. Record the actual GPU and wall-clock time.

Gaming search must inspect high-reward wrong content, always-abstain behavior,
modal payslips, copied clause lists, valid-JSON nonsense, excess length, prompt
injection into explanations and truncation. Preserve run/task/step/transcript IDs.
If none is found, state how many outputs and which attacks were checked; do not
invent an exploit to fit an expected story. If found, version the reward fix and
rescore both compared policies under that version; preserve original evidence.

Training failure does not block remaining code or deployment preparation. Export
failed/partial metadata and last checkpoint location. Resume the same configuration
or use an explicitly documented smaller symmetric sweep, e.g. forty steps per beta.
Never represent smoke or partial training as a full sweep or fill missing curves.
