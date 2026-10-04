# RL training report

**Status: pending Colab execution. No training or model inference ran locally.**

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
