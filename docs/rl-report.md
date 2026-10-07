# RL training report

**Not done: no RL sweep, learning curves or held-out trained-policy results.**
The [Colab preflight](colab-final-preflight.json) passed on a Tesla T4 with pinned
dependencies. [Execution records](colab-final-run-status.json) show skipped training
and completed initial-policy inference. No training, inference or weight downloads
ran on the local development machine.

## Implemented workflow

Qwen2.5-0.5B-Instruct uses LoRA rank 8/alpha 16 and GRPO, with four completions per
prompt, temperature 0.8, 256 completion tokens and learning rate 5e-5. Both beta
values (0.001 and 0.10) start from the same initial policy; default length is
80 optimizer steps per beta, with checkpoints every 20 steps. The frozen initial
reference is checked for equivalence and immutability. The clipped objective and
k3 KL estimator are in the README; [Colab](colab.md) documents execution and resume.
Validation selects beta before held-out evaluation with three rollouts per task.

Logs are designed to capture reward, KL and beta×KL, entropy, completion length,
per-tier pass rate, reward components, equal-reward group fraction and completions.
Checkpoints remain in cloud storage. A failed run preserves failure/resume metadata.
The training guard requires hosted Colab and CUDA before importing model libraries.

## Evidence and limitations

Qwen completed 486 initial-policy inference samples: 90 comparison, 360 full
evaluation and 36 tool answers. None passed. The full evaluation's all-zero reward
warns of flat group-relative advantages; a smoke run must establish usable reward
variation and actual adapter updates before a full sweep. Inference is not training.

The remote judge repeat study completed, but Colab secret access was not confirmed
for mixed-tier training. No replacement judge or simplified training reward is used.
Default notebook switches leave training off until explicitly started.

Missing artifacts: actual step logs and five curves, reference proof from a training
run, saved checkpoints, validation beta selection, held-out initial/trained scores
and trained-policy reward-gaming inspection. No improvement, collapse or trained
policy exploit is claimed. The [975-output audit](reward-gaming-audit.json) covers
initial policies only, not a trained policy.

A follow-up should smoke-test three steps, inspect reference/update evidence, run
both sweeps (or explicitly document equal shorter runs), compare held-out policies,
and inspect actual high-reward outputs for abstention, citation, format, length
and explanation hacking. Import real artifacts and regenerate the dashboard.
