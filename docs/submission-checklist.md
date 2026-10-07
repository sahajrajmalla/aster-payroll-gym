# Submission checklist

**Implementation checked; submission incomplete.** Public Tier-2/3 answers scored
1.0 and survived an owner-confirmed Render restart. Eleven API contract checks and
no-clone round trips passed. [Hosted evidence](render-sandbox-acceptance.json).
The latest owner-supplied Colab run passed GPU, dataset and cohort preparation;
its API-dependent experiments were skipped. Setup is not training evidence.

## Completed

- [x] Written rules, strict schemas, independent reference and frozen datasets.
- [x] Single-turn/tool environments, shared scorer, gates and adversarial tests.
- [x] Evaluation CLI, cost controls, transcripts and deterministic baselines.
- [x] Genuine Gemini 3.5 and Qwen initial-policy comparisons: 30 tasks × 3 rollouts
  each, reported separately because judge/sampling settings differ.
- [x] Public dashboard, Render/Neon sandbox and hosted restart persistence.
- [x] Guarded Colab workflows, resumable cloud checkpoints and safe result import.
- [x] Required documentation, review worksheets, CI and lightweight QA.

## Required evidence still outstanding

- [ ] Independent seed checks and >=10 generated-task rule-fidelity reviews.
- [ ] Five transfer-task approvals/freeze and measured ranking agreement.
- [ ] Complete three-model comparison and small-model 120-task evaluation.
- [ ] Judge reliability measurements and independent review labels.
- [ ] Two actual KL sweeps, all five curves and held-out initial/trained comparison.
- [ ] Actual reward-gaming inspection with counts and transcript references.
- [ ] Personal reading/classification of 5–10 genuine failures.
- [ ] Final experiment audit, timestamps and total resource spend.
- [ ] 8–12-minute Loom: >=3 minutes rewards, >=1 minute real failure, live sandbox.

Do not check an experiment complete from a successful setup command. No-key mode
supports partial Qwen evidence; it intentionally leaves RL and judge work pending.
Use [Colab instructions](colab.md), [human review](human-review.md) and
[Loom script](loom-script.md). If submitting with gaps, state them clearly.

## Final links and email draft

Subject: **AI Product Operator Assignment - Sahaj Raj Malla**

Repository: https://github.com/sahajrajmalla/aster-payroll-gym

Dashboard: https://sahajrajmalla.com.np/aster-payroll-gym/

Sandbox: https://aster-payroll-gym.onrender.com

Loom: add your reviewed recording URL.

Include one measured finding and remaining limitations. Verify all four links in
a private browser window. The deadline depends on the actual assignment receipt
date. Send the email yourself after reviewing the evidence and recording.
