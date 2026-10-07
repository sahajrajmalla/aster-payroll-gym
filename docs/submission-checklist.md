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
- [x] Qwen 120-task evaluation (360 samples) and 12-task tool cohort (36 samples),
  imported with checksums and raw transcripts; zero passes.
- [x] Public dashboard, Render/Neon sandbox and hosted restart persistence.
- [x] Guarded Colab workflows, resumable cloud checkpoints and safe result import.
- [x] Required documentation, review worksheets, CI and lightweight QA.
- [x] Independent automated numerical audit: 27 inputs, zero substantive divergences.
- [x] Non-degenerate genuine distribution: 120 tasks × 3 rollouts, 360 answers, rewards 0–1.
- [x] Judge repeat study: 45 responses, one malformed; 2/14 valid triples disagree.
- [x] Initial-policy gaming inspection: 975 outputs, 41 high-reward cases, no deterministic bypass.
- [x] Available-experiment audit, checksums, timestamps and recorded API spend ($0).

## Required evidence still outstanding

- [ ] Independent seed checks and >=10 generated-task rule-fidelity reviews.
- [ ] Five transfer-task approvals/freeze and measured ranking agreement.
- [ ] Complete compatible three-model comparison, including a frontier policy.
- [ ] Independent human judge labels and agreement (repeat stability is measured).
- [ ] Two actual KL sweeps, all five curves and held-out initial/trained comparison.
- [ ] Trained-policy reward-gaming inspection (initial-policy inspection is recorded).
- [ ] Personal reading/classification of 5–10 genuine failures.
- [ ] Final audit after RL/transfer; independent confirmation of Colab resource spend.
- [ ] 8–12-minute Loom: >=3 minutes rewards, >=1 minute real failure, live sandbox.

Do not check an experiment complete from a successful setup command. No-key mode
supports partial Qwen evidence; it intentionally leaves mixed-tier RL pending.
Remote judge repeat stability has since been measured; human labels are separate.
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
