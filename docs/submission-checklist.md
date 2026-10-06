# Submission checklist

For an ordered explanation of these actions, read
[the submission walkthrough](submission-walkthrough.md). For recording and interview
practice, use [the Loom script](loom-script.md) and
[the project explanation](project-understanding.md).

**Current state: implementation checked; submission not ready.** The public
dashboard contains genuine remote results and programmatic baselines. Your permanent Neon project and Gemini key are
configured privately. Correct independent Tier-2/3 answers both scored **1.0** with
the real judge, and both persisted after a full local API restart
([proof](permanent-neon-persistence-smoke.json)). This is not public Render acceptance.

Next, deploy the **Docker Web Service** on Render using the
[deployment guide](deployment.md), then provide its public URL for external checks.
Independent reviews, Colab experiments and final evidence audits remain required.

## Implementation readiness

- [x] Separate written authority, strict schemas and deterministic Python reference.
- [x] Twelve seed tasks, three trap families, frozen seed-separated partitions.
- [x] Single-turn/tool environment, bounded parser/tools, shared auditable scorer.
- [x] Resumable remote eval and cost controls, actual programmatic baselines.
- [x] Independent submission API, persistence, quotas and oracle-safe responses.
- [x] Static dashboard, pending learning states and transcript viewer.
- [x] Guarded Colab scripts/notebook, cloud checkpoints, safe result handoff.
- [x] Documentation, review packets and regression tests.
- [x] Public GitHub repository and results dashboard; green GitHub CI.
- [x] Clean-clone install/startup and independent public-contract Tier-2 local smoke.
- [x] Permanent remote Neon: correct Tier-2/3 scores 1.0, real judge and process restart.
- [ ] Actual optional verifiers and GPU startup verified in Colab.

## Submission readiness — must not be silently skipped

- [ ] Your independent 12-seed review and >=10 generated-rule fidelity audit.
- [ ] Your authorship/approval of five transfer tasks, frozen before evaluation.
- [x] Flash-Lite remote comparison: 30 tasks × 3 rollouts, real mean±SD and transcripts.
- [ ] Complete Flash/frontier and Colab small-model comparisons: three model configs.
- [ ] Non-degenerate real score distribution over >=100 generated tasks.
- [ ] Judge repeated reliability and/or 15 human labels, recorded disagreements.
- [ ] Two real positive-beta training sweeps, frozen reference and all five curves.
- [ ] Never-trained held-out initial/after comparison with separate seeds.
- [ ] Actual reward-gaming evidence or explicit completed search with counts.
- [ ] Your reading of 5–10 real failures and counted taxonomy.
- [ ] Measured five-task transfer ranking agreement.
- [x] GitHub repo and public results dashboard, tested externally.
- [ ] Public Render/Neon sandbox, tested externally and after restart.
- [ ] One-command no-clone sandbox round trip including Tier2/3 correct submission.
- [ ] 8–12 minute Loom: >=3min rewards, >=1min real failure aloud, live sandbox.
- [ ] Four real links replace README placeholders; known limitations visible.
- [ ] Actual receipt/submission timestamps and total compute spend recorded.

Submission email draft (do not send automatically):

Subject: AI Product Operator Assignment - [Your Name]

Repository: [URL]

Dashboard: [URL]

Sandbox: [URL]

Loom: [URL]

Mention actual domain, strongest measured finding and any remaining limitation.
The five-business-day deadline depends on the actual receipt date; it is not
inferred from the workspace creation date. Speed points do not justify fake evidence.
