# Submission checklist

For an ordered explanation of these actions, read
[the submission walkthrough](submission-walkthrough.md). For recording and interview
practice, use [the Loom script](loom-script.md) and
[the project explanation](project-understanding.md).

**Current state: implementation checked; submission not ready.** The public
baseline dashboard is live. A correct Tier-2 submission and full API restart
passed using real Neon Postgres, with the API running locally
([proof](neon-persistence-smoke.json)). Render deployment and genuine experiments
still need account access. This does not count as a deployed sandbox.

Next, follow the [deployment guide](deployment.md): create your own free Neon
project, sign into Render, and provide verified-free provider secrets. The temporary
test database expires **8 October 2026, 15:26 Nepal time**; use your own project.
Then complete the independent reviews and explicitly run the Colab workflow.
Finish the evidence-dependent items below before recording Loom.

## Implementation readiness

- [x] Separate written authority, strict schemas and deterministic Python reference.
- [x] Twelve seed tasks, three trap families, frozen seed-separated partitions.
- [x] Single-turn/tool environment, bounded parser/tools, shared auditable scorer.
- [x] Resumable remote eval and cost controls, actual programmatic baselines.
- [x] Independent submission API, persistence, quotas and oracle-safe responses.
- [x] Static dashboard, pending learning states and transcript viewer.
- [x] Guarded Colab scripts/notebook, cloud checkpoints, safe result handoff.
- [x] Documentation, review packets and regression tests.
- [x] Public GitHub repository and baseline dashboard; green GitHub CI.
- [x] Clean-clone install/startup and independent public-contract Tier-2 local smoke.
- [x] Real remote Neon storage: correct Tier-2 score 1.0 and full process restart.
- [ ] Actual optional verifiers and GPU startup verified in Colab.

## Submission readiness — must not be silently skipped

- [ ] Your independent 12-seed review and >=10 generated-rule fidelity audit.
- [ ] Your authorship/approval of five transfer tasks, frozen before evaluation.
- [ ] Three genuine model configs, three stochastic rollouts, mean±SD.
- [ ] Non-degenerate real score distribution over >=100 generated tasks.
- [ ] Judge repeated reliability and/or 15 human labels, recorded disagreements.
- [ ] Two real positive-beta training sweeps, frozen reference and all five curves.
- [ ] Never-trained held-out initial/after comparison with separate seeds.
- [ ] Actual reward-gaming evidence or explicit completed search with counts.
- [ ] Your reading of 5–10 real failures and counted taxonomy.
- [ ] Measured five-task transfer ranking agreement.
- [x] GitHub repo and public baseline dashboard, tested externally.
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
