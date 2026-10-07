# Submission status

**Code and deployed services checked; experimental submission incomplete.**
The sandbox accepts correct Tier-2/3 submissions with score 1.0 and retains runs
after restart. [Hosted evidence](render-sandbox-acceptance.json).

## Completed

- Written rules, schemas, deterministic ground truth and disjoint datasets.
- Single-turn/tool environments, shared reward, hard gates and adversarial tests.
- Resumable evaluation, transcripts, cost limits and three programmatic baselines.
- Qwen initial-policy inference: 486 samples on Colab, zero passes.
- Gemini full evaluation: 120 tasks × 3 rollouts, 360 answers, 32 passes;
  reward **0.12049 ± 0.00732** across rollout means.
- Judge repeat study: 45 responses, one malformed; 2/14 valid triples disagree.
- Automated numerical cross-check: 27 inputs, no substantive divergences.
- Initial-policy output audit: 975 answers, 41 high-reward cases, no deterministic bypass.
- Dashboard, Render/Neon API, hosted persistence, clean-install checks and CI.
- Cloud training implementation, configuration, recovery and safe artifact handoff.

## Requested evidence status

1. **Personal task checks, judge labels and failure reviews: NOT DONE.**
   Numerical cross-checks and ten technical failure analyses are available; signed
   independent applicant reviews and human judge agreement are absent.
2. **Two RL sweeps, curves and held-out comparisons: NOT DONE.**
   The workflow is implemented. No completed training run or learning improvement is claimed.
3. **Compatible three-model comparison with a frontier model: NOT DONE.**
   Qwen and Gemini runs exist but differ in judge/sampling configuration. Frontier
   attempts are partial or operational failures; they cannot establish a controlled ranking.
4. **Approved transfer tasks and measured transfer results: NOT DONE.**
   Five drafts exist; approval, freeze and comparable model outcomes are absent.
5. **Final experiment/resource-spend audit: PARTIAL.**
   Available runs, checksums, scores, timestamps and recorded API spend ($0) are
   audited. Overall GPU charges are unverified, and no RL/transfer audit exists.

See [evaluation](evaluation.md), [RL report](rl-report.md), [transfer](advanced-track.md)
and [available-experiment audit](final-experiment-audit.json). These gaps must accompany
submission. [Review procedure](human-review.md) and [Colab](colab.md) describe completion.

## Submission links

- Repository: https://github.com/sahajrajmalla/aster-payroll-gym
- Dashboard: https://sahajrajmalla.com.np/aster-payroll-gym/
- Sandbox: https://aster-payroll-gym.onrender.com
- Loom: add an 8–12-minute recording with ≥3 minutes on reward design,
  ≥1 minute reading a genuine failure and a live sandbox request.

Include the separate tooling statement and verify all share links. Email subject:
**AI Product Operator Assignment - Sahaj Raj Malla**. Submit the measured work with
these limitations; do not mark missing experiments complete.
