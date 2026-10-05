# Interview reminder

**Pitch:** A narrow synthetic payroll gym with a written authority, independent
Python ground truth and shared clause-auditable scoring. It evaluates correct
computation and justified clarification, then tests cloud training on held-out tasks.

**Remember:**

- ASTER-1.0; clauses R1–R11; integer cents; Decimal half-up.
- Tiers: retrieval, interacting arithmetic, evidence blockers.
- Traps: missing YTD, equal-authority salary conflict, absent effective schedule.
- Splits: 12 seeds / 30 train / 15 validation / 120 evaluation / five transfer.
- Tools: `read_document`, `lookup_rules`, bounded `calculate`.
- Normal weights: .60 correctness / .25 fields / .10 action / .05 format.
- Blocked weights: .55 diagnosis / .25 fields / .10 action / .05 judge / .05 format.
- Invalid, unsafe computation and unjustified refusal: zero. Wrong substance: ≤.20.
- Pass: ≥.975. Judge outage: pending; no substitute score.
- Qwen0.5B + LoRA + GRPO group4; beta .001/.10; defaults80 steps each.
- Frozen initial reference; validation selects beta; untouched evaluation follows.
- Curves: reward, KL, entropy, length, per-tier pass rate.
- Track B only: five reviewed presentations frozen before evaluation.
- Sandbox: `/tasks`, `/submit`, `/runs/{run_id}`, `/healthz`; no expected answers.

**Example:** salary300,000 + bonus20,000 → gross320,000. YTD1,190,000 leaves
10,000 ceiling → contribution500. Taxable309,500 → tax21,900 → net297,600 cents.
Missing YTD changes the decision to `needs_information`, citing R4/R9/R10.

**Before speaking, fill from genuine records:** model means±SD/coverage; chosen beta;
held-out before/after; judge agreement; real failure ID/clause; gaming review count;
transfer ranking; public restart test; spend/latency. Pending is not zero.

If uncertain, identify the rule and artifact to inspect. Do not invent results.

[Status](submission-checklist.md) · [Questions](interview-preparation.md) ·
[Loom](loom-script.md).
