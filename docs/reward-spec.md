# Reward specification — 1.0

The same async scoring function serves every path. Written clauses define authority.
The oracle uses Decimal half-up to cents; comparisons tolerate zero cent deviation.

For determined work: exact 0.60, field accuracy 0.25, appropriate action 0.10, strict
format 0.05. For unresolved work: exact blocker/citation match 0.55, blocker/field
set F1 0.25, correct action/content 0.10, explanation judge 0.05, format 0.05.
Applicability follows rule-determined disposition, not a supplied difficulty label.

Exact requires complete substantive content and relevant clause coverage. Field
accuracy penalizes extra result keys. Trap partial credit averages issue-code F1
and affected-field F1. Calibration rewards matching the evidence-supported action
and content; unnecessary abstention is always zero, without a separate safety bonus.

Invalid contract, unsafe computed trap answer and unjustified abstention score 0.
Rejected substantive answers earn at most min(0.20, 0.25 × field accuracy), excluding
format/action bonuses. Thus polished JSON cannot conceal a wrong net amount. A
missing or unusable correct blocking explanation caps the result at 0.20. With an
adequate judge label (1), an otherwise correct trap scores 0.975; an excellent label
(2) scores 1. Fully reviewed ordinary work scores 1. Scores below 0.20 are not passes.
Training pass-rate uses score >=0.975, distinguishing graded progress from approval.

Every component records its score, weight, clause IDs and fixed diagnostic code.
The reference result is never put in these public diagnostics. The three handwritten
baselines attack format-only, constant financial content and unsupported abstention.
Their exact repeated outputs have SD zero by design, not because models are stable.

Judge prompt `prompts/judge-v1.txt` is versioned and hashed. The configured remote
model assigns only 0/1/2 labels for understandable issue/next action; Python already
decided correctness. Cache key includes task, canonical candidate, all scoring
versions, model, endpoint, prompt hash, reasoning effort and token budget. Google
Gemini requests use documented `low` effort and a bounded 512-token allowance for
reasoning plus the strict label, configured through `JUDGE_MAX_TOKENS` (64–2048)
and `JUDGE_REASONING_EFFORT`. Other endpoints omit effort unless explicitly set.
These settings are hashed in evaluation/training resumes, private sandbox state
and reliability studies, and recorded in each judge ledger. A changed setting
requires a fresh experiment or blocker-task run. No alternative judge or renormalized weights are
used on outages. Pending scores are excluded from mean rewards and counted in coverage.

Judge reliability remains pending. Label 15 examples yourself, obtain three uncached
ratings each, report human agreement and fraction with inconsistent repeated labels.
Stability alone is not validity. If prompt changes, version it and rerun evidence.

The low correctness cap protects against reward optimization, but judge false
negatives still affect correct traps through the explanation gate. Report that risk
and disagreements rather than presenting the LLM judge as truth. Preserve original
results when a reward fix is introduced and rescore compared policies consistently.
