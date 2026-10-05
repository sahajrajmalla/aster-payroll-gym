# Track B — Transfer test

**Status: five authored drafts exist; your edits/approval and measured outcomes are pending.**

The tasks keep ASTER-1.0 rules while varying presentation: narrative handover,
unsigned/signed evidence precedence, conflicting amendments, missing prior register,
and next-year schedule distractor. They are manually assembled rather than sampled
from generator templates; your independent edits/approval are required before freezing.

Freeze data/transfer.jsonl before final outcomes are inspected. Evaluate the same
three configurations with three rollouts and the same reward/judge versions.
After your review, `uv run aster-gym freeze-transfer` records the exact task hash
and approval date; actual transfer model calls require this marker. Then run
`uv run aster-gym analyze` to write the computed ranking agreement and task-level
reversals to `results/analysis.json` and regenerate the dashboard.
Use the initial small model for the three-model comparison; report trained transfer
performance separately if collected. Compute generated-cohort ranking and transfer
ranking, Spearman rank correlation with average ranks for ties, and task-level
reversals. Never substitute baseline ranks for model ranks.

Five tasks support a limited diagnostic finding, not a statistically reliable claim
of real payroll readiness. Fill in actual rankings, correlation, failure examples,
cost and variance only from saved records. No transfer finding is claimed yet.
