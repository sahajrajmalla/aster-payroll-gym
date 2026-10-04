# Failure analysis

**Status: real model failures and your manual review are pending.**

The deterministic baseline attacks have saved transcripts, but they do not replace
reading 5–10 failed model runs. Review at least two tiers and both arithmetic and
calibration problems when available. Distinguish operational errors from wrong work.

Suggested taxonomy: wrong schedule/authority; ceiling ignored; marginal tax treated
as flat tax; proration/rounding; omitted field/citation; unsafe trap computation;
unnecessary refusal; irrelevant blocking explanation; tool routing/loop; malformed
output. Add categories only when an actual transcript supports them.

For each reviewed failure, enter run_id, task_id, rollout/step, observed output,
clause violated, primary category, secondary tags, reviewer, and remedy in
`reviews/failure-review-packet.json`. Report category counts and denominator,
representative quotes, whether the reward caught it, and the corresponding fix/test.
Multiple tags do not increase the number of independent failures.
