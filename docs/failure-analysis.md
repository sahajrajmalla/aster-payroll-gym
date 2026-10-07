# Failure analysis

**Status: ten genuine failures are selected; independent human review is pending.**

The [review packet](../reviews/failure-review-packet.json) preserves exact outputs,
run/task/rollout identifiers, source-record checksums and clause-level scorer
diagnostics. It covers two Tier-1, three Tier-2 and five Tier-3 outputs from two
remote model configurations. Selection is purposive, so its categories are not
population frequency estimates. Suggested categories and remedies remain drafts;
all reviewer fields are null and all rows are unreviewed.

The finalized Gemini 3.5 Flash-Lite comparison contains 90 completed answers,
82 below the **0.975 pass threshold**: 63
`INVALID_CONTRACT` and 19 `CORRECTNESS_CAP` gates. Rate-limited samples were resumed
without rerunning completed answers. The interrupted Gemini 3.8 run has eight
completed answers, all below threshold, plus 43 observed operational failures;
39 planned samples did not complete. Its incomplete replicates cannot establish
a model ranking. The packet retains its earlier selection snapshot and all ten
selected completed outputs remain unchanged.

A concrete arithmetic failure is Gemini 3.8, run `c697a5f3d20e737d05ae`, task
`task-0ec7d48247ff1b45b74b`, zero-based rollout 0. Its
[saved output, packet row `failure-02`](../reviews/failure-review-packet.json)
reports gross **522,564**, tax **64,513**, and net **458,051** cents. The independent
Python reference for the [frozen task](../data/evaluation.jsonl) gives **522,587**,
**62,517**, and **460,070** respectively. Only the contribution field matches.
The shared scorer records field accuracy 0.20, `SUBSTANTIVE_MISMATCH`, and a
`CORRECTNESS_CAP` reward of **0.05**; valid JSON and citations cannot rescue wrong
arithmetic. The draft remedy is calculator-assisted proration and marginal-band
arithmetic, followed by a check of all five fields under R1/R3/R5/R6/R7.

Other selected outputs show a scalar retrieval result, truncated JSON, markdown
fences, missing citations, misspelled issue codes, and `schedule` instead of the
canonical `schedules` input field. These are answer failures, not provider 429s.
Existing strict-parser and reward tests exercise these failure families; no prompt
change or observed model improvement is claimed.

Read the ten rows against the rule text before recording your name, final category,
clauses, whether the reward caught the failure, and remedy. Then regenerate analysis
with `uv run aster-gym analyze`. Report reviewed category counts separately from
the snapshot gate counts above. Multiple tags never create additional independent
failures.

## Qwen baseline observation

The accepted Colab comparison contains 90 completed outputs: 89
`INVALID_CONTRACT` and one `CORRECTNESS_CAP` (.125). All are below .975. These
are automated gate counts, not independent human classifications.

For example, task `task-04f13390131199f41682`, rollout 0, is a Tier-1 allowance
retrieval. Its [raw transcript](../results/qwen-colab-comparison/transcript.jsonl)
returns a Markdown-fenced `needs_information` object, claims `SALARY_CONFLICT`,
and supplies citation objects instead of clause-ID strings. The salary value is
irrelevant to this retrieval. R10 rejects the contract before substantive scoring.
Removing fences alone would not correct the diagnosis or citation structure.
The unchanged prompt already specified six properties and no Markdown. A revised
prompt or training would require a new, separately recorded experiment; no
improvement is inferred from this failure inspection.

The full 120-task Qwen run adds 360 completed answers: 357 invalid contracts,
two unjustified abstentions and one incorrect blocker diagnosis with reward zero.
For example, task `task-769ac5a5d96d99333ab3`, rollout 2, refuses a solvable task
and claims unsigned salary evidence can be authoritative. That contradicts R2;
R9 rejects unnecessary abstention. In the 36 tool samples, every answer failed
`TOOLS_NOT_USED`. Task `task-0e33888836c453dcb91e`, rollout 0, claims payroll
information was obtained from documents despite zero reference reads. This is a
fabricated evidence claim, caught by the mandatory-tool gate.

[Deterministic replay](qwen-final-reward-audit.json) checked all 486 Qwen records,
including scores, component diagnostics and tool eligibility, with zero mismatches.
This audit does not replace the required human reading or a trained-policy gaming
study. No Qwen output passed, so there is no high-reward wrong Qwen answer in these
runs. The flat larger distribution is an unresolved experimental limitation.
