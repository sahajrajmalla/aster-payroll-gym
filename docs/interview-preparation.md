# Interview preparation

Practise answering in your own words. Show a rule, code path or saved transcript
when challenged. Measured claims must come from actual records; check
[submission status](submission-checklist.md) before rehearsing.

## Opening statement

> I built a narrow synthetic payroll gym. A versioned written authority and an
> independent Python reference define correctness. Tasks test retrieval,
> calculations and justified clarification. Shared scoring connects evaluation,
> cloud RL and an independent sandbox. The central question is whether the reward
> measures useful work and whether improvement survives held-out evaluation.

## Twenty questions to prepare

**1. What was the assignment?** Build an end-to-end training environment, including
an auditable reward, model evaluation, small-model RL, dashboard, independent
sandbox and one advanced experiment. The deliverable is reproducible evidence,
not merely an API or notebook.

**2. Why payroll?** It naturally combines authority, effective dates, chained
arithmetic and missing-information decisions. Fictional rules provide deterministic
truth without private data. [Rules](../rules/aster-payroll-v1.md).

**3. Why narrow scope?** One monthly gross-to-net workflow is feasible to verify.
Extra features would introduce ambiguity without improving the required evidence.
It is a research environment, not a production payroll service.

**4. Where does ground truth come from?** The independent Python reference applies
R1–R11 using integer cents and Decimal half-up rounding. No model creates labels.
Tests and independent rule-fidelity review are still needed. [Reference](../src/aster_gym/reference.py).

**5. What do the tiers mean?** Tier 1 retrieves schedule fields; Tier 2 computes
interacting payroll lines; Tier 3 recognizes evidence blockers. Difficulty labels
never determine whether the reference says answer or clarify.

**6. What are the three traps?** Missing prior-year-to-date pensionable earnings,
conflicting equally authoritative salaries, and no schedule effective on the pay
date. The answer identifies all blockers and needed fields rather than guessing.

**7. How do you prevent leakage?** Model prompts and API responses use public
allowlists. Private labels and provenance remain server-side. Splits have separate
seed namespaces and normalized business-input fingerprints. Repeated templates
remain a disclosed limitation. [Schemas](task-schema.md).

**8. Why strict JSON?** A fixed six-key contract makes results auditable. Parsing
rejects duplicate keys, wrappers, malformed JSON, extra fields and invalid money
types. Correct syntax alone earns no positive total reward.

**9. Why these reward weights?** Correctness dominates; field accuracy supplies
bounded feedback; action rewards answering or clarifying appropriately. Explanation
judging contributes only to verified blocking responses. [Reward spec](reward-spec.md).

**10. Why hard gates?** Weighted sums alone can reward polished errors. Invalid
output, unsafe computation and unnecessary refusal score zero. Substantive
mismatches are capped at .20, far below the .975 pass threshold.

**11. What does the judge decide?** Only whether a correct blocking explanation
communicates the issue and useful next action. It cannot calculate pay or reverse
Python rejection. Missing judge service produces pending scoring.

**12. How is judge reliability checked?** Independently label fifteen examples,
then obtain three uncached ratings each. Report exact human agreement, confusion
counts and repeat instability. Consistency does not prove validity.

**13. What attacks were tested?** Format-only, constant-answer and always-abstain
policies, plus injection, citations, length, malformed JSON, hidden-answer and
operational failure cases. Baseline tests are not model benchmarks or proof that
all future exploits are impossible.

**14. Why three rollouts?** Sampling varies. Compare identical cohorts with three
complete replicate means and sample SD, while reporting coverage and task-level
variation. SD is not a confidence interval. [Evaluation](evaluation.md).

**15. How are cost and failures handled?** Reserve conservative cost before calls,
include retries and judge usage, and enforce a hard cap. Unknown prices fail closed.
Infrastructure failures remain separate from incorrect answers; resume preserves
completed answers and their original costs.

**16. What does GRPO do?** It compares four sampled answers to each prompt through
relative group rewards, then updates the policy with a clipped objective. Groups
with identical rewards provide no reward preference; log their fraction.

**17. Why LoRA and a frozen reference?** LoRA trains small adapters while freezing
the backbone. The initial policy regularizes learning through KL. Verify initial
logit equivalence and unchanged reference hashes in the actual cloud run.

**18. Why compare two KL coefficients?** A lower penalty permits more movement;
a higher penalty restricts it. Neither is automatically best. Select beta using
validation only, then compare initial and selected policies on untouched tasks.
[RL report](rl-report.md).

**19. How do you detect reward gaming?** Read high-reward outputs, not only curves.
Look for blanket refusal, copied clauses, repeated payslips, format gains, excessive
length and truncation. Record inspected sample size and actual findings, including
negative or inconclusive findings.

**20. What does transfer prove?** Five human-reviewed, frozen presentations use the
same rules. Compare ranking agreement and task-level reversals across the same
three policies. This small sample probes template dependence; it cannot establish
broad real-world generalization. [Transfer](advanced-track.md).

## Demonstration and limitations

Explain the worked example in [project-understanding.md](project-understanding.md),
then remove YTD and show why the decision changes. Demonstrate public task issuance,
correct independent submission and persisted retrieval; never import the private
reference into that client. Explain identical retries and pending judge responses.

A good limitation answer is: “The rules are synthetic, retrieval has few underlying
templates, the transfer sample is small, and judge errors can affect correct
blockers. Those limits are reported alongside coverage and genuine outcomes.”

Before interview, fill your three-model comparison, selected beta, held-out
before/after, judge agreement, one failure ID, gaming review and transfer findings
from saved evidence. If something is pending, say precisely what remains.
