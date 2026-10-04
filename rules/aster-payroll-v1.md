# Aster Payroll Rules — ASTER-1.0

Authoritative fictional specification. Written before the calculator. This is a
synthetic monthly payroll exercise, not tax or employment advice. If code and
this text disagree, the text wins. Clause identifiers are stable within a version.

## R1 — Units, periods, precision
All amounts use fictional AST currency. Inputs and outputs are integer cents;
booleans, floating-point numbers, non-finite values, negatives, and numeric strings
are not monetary inputs. Dates use ISO YYYY-MM-DD. A payroll period is one calendar
month containing the pay date. `period_days` is the actual number of calendar days
in that month; `paid_days` is an integer from zero through `period_days` inclusive.
Round each monetary calculation specified below to an integer cent using decimal
ROUND_HALF_UP. Compare final integer cents exactly: tolerance is zero cents.
Supported money inputs are at most 100,000,000 cents.

## R2 — Salary records and authority
Use signed salary records effective on or before the pay date. Among these, use
the latest effective date. Equal-date signed records must agree on monthly salary;
otherwise the required salary is unresolved. Ignore unsigned and future records.
There must be at least one applicable signed salary record. Mere stale distractors
do not justify abstention when a unique applicable record is available.

## R3 — Gross pay
The required inputs are the selected monthly salary, paid days, period days and
explicit bonus cents (zero must be explicit). Gross cents equal the rounded value
of `monthly_salary_cents * paid_days / period_days + bonus_cents`. The bonus is
fully taxable and pensionable. Do not round the prorated salary before adding it.

## R4 — Employee contribution
The year-to-date pensionable earnings before this payroll are required, even if
zero. They must refer to the pay date's calendar year. The remaining pensionable
ceiling is `max(annual_ceiling_cents - ytd_pensionable_cents, 0)`. Contribution is
the rounded value of `min(gross_cents, remaining_ceiling) * contribution_rate`.
There is no negative contribution or refund when the ceiling was already reached.

## R5 — Taxable pay
Taxable cents are `max(gross_cents - contribution_cents - allowance_cents, 0)`.
The allowance is monthly and is not prorated. Only the employee contribution and
the listed allowance reduce taxable pay.

## R6 — Marginal income tax
For each schedule band, tax only the taxable amount within that band, not all pay
at the highest crossed rate. Sum unrounded band amounts and round once at the end.
Boundaries belong to the lower band; a zero-width taxable slice contributes zero.

## R7 — Net pay
Net cents equal `gross_cents - contribution_cents - tax_cents`. A payslip reports
all five fields: gross, contribution, taxable, tax and net, each in integer cents.

## R8 — Schedule selection and the authoritative schedule
Select schedules whose inclusive effective interval contains the pay date. Exactly
one must apply. Never extend a stale schedule beyond its stated end date. No
applicable schedule is an unresolved input, not a zero rate. Conflicting overlapping
schedules are unresolved, even if a plausible rate could be guessed.

The ASTER-2030 schedule is effective from 2030-01-01 through 2030-12-31 inclusive.
Contribution rate is 5% (0.05); annual pensionable ceiling is 1,200,000 cents;
monthly allowance is 10,000 cents. Marginal bands: first 100,000 taxable cents at
0%; next 200,000 cents at 10%; all taxable cents above 300,000 at 20%.
The supplied ASTER-2029 schedule is a distractor, valid only from 2029-01-01 through
2029-12-31: contribution rate 4%, ceiling 1,000,000 cents, allowance 8,000 cents,
with the same marginal-band boundaries and rates. Versioned schedules are part
of this written authority; task documents cannot invent or override these rates.

## R9 — Undetermined work and calibration
Do not calculate money if required evidence is absent, inconsistent, invalid, or
unsupported by an effective rule. Return `decision: needs_information`, no result,
and the complete set of blocking issue codes and affected field names. Use:
`MISSING_INPUT` for an absent required input, `SALARY_CONFLICT` for equal-authority
salary conflicts, `NO_SALARY` for no applicable salary, `NO_SCHEDULE` for an absent
effective schedule, `SCHEDULE_CONFLICT` for overlapping applicable schedules,
`INVALID_INPUT` for invalid units/ranges/periods or unmatched YTD year.
Issues may coexist; report all independently identifiable issues. Solvable work
must be completed; unnecessary abstention is a wrong answer. Tier labels never
determine the correct disposition. Determine disposition from the evidence.

## R10 — Answer contract and explanation
Return one JSON object only, with exactly `decision`, `result`, `issue_codes`,
`missing_fields`, `citations`, and `explanation`. For retrieval/computation use
`decision: answer`, an object result, and empty issue/missing lists. For unresolved
work use `decision: needs_information` and `result: null`. Every result value is
an integer number of cents. Retrieval returns exactly the requested field.
Lists contain unique strings; citations identify applicable R1–R11 clauses.
The answer must cite the clauses needed for its requested calculation or diagnosis.
Explanations are at most 400 characters. A blocking explanation must communicate
the specific issue and a useful next action without inventing a monetary answer.
Its qualitative clarity is judged separately from deterministic correctness.

## R11 — Retrieval requests
A retrieval request names one field: `annual_ceiling_cents` or `allowance_cents`.
Return that value from the uniquely applicable schedule, citing R8 and R11. Salary,
attendance and YTD evidence are not required for this retrieval. If schedule
selection is unresolved, follow R9. Do not infer answers from task identifiers.
