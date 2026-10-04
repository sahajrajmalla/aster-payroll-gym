# Assumptions and deliberate trade-offs

1. Fictional rates replace legal research. This makes an independent authority and
   deterministic oracle possible but does not establish validity for actual payroll.
2. One jurisdiction and five payslip fields keep scope narrow. FX, employer cost,
   classification and discretionary disputes are omitted, not partially mocked.
3. RL trains single-turn structured answers. Tool mode is measured separately;
   multi-turn tool-policy training would consume more compute and is not required.
4. Local ML is prohibited. Colab GPU availability is an external dependency; code
   and deployment preparation continue if training fails. Incomplete experiments
   remain visible and cannot prove learning.
5. The explanation judge uses a free remote API, never local model weights. Its
   influence is bounded and measured, but rate limits and version drift can interrupt
   grading. Cached exact inputs and recorded identifiers reduce repeated work.
6. Filesystem results and static reporting minimize hosting/secret complexity.
   Only the independent sandbox needs a database and live scoring server.
7. Repeated generated templates test rule execution within a narrow distribution.
   Five authored transfer presentations test a limited external-validity claim.

With two more days: repeat held-out experiments for uncertainty, extend human rule
fidelity checks, and resolve observed gaming. Do not spend them adding UI features.
