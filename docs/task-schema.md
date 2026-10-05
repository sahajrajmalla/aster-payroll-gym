# Task schema and data separation

`schemas/task-v1.json` and `schemas/answer-v1.json` are generated JSON Schema
artifacts. Internal tasks have id, slice, difficulty, prompt, context_files,
ground_truth, verifier, tags and schema_version plus provenance/version fields.
Difficulty metadata never determines the correct answer and is never sent to
models. A changed unsupported version fails validation rather than silently adapting.

Answers require decision, result, issue_codes, missing_fields, citations and
explanation. No extras, duplicate list values, boolean-as-money, floats or numeric
strings. String parsing rejects duplicate object keys, non-finite tokens, prose,
multiple JSON objects, markdown wrappers and oversized payloads. A payslip has five
cent fields. Retrieval has exactly the requested field. Unresolved work has null
result and complete issue/affected-field sets. Output cents compare exactly (R1).

Frozen seed/validation/training/evaluation/transfer manifests record input hashes.
Generator parameters expose proration, ceiling binding, irrelevant documents,
interacting rules, lookup fragmentation and missing inputs. Same seed/config yields
identical objects. Inputs must agree with their public evidence and reference.

Tier 1 inherently includes repeated answer values (two fields across two schedules).
The corpus reports this small answer vocabulary honestly; varied source dates and
lookups do not create new arithmetic capability. Exact business-input duplicates
are rejected. Transfer layouts are separately authored but not yet human-approved.

Retrieval fingerprints exclude irrelevant payroll money under R11. Changing a
bonus or salary distractor cannot hide an identical date/field/schedule request.
The corpus builder excludes prior partitions' fingerprints before sampling each
later split. Validation additionally reports schedule-year/field template groups;
four repeated Tier-1 templates are explicitly not forty distinct rule tests.
