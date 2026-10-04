# Your review packet — approximately 2–3 hours

Only you can attest to independent human review. Every row starts unreviewed.

1. Open the written rules. For the 12 seed rows in
   `reviews/task-review-packet.json`, calculate from inputs before looking at
   reference_result. Enter human_result (result or blockers), reviewer, notes,
   and reviewed:true only after comparing. Correct the prose/code if they disagree.
2. Repeat independently for the ten fidelity rows. Report disagreement count / ten
   before corrections and after. A 0/10 result is not yet available merely because
   the automated reference tests pass.
3. Approve/edit the five authored transfer presentations. The current drafts were
   AI-assisted, so do not call them human-written without your actual authorship or
   edits and approval. Freeze them before examining transfer model outcomes.
4. The fifteen handwritten blocking examples and misleading controls in
   `reviews/judge-review-packet.json` are explicitly not model outputs. Label
   qualitative quality 0/1/2 and enter your reviewer name without seeing judge
   predictions, then run `uv run aster-gym judge-study` for three uncached remote
   ratings each. Its summary records exact human agreement, confusion counts and
   repeated disagreement. Replace/add real outputs in a separate versioned packet
   after experiments when auditing the deployed judge's distribution.
5. Read 5–10 actual failed model transcripts. Spend at least one minute reading one
   aloud in the Loom. Record taxonomy, counts, run/task/rollout IDs, and what changed.
6. Review the AI disclosure and record the 8–12 minute Loom using its timed outline.

Automated software reviews are helpful but are not these human checks. Reference
results in the packet are comparison aids, not a substitute for independent work.

After approving all five transfer rows, run `uv run aster-gym freeze-transfer`.
Actual transfer evaluation refuses to start without the matching frozen review
marker. If you edit a presentation, update the corresponding task/context and
review row, validate it, and freeze before inspecting any model outcomes. Do not
mark the provided AI-assisted drafts as independently human-authored.

Run `uv run aster-gym analyze` after completing labels or importing experiments,
then `uv run aster-gym report`. These commands compute evidence summaries locally
without running a model. The transfer analysis requires the same three policies,
the same 30-task generated cohort and the same five transfer tasks, all with three
completed stochastic rollouts and matching rule/reward versions.
