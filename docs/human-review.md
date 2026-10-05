# Independent review packet

Allow approximately 2–3 hours. Rows start unreviewed; automated checks do not
replace these independent reviews.

1. Read the rules. In `reviews/task-review-packet.json`, independently solve the
   twelve seed rows before comparing with `reference_result`. Enter `human_result`,
   reviewer, notes and `reviewed: true` only after checking.
2. Repeat for ten fidelity rows. Report disagreements before/after corrections;
   a passing test suite is not a completed 0/10 human audit.
3. Edit/approve five transfer drafts. Keep prompt/input review rows synchronized.
   Record only authorship and review you actually performed. Freeze before seeing
   transfer model outcomes:

   ```bash
   uv run aster-gym validate
   uv run aster-gym freeze-transfer
   ```

4. In `reviews/judge-review-packet.json`, independently label fifteen explanation
   examples 0/1/2 and enter your name before viewing predictions. They are a rubric
   review set, not measured model outputs. With verified remote judge access, run
   `uv run aster-gym judge-study` for three uncached ratings each. Report exact human
   agreement, confusion counts and repeat instability.
5. Read 5–10 actual failed model transcripts. Record run/task/rollout IDs,
   categories and remedies in `reviews/failure-review-packet.json`. Read one genuine
   failure aloud for at least one minute in the Loom.

After completed labels and experiments:

```bash
uv run aster-gym analyze
uv run aster-gym report --output site
```

These commands calculate summaries without invoking a model. Transfer analysis
needs three matched policies, the same 30-task cohort and five frozen transfer tasks,
three completed stochastic rollouts, and matching scoring versions. Do not fill
review fields merely to bypass a gate.
