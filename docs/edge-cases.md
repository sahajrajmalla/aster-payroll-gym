# Encountered edge cases and fixes

These are observed development/specification issues, not invented model failures.

1. A missing current table is not genuinely uncertain if the prose already fixes its
   rates. The no-schedule trap uses 2031, beyond the written schedules' validity.
2. Python treats True as an integer. Monetary input validators and output StrictInt
   reject it; parser tests also reject extra fields, NaN and duplicate JSON keys.
3. A raw weighted sum can reward polished wrong answers. Hard disposition zeros and
   the 0.20 correctness cap prevent format/length/abstention hacks dominating scores.
4. Missing judge access could silently become a zero correctness result. Correct
   applicable answers now retain score:null/pending; retries preserve completed work.
5. Globally banning expected numeric substrings is impossible for retrieval tasks
   or caller transcripts. Leakage checks use server-field provenance and allowlisted
   DTOs; only legitimate source documents and caller-originated answers may contain
   those values. Raw seeds and reference objects stay private.
6. Irrelevant salary/bonus values could hide duplicate retrieval requests in input
   hashes. R11 fingerprints now ignore those distractors, and the frozen generator
   excludes fingerprints from every earlier split. Validation reports the four
   repeated schedule-field template groups instead of treating them as forty
   independent rule tests. The stronger check caught a corpus overlap during QA;
   the corpus was regenerated before any model experiment or human review.

Additional tested edges: provider outages, retry/cost exhaustion, tool loops, stale
salary authority, period lengths, schema drift, split collisions, concurrent
submissions, malicious bundle paths and incompatible result versions.
