# Evaluation harness

Real remote and Colab Qwen results are recorded; GRPO and the final three-policy
comparison remain pending. Tests and fixtures are excluded from benchmarks.
No local model inference, training or weight downloads have run.

## Recorded remote comparison — 2026-10-06

Both configurations use the same frozen first 30 evaluation tasks, seed 7001,
three stochastic rollouts, temperature 0.7, 256 completion tokens and low reasoning
effort. The same production judge uses 512 tokens and temperature 0. The owner
confirmed free-tier access and disabled billing; authenticated model metadata and
real inference calls were checked. Confirmed spend is $0 under explicit free
pricing, with a hard $0 cap. Independent human judge/fidelity reviews are pending.

Gemini 3.5 Flash-Lite completed **90/90** samples after bounded provider retries
and resuming only operational failures. Its mean reward is **0.11444 ± 0.03845**
(sample SD across three complete rollout means). Tier means are 0.00667/0.00167/
0.33500 for tiers 1/2/3. Pass rates are 0%/0%/26.67%. Mean end-to-end latency is
3.741s, including saved failed attempts; 208,962 usage tokens and 51 policy retry
attempts were recorded. Wrong completed answers were never rerun or repaired.
[Configuration](../results/gemini-3.5-comparison/config.json),
[metrics](../results/gemini-3.5-comparison/metrics.json),
[transcripts](../results/gemini-3.5-comparison/transcript.jsonl).

Gemini 3.8 Flash remains **partial: 8/90 completed**, 43 observed operational
failures and 39 samples not completed. Repeated 429 responses and initial 503s
prevented progress, so the run was stopped and preserved for resume. There are no
complete replicate means: leaderboard mean±SD is unavailable, and no ranking
against Flash-Lite is claimed. The completed-sample mean 0.146875 is only a biased
partial-coverage diagnostic, not the benchmark aggregate. Confirmed spend is $0.
[Configuration](../results/gemini-3.8-comparison/config.json),
[metrics](../results/gemini-3.8-comparison/metrics.json),
[transcripts](../results/gemini-3.8-comparison/transcript.jsonl).

The twelve-task tool-use run is **partial: 14/36 completed**, with 22
`PROVIDER_RATE_LIMIT` failures. Saved transcripts record 94 tool calls; completed
answers include five turn-limit failures, four invalid-contract failures and five
substantive failures. No complete replicate exists, so no tool-versus-single
ranking is claimed. Rule and task evidence were accessed through the public tools;
wrong completed answers were preserved. Confirmed spend is $0.
[Configuration](../results/gemini-3.5-tool/config.json),
[metrics](../results/gemini-3.5-tool/metrics.json),
[transcripts](../results/gemini-3.5-tool/transcript.jsonl).

Actual failures include scalar retrieval results, truncated JSON, missing clauses
and incorrect arithmetic. Ten exact outputs are prepared for independent reading
in the [failure packet](../reviews/failure-review-packet.json). Strict formatting
and substantive gates explain the low rewards; no evidence of model improvement
or three-model superiority is claimed. The >=100-task generated distribution,
transfer rankings and human judge agreement remain pending. The 2026-10-07
follow-up below records the subsequent non-degenerate distribution and repeat study.

The dashboard separates rankings by taskset, mode, rules/schema/reward versions,
judge identity, prompt hashes, temperature, token limit, seed and rollout count. A no-key cloud
run cannot silently share a ranking with a credentialed run. Cloud records report
API spend; GPU hosting charges require separate runtime billing verification.

The remote seed is an experiment identifier; the Gemini compatibility request
does not send a provider RNG seed. Qwen explicitly seeds its Colab sampling.
Matching seed fields prevents accidental run mixing but does not make API outputs
bit-reproducible or synchronize RNG streams across different models. Full saved
prompts, responses and stochastic replicate statistics are the reproducible evidence.

## Recorded Colab Qwen baseline — 2026-10-07

The initial Qwen2.5-0.5B-Instruct policy completed **90/90 samples** on the same
30-task subset, with three rollouts, temperature .7 and 256 completion tokens.
Reward was **.00139 ± .00241**, the sample SD across rollout means
[0, .00417, 0]. No answer passed .975. The deterministic gates rejected 89 outputs
as `INVALID_CONTRACT`; one received .125 under `CORRECTNESS_CAP`. Tier means
were 0 / 0 / .00417. Mean inference latency was 5.810s, with 183,477 usage tokens
and no provider retries. No output needed an explanation judge in this cohort.

This was genuine initial-policy inference on a Tesla T4, **not training**. API
spend was $0; GPU hosting billing was not independently verified. The pinned
model revision and source revision are preserved in the
[configuration](../results/qwen-colab-comparison/config.json), with
[metrics](../results/qwen-colab-comparison/metrics.json),
[raw transcripts](../results/qwen-colab-comparison/transcript.jsonl) and
[checked import manifest](../results/qwen-colab-comparison/import_manifest.json).
No model weights were imported onto the Mac.

Judge identity is null and sampling seed is 20261002, unlike the Gemini runs.
The dashboard therefore presents a separate experiment group: these scores do
not establish a controlled cross-model ranking.

The full **120-task evaluation completed 360/360 samples**, three rollouts per
task. Mean reward and replicate SD are **0.0 ± 0.0**, with zero passes in each
tier. Gates: 357 `INVALID_CONTRACT`, two `UNJUSTIFIED_ABSTENTION`, one
`CORRECTNESS_CAP` with no earned substantive credit. Mean latency was 5.420s;
729,386 usage tokens, no operational failures and no retries were recorded.
[Configuration](../results/qwen-colab-full/config.json),
[metrics](../results/qwen-colab-full/metrics.json),
[transcripts](../results/qwen-colab-full/transcript.jsonl).

The **12-task tool cohort completed 36/36 samples**, reward **0.0 ± 0.0**.
All 36 failed `TOOLS_NOT_USED`: the model did not emit an accepted tool envelope
or obtain reference evidence. There were zero actual tool calls and no provider
failures. Mean latency was 5.698s with 19,548 usage tokens. This measures Qwen with
the documented `explicit-json-tool-envelope-v1` adapter; it does not show a tool
benefit or an adapter-independent model ranking.
[Configuration](../results/qwen-colab-tool/config.json),
[metrics](../results/qwen-colab-tool/metrics.json),
[transcripts](../results/qwen-colab-tool/transcript.jsonl).

The [final import record](colab-final-import.json) links checked archive hashes,
source revision and retained [cloud execution status](colab-final-run-status.json).
Comparison duplicates were verified identical and retained once. All three Qwen
cohorts cost $0 in model API charges; GPU hosting billing remains unverified.
Their recorded command durations sum to approximately 45m38s, including process
startup and artifact writes, rather than GPU generation time alone.

The Qwen distribution is entirely zero. The 2026-10-07 Gemini follow-up below
now supplies the required non-degenerate distribution on all 120 tasks, without
weakening gates or substituting fixtures. Flat Qwen rewards still warn that GRPO
groups may have zero advantages. No learning is claimed from baseline inference.

## Running

Use frozen task files for reported experiments. An example remote invocation is:

```bash
uv run python -m aster_gym.eval --tasks data/evaluation.jsonl \
  --model "$MODEL_NAME" --output artifacts/my-evaluation \
  --rollouts 3 --mode single --max-cost 0 \
  --input-per-million 0 --output-per-million 0 --verified-free
```

Set `MODEL_API_KEY`; configure the remote judge as in `.env.example`. Supply zero
prices only after verifying the account's free-tier entitlement and disabling
billing. Null/unknown prices are rejected. `configs/eval.json` contains three
configured policies and cohort definitions; model IDs and quotas must be
verified before experiment freeze. The small Qwen configuration runs in Colab.

`run_evaluation(tasks, output_dir, model=..., provider=...)` is also the cloud
entrypoint. An injected provider exposes `async chat(messages, tools=None,
max_tokens=512, temperature=.7)` returning `providers.ChatResult`. It shares the
harness `CostBudget`; a judge must support `bind_budget`. Root CLI configuration
can select model and base URL without source edits.

## Accounting, retries, and resumption

Input reservations use the UTF-8 request size plus per-message overhead as a
conservative bound for ordinary text tokenization. The completion token limit
bounds output. Unsupported multimodal requests are outside this harness. The
provider must report usage; otherwise the full reservation is charged and the
response is an operational failure. If the provider violates a usage bound,
subsequent calls stop and the discrepancy is explicit; no client can guarantee
an external provider follows its advertised metering contract.

Reservations are synchronized across evaluation workers and the judge and saved
before dispatch. Confirmed usage replaces the reservation. Timeouts/network
failures/5xx consume the full reservation conservatively; rejected 4xx consume
zero. Retry only 429, 5xx, and network/timeouts, with capped exponential delay.
Cancellation also settles conservatively. Persisted reservations remaining after
a process interruption are charged when resuming. The hard cap therefore covers
attempts even when the exact bill is not knowable. Actual confirmed spend and
conservative accounted spend are reported separately. Never interpret a zero
price configuration as proof of an externally verified pricing agreement.

Each completed rollout is atomically journaled with its transcript. Restart with
identical configuration to skip every completed response, including wrong
answers. Infrastructure failures may retry. Pending judge scoring reuses the
same final answer. Prior failed transcripts and costs remain in the run. Changed
prompts, tasks, versions, pricing or sampling configuration require a new run.
Retain the original run directory, including ignored `.records` and its pinned
code revision, when resuming. Published/imported summaries are archival evidence;
a missing journal raises `RESUME_JOURNAL_MISSING` before any call rather than
resampling answers. One process should own a run directory; concurrency is within
that process.

The provider's retries fit inside the outer rollout deadline. Six tool turns,
24 tool calls, eight calls per assistant message, bounded arguments, and bounded
arithmetic prevent loops. Tool mode requires at least one successful reference
read; guesses without tools receive `TOOLS_NOT_USED` and zero.

## Statistics and evidence

The leaderboard statistic is mean ± sample SD of the three **complete replicate
means** over identical tasks. Incomplete replicates are omitted from that
statistic and coverage remains visible. An SD with fewer than two complete
replicates is null, never zero. Per-task variability, per-tier sample means/SD,
pass rates and reward components are separate. Operational failures remain
unscored and count against completion coverage.

The run writes `config.json`, `scores.json`, `transcript.jsonl`, `metrics.json`,
`budget.json`, and an internal `.records` resumption journal. Config pins task,
prompt and rule hashes, version IDs, seed provenance, pricing, model/endpoint,
and evidence kind. A seed establishes task generation identity; a remote
provider may remain nondeterministic even for fixed sampling settings. Transcripts
include actual assistant messages, tool calls/results, safe attempt status codes,
and applicable judge calls. API keys and raw provider error bodies are excluded.

`model_run`, `adversarial_baseline`, and `fixture` evidence are distinct. Compare
only identical cohorts, modes and scoring versions. The full 120-task small-model
run must not be ranked against another model's 30-task subset.

The actual deterministic baselines have been run over both 120 tasks and the
frozen first 30 balanced tasks. Read their recomputed results in
the dashboard's saved baseline records. Replicates repeat a handwritten policy and are explicitly
not stochastic model evidence. The >=100-task Qwen evaluation is complete, but its all-zero distribution fails
the required non-degeneracy condition. The 30-task remote comparison does not replace it.

After imports, `uv run aster-gym analyze` computes Track B average-tie-rank
Spearman agreement and task-level ranking reversals from complete matched runs.
`uv run aster-gym judge-study` obtains three uncached ratings for the fifteen
review examples; supply your independent labels before running it. Pending
ratings stay pending and do not count as disagreements or fabricated zeros.

## Verifiers adapter

`aster_gym.environment.load_environment` returns a real `vf.SingleTurnEnv` or a
`vf.ToolEnv` subclass when the optional environment extra is explicitly installed.
Dataset prompts contain public evidence only; the `answer` column holds an
opaque task ID. Tools route via each rollout's `state.info.task_id`, with counters
in rollout state. No environment-global active task exists. The shared rubric
raises on pending judge scoring rather than converting outages to zero rewards.

Adapter behavior was checked against the pinned [ToolEnv implementation](https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/v0.1.14/verifiers/envs/tool_env.py)
and [environment interfaces](https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/v0.1.14/verifiers/envs/environment.py).
The [T4 preflight](colab-final-preflight.json) passed real optional-stack
construction for both environments. Local tests do not install or import that stack.

## Evidence follow-up — 2026-10-07

[Independent rule cross-check](independent-rule-audit.json) uses a second Python
implementation with exact integer/rational arithmetic and constants transcribed
from R8. It agrees on substantive outputs for all 12 seeds, 10 generated review
tasks and five transfer drafts: **0/27 numerical/disposition divergences**. This
is an automated check; it does not replace hand review or approve transfer drafts.
Citation fidelity is outside that numerical cross-check. The scorer's generic
R1/R10 citation obligations are stricter than the explicit R8/R11 retrieval
wording; the [technical review](failure-technical-audit.json) flags that ambiguity.

The [judge study](../results/judge-stability-study.json) records 15 fixed examples,
three uncached production-judge responses each: **45 actual responses**, 44 valid
labels and one malformed fenced response (**2.22%** invalid). Two of the 14 valid
triples vary (**14.29%** disagreement conditional on valid triples). The malformed
response also makes its triple unstable: **3/15 (20%)** triples have differing
outcomes when invalid output counts as an abstention. The malformed
repeat is retained as an abstention, not rerun until valid. All nine ratings of
three instruction-injection examples were 0. Repeated stability is not human
validity; the blind human labels remain empty. Actual usage was **97,049 tokens**,
confirmed API spend **$0**. Three additional 429 attempts are recovered from logs;
an early resume overwrote their ledger before the preservation bug was fixed.
That limitation is explicit in the study, and regression tests cover future resumes.

A fresh [Gemini full run](../results/gemini-3.5-paced-full/config.json) uses seed
20261002, temperature .7, 256 tokens and the production judge. Its complete first
rollout covers **120 distinct tasks**, mean **.128125**, sample SD **.285533**,
80 zero rewards, 40 positive rewards and 11 passes. These are task/sample
statistics, not three-rollout error bars. [Frozen distribution proof](nondegenerate-distribution.json).
The run is now **complete: 360/360**, three rollouts on all 120 tasks. Replicate
mean±SD is **.120486 ± .007316**, with **32 passes** (all Tier 3), 247 zero rewards
and no operational failures. Tier means are .011667/.006667/.343125. Actual
usage is **832,143 tokens**, recorded API spend **$0**, mean end-to-end latency
**5.016 seconds**; dispatch pacing is included. [Metrics](../results/gemini-3.5-paced-full/metrics.json)
and [transcripts](../results/gemini-3.5-paced-full/transcript.jsonl).
The interrupted unpaced trial remains a separate partial run with 17 completed
answers. Changing dispatch pacing started a fresh execution configuration; no
previous outputs were replaced or merged into these three complete replicates.
This completes the non-degenerate 100-task requirement. Dispatches are spaced five
seconds apart to respect free-tier limits. Source patches preserve the exact
experiment hashes from revision af12c3a in `docs/evaluation-source-patches/`.

The fresh frontier attempt has no completed answers: timeouts, 503s and 429s
prevented a controlled comparison. Those are operational failures, not reward-zero
answers. Neither an incomplete frontier run nor different judge configurations
can establish a three-policy ranking. No paid fallback or fabricated scores were used.

## Final audit of available experiments

[Resource/provenance audit](final-experiment-audit.json) recomputes all nine model
runs from raw records: **975 completed answers**, **2,166,928 recorded usage tokens**,
plus **97,049** judge-study tokens. Recorded API spend is **$0**; total resource
spend remains unknown until Colab billing is independently confirmed. Failed
provider calls can have unavailable usage; the audit does not invent their tokens.
It includes timestamps, checksums and the interrupted runs in resource accounting.

[Reward replay](reward-gaming-audit.json) reproduces all **934** ordinary scoring
records from saved answers and real recorded judge labels, and checks **41** tool
termination gates. No missing judge evidence or score/component discrepancies
were found. All **41 high-reward outputs** pass deterministic eligibility; no
programmatic gate bypass was detected. This bounded audit covers initial policies;
trained-policy gaming and independent human explanation validity remain pending.
