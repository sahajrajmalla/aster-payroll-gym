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
Qwen full evaluation, transfer rankings and judge agreement study remain pending.

The dashboard separates rankings by taskset, mode, rules/schema/reward versions,
judge identity, temperature, token limit, seed and rollout count. A no-key cloud
run cannot silently share a ranking with a credentialed run. Cloud records report
API spend; GPU hosting charges require separate runtime billing verification.

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
not establish a controlled cross-model ranking. The 120-task evaluation and
12-task tool cohort finished in Colab; their results remain pending archive import.

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
`docs/saved-results.md`. Replicates repeat a handwritten policy and are explicitly
not stochastic model evidence. The required >=100-task model distribution remains pending; the 30-task remote comparison does not replace it.

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
Live optional-stack execution is a Colab acceptance check; local tests do not
install or import that stack.
