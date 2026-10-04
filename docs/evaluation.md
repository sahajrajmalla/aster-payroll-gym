# Evaluation harness

No genuine model results are included before model calls are explicitly executed.
Tests use `httpx.MockTransport` or named fixture providers and are not benchmark
results. The default install cannot perform local model inference.

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
suggested configurations and cohort definitions; model IDs and quotas must be
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
One process should own a run directory; concurrency is within that process.

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
not stochastic model evidence. No genuine model distribution is claimed yet.

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
