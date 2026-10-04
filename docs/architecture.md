# Architecture

Written ASTER-1.0 authority → business-evidence generator → independent Python
reference → strict Task → single-turn/tool environment → owned JSON parser → shared
ScoreReport. Evaluation, Colab RL and sandbox all call `scoring.score`.

The generator never supplies precomputed intermediates to the reference. The
reference imports no generator or model library. Source documents and oracle
results are separate. Public API DTOs explicitly allowlist source fields; seed,
trap tags, expected results and raw internal inputs stay private.

`schemas` owns contracts; `reference` arithmetic/decision rules; `generator` source
variation/partitions; `parser` syntax; `scoring` gates/weights; `judge` a constrained
remote communication assessor; `tools` scoped lookups and bounded arithmetic;
`environment` optional verifiers adapters; `providers/eval` retries/cost/journals;
`store/api` live answer submissions; `cloud` GPU-only workflows; `bundles` safe
lightweight interchange; `reporting` static output from recorded data.

Eval writes filesystem artifacts. The static dashboard reads them and never calls
providers. SQLite is used for local sandbox development; hosted Postgres holds
issued private tasks, hashes, submissions and reports across server restarts.
Versioned results are separated into comparable cohorts, not silently mixed.

Only Colab may import model libraries or download model weights. Its guard checks
explicit start, Linux, /content, Colab markers and google.colab before torch import,
then requires CUDA. The default lock resolution knows optional ML dependencies but
default installation does not install them. No CPU/MPS fallback exists.

Operational errors are distinct from policy mistakes. Missing judge or provider
availability leaves scoring pending; malformed model output receives zero. Exact
retries resume stored work. Caller-submitted correct values may appear in their
own run transcript; server-owned expected answers never appear.
