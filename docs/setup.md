# Setup and clean startup

Use Python 3.11 and uv. `uv sync --locked` installs only the small default runtime
and test tools. `uv run aster-gym validate` checks frozen tasksets and partitions.
`uv run pytest -q` runs bounded tests. `uv run ruff check .` and `uv run mypy
src/aster_gym` check code. `uv run aster-gym serve` starts the API; GET /healthz
checks storage. Local OpenAPI documentation is /docs.

Create `.env` from `.env.example` only if it does not already exist. Preserve
existing database credentials. Do not put provider keys in source, notebook cells,
git, chat, or result files. For hosted setup, configure provider/database values as
service secrets. Caller API keys are neither requested nor stored.

No model access is needed for validation, tests, baseline scoring, report rendering
or a trivial API round trip. Correct Tier-3 submissions need the configured remote
judge and may be pending without it. This is intentional and uses the same rubric.

Install optional verifiers adapter with `uv sync --locked --extra environment`
only when needed. Install `--extra cloud` solely inside Colab, following its guard.
Never install the cloud extra or run model workloads on the user's computer.

For clean-clone verification, create a separate temporary checkout/copy, use a new
virtual environment, run the default sync/test/validation/startup commands and
record the result. Do not carry over .env, local database or caches as requirements.
