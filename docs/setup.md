# Understand and run the project

Aster is a test centre for a fictional payroll assistant: it creates questions,
checks submitted answers and saves the marks. Tier 1 finds a rule value; Tier 2
calculates pay; Tier 3 identifies missing or conflicting evidence.

## Only the essential parts

- [Rulebook](../rules/aster-payroll-v1.md): defines correct behaviour.
- [Generator](../src/aster_gym/generator.py) and [reference](../src/aster_gym/reference.py): create synthetic tasks and compute private answers using Python.
- [Scorer](../src/aster_gym/scoring.py): checks JSON, arithmetic, decisions and clauses. A remote judge assesses only eligible blocking explanations, worth 5%.
- [API](../src/aster_gym/api.py) and [storage](../src/aster_gym/store.py): receive external answers and persist token-protected runs in Neon.
- [Evaluation](../src/aster_gym/eval.py): records model performance. [Cloud training](../src/aster_gym/cloud/train.py) trains Qwen in Colab.
- `results/` and [reporting](../src/aster_gym/reporting.py): saved transcripts/scores and the dashboard. `reviews/` holds independent human checks.

The flow is: **public task → your/agent answer → shared scorer → saved record**.
Evaluation files feed the dashboard; sandbox runs stay in Neon and are retrieved through the API.
The sandbox accepts answers; the demonstration client calculates its own answer.
A passing demonstration proves the service works, not model performance.

## 1. Install

Install uv and use Python 3.11. Copy `.env.example` to `.env` on a fresh clone,
then configure private credentials. Preserve an existing `.env`.

```sh
cd /Users/sahajrajmalla/Documents/NeurlAI
uv sync --locked --extra server
uv run --extra server aster-gym validate
uv run --extra server pytest -q
```

The `server` extra adds Postgres support. Keep both `cloud` and `environment`
extras in Colab; never run training or Qwen inference on the local development machine.

## 2. Run a correct answer against the live sandbox

```sh
uv run --extra server python scripts/sandbox_smoke.py https://aster-payroll-gym.onrender.com --receipt tmp/my-tier2-private.json --proof tmp/my-tier2-proof.json
uv run --extra server python scripts/sandbox_smoke.py https://aster-payroll-gym.onrender.com --tier 3 --receipt tmp/my-tier3-private.json --proof tmp/my-tier3-proof.json
```

Look for `submission_verified`, `complete` and a score at least `0.975`.
Tier 3 uses the hosted explanation judge. Exit 2 means scoring remains pending.
For an existing receipt, repeat with `--resume` instead of issuing another task:

```sh
uv run --extra server python scripts/sandbox_smoke.py --resume --receipt tmp/my-tier2-private.json --proof tmp/my-tier2-proof.json
```

Keep `.env` and private receipts out of Git/video. `scripts/quickstart.py`
deliberately submits an invalid answer; use `sandbox_smoke.py` for a passing demo.

## 3. Start your own local API

In one terminal:

```sh
uv run --extra server aster-gym serve
```

Open <http://127.0.0.1:8000/docs>. This uses your configured Neon database.
For a separate practice database, start with:

```sh
DATABASE_URL=sqlite:///tmp/practice.sqlite uv run --extra server aster-gym serve
```

In another terminal, run the same smoke command using `http://127.0.0.1:8000`
and different receipt/proof paths. Stop the server with **Ctrl+C**.

The [live API explorer](https://aster-payroll-gym.onrender.com/docs) also supports
manual testing: fetch `/tasks`, retain `run_id`/`run_token`, submit `/submit` with
`X-Run-Token`, then retrieve `/runs/{run_id}`. See [answer format](task-schema.md).

## 4. View the dashboard locally

```sh
uv run --extra server aster-gym report --output site
uv run --extra server python -m http.server 8002 --bind 127.0.0.1 --directory site
```

Open <http://127.0.0.1:8002>. Ctrl+C stops it. Reporting reads saved results and
makes no model calls. The [published dashboard](https://sahajrajmalla.com.np/aster-payroll-gym/) is ready too.

## 5. Finish experiments in Colab

Complete [human reviews](human-review.md), including transfer approval/freeze.
Then open the [Colab notebook](https://colab.research.google.com/github/sahajrajmalla/aster-payroll-gym/blob/main/notebooks/aster_colab.ipynb),
save a copy, choose a hosted GPU, add keys through Secrets and follow [Colab steps](colab.md).
Enable phases deliberately; inspect the smoke before full training.

Return only the small results ZIP; checkpoints stay on Drive:

```sh
uv run --extra server aster-gym import-results --bundle /path/to/aster-results.zip --output results/my-colab-import
uv run --extra server aster-gym analyze
uv run --extra server aster-gym report --output site
```

The [checklist](submission-checklist.md) separates completed implementation and
sandbox checks from pending experiments, independent reviews and Loom.
