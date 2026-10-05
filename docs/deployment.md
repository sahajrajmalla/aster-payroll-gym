# Deployment and independent sandbox quickstart

The [dashboard is deployed](https://sahajrajmalla.com.np/aster-payroll-gym/).
The [repository is public](https://github.com/sahajrajmalla/aster-payroll-gym).
Render/Neon account access is still needed for the live sandbox. No hosted sandbox
endpoint is claimed until its public quickstart is tested. Training,
inference, model weights and GPU libraries are absent from the server image.

## Local lightweight API

```sh
uv sync --frozen --no-dev
uv run uvicorn aster_gym.api:app --host 127.0.0.1 --port 8000 --no-proxy-headers
python scripts/quickstart.py http://127.0.0.1:8000
```

SQLite persists runs in `data/sandbox.sqlite`. This exercises scoring with a
deliberately malformed answer and consumes no judge quota. For complete Tier-3
explanation scores configure the shared judge credentials documented in setup.

## Render Free and Neon

1. Push this repository to GitHub; never commit `.env` or database files.
2. Create a free Neon Postgres database. Copy the pooled connection URL with TLS
   (`sslmode=require`) into Render's secret `DATABASE_URL`.
3. In Render choose **New > Blueprint**, connect the repository and use
   `render.yaml`. Supply the shared judge API key as `JUDGE_API_KEY`; the blueprint
   generates `ASTER_RATE_SALT` to salt private caller-address hashes. Keep
   billing disabled and the judge provider restricted to a verified free model.
4. Deploy and open `/healthz`. Substitute the actual base URL in the example below.
5. From a separate browser or terminal test task fetch, submission and run retrieval.
   Repeat after a restart to verify Neon persistence. Record the actual URL/date.

The API normalizes `postgres://` and `postgresql://` URLs to the psycopg driver.
Database tables are created on startup for schema v1. Future migrations must be
explicit; changing columns in Python does not migrate deployed data.

Free Render instances may sleep; allow up to two minutes for a cold start.
Neon stores runs independently of the ephemeral container. Free quotas and model
availability can change: verify them in the account before claiming availability.
One server worker is configured; SQL transactions also protect multiple workers.

## Two-minute no-clone quickstart

This Python standard-library example works without cloning, a model, or an API key.
Change only `BASE`. A malformed answer intentionally returns zero reward.

```python
import json, urllib.request
BASE = "https://YOUR-SANDBOX.onrender.com"
def call(path, body=None, token=""):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data, {
        "Content-Type": "application/json", "X-Run-Token": token})
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.loads(res.read())
r = call("/tasks?tier=2&n=1")
s = call("/submit", {"run_id": r["run_id"], "answers": [
    {"task_id": t["id"], "answer": "{}"} for t in r["tasks"]]}, r["run_token"])
print(json.dumps(s, indent=2))
print(call("/runs/" + r["run_id"], token=r["run_token"]))
```

An agent replaces `"{}"` with its own six-field JSON answer after reading the
public prompt/context. API schema and interactive examples are at `/docs`.

## API contract and limits

- `GET /tasks?tier=all&n=3`: one private-seed run, bearer `run_token`, public tasks,
  versions and seed fingerprint. `tier` accepts `all`, `1`, `2`, `3`; `n` is 1–10.
- `POST /submit`: `run_id` and `answers: [{task_id, answer}]`; exactly one answer
  for every task. Send the token in `X-Run-Token`. A raw string preserves malformed
  JSON for scoring; an object is also accepted.
- `GET /runs/{run_id}`: token-protected status, tier mix, component scores, and the
  full caller transcript (public task context, submitted answers, safe reports).
- `GET /healthz`: storage and scorer versions.

Tokens protect runs; they are never logged or echoed in later responses. Lost
tokens cannot be recovered. Expected answers, numeric corrections, internal task
inputs, generation seeds and trap tags are never public response fields. A number
submitted by the caller can appear in its own transcript even if it coincides with
ground truth; source provenance distinguishes that echo from an oracle disclosure.

Five run creations per minute per direct client address; at most ten tasks per run;
64 KiB submission body; explanation contract limited to 400 characters; judge
quota defaults to 100 calls/caller and 500 global calls in a rolling day. Configure
`ASTER_CALLER_JUDGE_LIMIT` and `ASTER_GLOBAL_JUDGE_LIMIT` to fit free quotas. Requests
behind the Render reverse proxy share the conservative socket-address bucket.
Do not trust arbitrary caller-supplied `X-Forwarded-For` headers. With the safe
default `--no-proxy-headers`, several public users may share a proxy-address quota.
True per-public-IP limits require the hosting provider's trusted proxy IP ranges
to be configured and tested explicitly; the current default is conservative.

The first submission locks an immutable canonical answer hash. Same-answer retries
are idempotent; changed answers return 409. An active scoring lease returns 202.
Pending tasks resume on an identical resubmission, preserving completed scores.
Judging is bounded to 30 seconds per task. Provider failure/quota exhaustion gives
202 and a pending score; weights remain unchanged and pending is never zero.

HTTP errors: 404 unknown run/token; 409 changed submission; 413 body too large;
422 invalid request or task membership; 429 run creation rate limit. Judge
quota errors are represented as pending scoring rather than failed arithmetic.

## Dashboard on GitHub Pages

Run `uv run python -c "from aster_gym.reporting import build_dashboard;
build_dashboard('results', 'site')"`. Open `site/index.html` to inspect saved
results; no model executes. Enable GitHub Pages with **GitHub Actions** as source,
then run the **Evidence dashboard** workflow. Only `model_run` and
`adversarial_baseline` evidence kinds enter rankings. Fixture data is labelled and
excluded. Missing evaluation, training and transfer evidence display pending.

The generated site intentionally includes reviewer transcripts; publish only
synthetic runs. Secrets must never enter configs or transcripts. API responses
retain safe component diagnostics; the dashboard can contain internal saved
scoring artifacts that are separate from the live sandbox. Keep runtime databases
and API tokens outside Pages.

## Recovery checklist

- Health fails: verify TLS/database secret, Neon availability, and logs.
- Pending judge: verify key/model quota, then resend the exact first submission.
- Deploy failed: run frozen lightweight install/startup checks; do not install cloud extras.
- Missing dashboard data: import the validated Colab results bundle, rebuild, publish.
- Scoring-version mismatch: keep results visibly separate; rerun with matching rules.

Record the actual deployment smoke test in `docs/submission-checklist.md`.
