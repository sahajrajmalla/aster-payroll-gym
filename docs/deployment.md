# Deployment and sandbox

The [repository](https://github.com/sahajrajmalla/aster-payroll-gym) and
[dashboard](https://sahajrajmalla.com.np/aster-payroll-gym/) are public.
Render deployment is pending account sign-in; no public API URL is claimed.
A correct independent Tier-2 submission scored **1.0** against real Neon Postgres
and survived a complete API process restart. [Recorded proof](neon-persistence-smoke.json)
uses a local API, so it does not replace the required external acceptance test.

## Deploy on Render with Neon

1. Sign into Render and Neon. Create your own **free Neon project** and copy its
   pooled connection string. The temporary database used for the recorded check
   expires **8 October 2026, 15:26 Nepal time**; it is not the permanent deployment
   database. Use your own project's credentials for the sandbox.
2. Open [Deploy to Render](https://render.com/deploy?repo=https://github.com/sahajrajmalla/aster-payroll-gym).
   Use the repository's `render.yaml`: one free Docker web service. Set secret
   `DATABASE_URL` to your Neon pooled Postgres URL with TLS (`sslmode=require`).
3. Create a [Gemini API key](https://ai.google.dev/gemini-api/docs/api-key) and
   supply it privately as `JUDGE_API_KEY`. Verify account eligibility, available model
   quota and disabled billing before setting `JUDGE_VERIFIED_FREE=true`.
   Keep the configured model/endpoint and judge limits consistent across experiments
   and API. Never put secrets in Git, chat, screenshots or Loom.
4. Deploy, check `https://ACTUAL.onrender.com/healthz`, and save the actual URL.
   The server image includes no training libraries or model weights.
5. Run the correct external check below, restart the Render service, then verify
   persistence. Only record a restart you actually performed. Test `--tier 3` too;
   pending judge scoring is incomplete acceptance, even if arithmetic passes.

[Render free services](https://render.com/docs/free) may sleep after inactivity;
allow a cold start before the two-minute demo. External Neon storage preserves
runs across Render's ephemeral container restarts. Free quotas can change.
[Neon's temporary provisioning](https://neon.com/claimable-neon) requires an owner
claim for lasting use. No paid resources are required by this configuration.

## Correct independent acceptance

This standard-library client derives the answer exclusively from public documents;
it imports neither the project reference calculator nor model libraries.

```sh
python3 scripts/sandbox_smoke.py https://ACTUAL.onrender.com \
  --receipt tmp/render-private.json --proof docs/sandbox-public-proof.json
```

After an actual Render restart:

```sh
python3 scripts/sandbox_smoke.py --verify-restart --restart-confirmed \
  --receipt tmp/render-private.json --proof docs/sandbox-public-proof.json
```

Use a different receipt/proof path with `--tier 3`. The private receipt contains
an access token and is created with mode 600; never publish it. The proof contains
HTTP statuses, score, versions and hashes. `--resume` retries the same immutable
submission; it creates no new task. Pending scoring exits 2, not success.

## Two-minute no-clone quickstart

This example needs Python only. Replace `BASE`; the intentionally invalid answer
checks rejection and run retrieval. An independent solver supplies its own answer
for a passing submission. A machine-readable contract is available at `/docs`.

```python
import json, urllib.request
BASE = "https://ACTUAL.onrender.com"
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

## Contract and limits

- `GET /tasks?tier=all&n=3`: public evidence, versions, run ID/token; `tier` accepts
  all/1/2/3 and `n` accepts 1–10. Seed provenance stays private.
- `POST /submit`: `run_id`, `answers: [{task_id, answer}]`, one answer per task.
  Send `X-Run-Token`. Raw strings preserve malformed JSON for shared scoring.
- `GET /runs/{run_id}`: token-protected status, scores, clauses and caller transcript.
- `GET /healthz`: storage health and scorer versions; storage outages return 503.

Expected answers, corrections, seeds, trap tags and internal inputs are excluded
from public models. A caller's own submitted numbers can appear in its transcript.
The first answer locks an immutable hash: identical retries resume; changed answers
return 409. Active leases/provider failures return 202 with pending scoring, never a
replacement rubric. Tokens cannot be recovered if lost.

Limits: five runs/minute/socket caller, ten tasks/run, 64KiB body, 400-character
explanations; judge quotas default to100/caller and 500 global/day in the deployment.
The safe `--no-proxy-headers` default can group users behind Render's proxy into one
conservative quota bucket. Trusted forwarded-IP configuration needs separate testing.
Errors: 404 unknown run/token, 409 changed submission/configuration, 413 oversized body,
422 invalid request/membership, 429 creation limit. Tables initialize on startup;
future schema changes require explicit migrations.

## Local checks and dashboard

Preserve an existing `.env`. For local SQLite: `uv sync --locked`, then
`uv run aster-gym serve`. For a privately configured Neon connection use
`uv sync --locked --extra server` and `uv run --extra server aster-gym serve`.
The server extra adds the Postgres driver only. Never install cloud extras locally.

Import validated results, run `uv run aster-gym report --output site`, then push.
The GitHub Pages workflow publishes saved synthetic artifacts without model calls.
Missing experiments remain pending. Keep runtime databases, receipts and tokens
outside the dashboard. Update [submission checklist](submission-checklist.md)
only after genuine public acceptance.
