"""Publish the exhaustive requirement/evidence register. No evidence is invented."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPEC = """F01|3-6|One narrow professional, verifiable, tool-shaped domain|rules/aster-payroll-v1.md|tests/test_core.py|implemented
F02|6|Eight phases and exactly one advanced track|docs/architecture.md|docs/submission-checklist.md|external_evidence_pending
F03|4,18|Synthetic fictional data only, no real personal/customer data|data/manifest.json|tests/test_core.py|implemented
F04|4,9|Written versioned authority before implementation; prose wins disagreements|rules/aster-payroll-v1.md|reviews/task-review-packet.json|human_review_pending
F05|18|Solo work; AI permitted with specific honest disclosure|docs/ai-usage.md|docs/submission-checklist.md|implemented
F06|1,17|Five-business-day submission window and speed scoring|docs/submission-checklist.md|record actual receipt/submission dates|human_action_pending
F07|3|Review 10-15 environments, verifiers docs, overview, Harvey architecture|docs/background-reading.md|owner reading checklist|human_review_pending
F08|1,6,15|No UI-polish or frontier-scale scope; free resources; documented mocks|docs/assumptions-and-tradeoffs.md|docs/ai-usage.md|implemented
F09|15,18|Report actual spend, blockers, cuts and two-day next steps|docs/evaluation.md|real experiment artifacts|external_evidence_pending
P01.01|7|Versioned programmatically validated task schema with all minimum fields|schemas/task-v1.json|tests/test_core.py|implemented
P01.02|7|At least ten hand-checked seed tasks across all three tiers|data/seeds.jsonl|reviews/task-review-packet.json|human_review_pending
P01.03|7|At least three genuine Tier-3 traps; confident answers never partial|src/aster_gym/scoring.py|tests/test_core.py|implemented
P01.04|7|Machine-checkable ground truth for every task; no prose-only Tier1/2|src/aster_gym/reference.py|tests/test_core.py|implemented
P01.05|7|One command validates entire corpus, preferably CI|src/aster_gym/cli.py|uv run aster-gym validate|implemented
P01.06|7|Prompt-only answer-leakage check and caught-findings report|docs/edge-cases.md|tests/test_audit.py|implemented
P01.07|7|Detect near-duplicate inflation and state limited variation|src/aster_gym/generator.py|tests/test_core.py|implemented
P01.08|7|Handle units, scale, locale and date ambiguity|rules/aster-payroll-v1.md|tests/test_core.py|implemented
P01.09|7|Explicit schema-drift behavior|src/aster_gym/schemas.py|tests/test_core.py|implemented
P02.01|8|Generator accepts seed/count and is deterministic|src/aster_gym/generator.py|tests/test_core.py|implemented
P02.02|8|Independent Python ground truth, never an LLM|src/aster_gym/reference.py|tests/test_core.py|implemented
P02.03|8|Seed recorded in each result artifact/provenance|src/aster_gym/eval.py|tests/test_harness.py|implemented
P02.04|8|Expose interacting-rule, distractor, lookup and missing-input knobs|src/aster_gym/generator.py|tests/test_core.py|implemented
P02.05|8|100+ generated tasks with observed non-degenerate model distribution|data/evaluation.jsonl|results/qwen-colab-full/metrics.json: all-zero distribution; requirement unmet|external_evidence_pending
P02.06|8|Unit-test reference, caps, precedence, precision and boundaries|src/aster_gym/reference.py|tests/test_core.py|implemented
P02.07|8|README unlimited grading and precise noncomputable limitations|README.md|docs/assumptions-and-tradeoffs.md|implemented
P03.01|8|Installable pyproject and load_environment -> vf.Environment|src/aster_gym/environment.py|docs/colab-final-preflight.json: real optional vf smoke passed on T4|implemented
P03.02|8|Both SingleTurnEnv and ToolEnv/equivalent|src/aster_gym/environment.py|tests/test_harness.py|implemented
P03.03|8|2-4 real tools, sole reference source in tool mode|src/aster_gym/tools.py|tests/test_harness.py|implemented
P03.04|8|max_turns and per-rollout timeout|src/aster_gym/eval.py|tests/test_harness.py|implemented
P03.05|8|Owned strict output contract/parser|src/aster_gym/parser.py|tests/test_core.py|implemented
P03.06|8|CLI model selection and arbitrary remote OpenAI-compatible endpoint|src/aster_gym/cli.py|tests/test_harness.py|implemented
P03.07|9|Malformed/truncated JSON handling|src/aster_gym/parser.py|tests/test_core.py|implemented
P03.08|9|Invalid tool arguments and repeated loops|src/aster_gym/tools.py|tests/test_harness.py|implemented
P03.09|9|Tool exceptions/timeouts recorded, not silently correctness zero|src/aster_gym/eval.py|tests/test_harness.py|implemented
P03.10|9|Provider 429/5xx retry/backoff and attempt logging|src/aster_gym/providers.py|tests/test_harness.py|implemented
P04.01|9|Separate pinned readable authority; version each result|rules/aster-payroll-v1.md|tests/test_core.py|implemented
P04.02|9|Faithful reference with per-rule clause citations|src/aster_gym/reference.py|human fidelity audit against prose|human_review_pending
P04.03|9|LLM-free correctness verifier with justified tolerance|src/aster_gym/scoring.py|tests/test_core.py|implemented
P04.04|10|Every reward component returns deciding clauses|src/aster_gym/scoring.py|tests/test_core.py|implemented
P04.05|10|Document what score levels mean; 1 passes review|docs/reward-spec.md|tests/test_core.py|implemented
P04.06|10|Gate substantive rejection to a low maximum|src/aster_gym/scoring.py|tests/test_core.py|implemented
P04.07|10|Meaningful graded field/set partial credit|src/aster_gym/scoring.py|tests/test_core.py|implemented
P04.08|10|Constrained qualitative LLM judge from rules; version prompt|src/aster_gym/judge.py|tests/test_judge.py|implemented
P04.09|10|Judge 3 repeat ratings or 15 human labels; report agreement|reviews/judge-review-packet.json|real uncached judge calls and human labels|external_evidence_pending
P04.10|10|Rule-grounded calibration; document why each trap unresolved|data/seeds.jsonl|tests/test_core.py|implemented
P04.11|10|Hand-check >=10 generated tasks and report divergence|reviews/task-review-packet.json|human independent calculations|human_review_pending
P04.12|10|Document and defend weights|docs/reward-spec.md|reward tests and review|implemented
P04.13|11-13|Three handwritten adversarial baselines, measured on same tasks|src/aster_gym/cli.py|results/baseline-*|implemented
P04.14|17|Attack reward and preserve fixes/evidence|docs/edge-cases.md|tests/test_core.py and RL transcripts|external_evidence_pending
P05.01|10|One-command evaluation with model/tier/rollouts/seed/cost|src/aster_gym/cli.py|tests/test_harness.py|implemented
P05.02|10|Three real model configs incl frontier and small/cheap|configs/eval.json|real API and Colab runs|external_evidence_pending
P05.03|10|3+ nonzero-temperature rollouts and mean +/- SD|src/aster_gym/eval.py|tests/test_harness.py and real runs|external_evidence_pending
P05.04|10|config.json includes model, seed, versions and prompt hashes|src/aster_gym/eval.py|tests/test_harness.py|implemented
P05.05|10|transcript.jsonl includes full messages/tool calls|src/aster_gym/eval.py|tests/test_harness.py|implemented
P05.06|10|metrics.json includes tokens, latency, costs and tool counts|src/aster_gym/eval.py|tests/test_harness.py|implemented
P05.07|10|scores.json task/reward breakdown|src/aster_gym/eval.py|tests/test_harness.py|implemented
P05.08|10|Hard cost abort, partial results, bounded concurrency, actual spend|src/aster_gym/providers.py|tests/test_harness.py|implemented
P05.09|10|Crash resume without repeating completed paid work|src/aster_gym/eval.py|tests/test_harness.py|implemented
P05.10|10|Provider outage isolation|src/aster_gym/eval.py|tests/test_harness.py|implemented
P05.11|10,17|Honest non-determinism/noise and indistinguishable rankings|docs/evaluation.md|real replicate statistics|external_evidence_pending
P05.12|6|Committed filesystem results and no rerun on deploy|src/aster_gym/reporting.py|tests/test_reporting.py|implemented
P06.01|11|Live dashboard with real saved results|site/index.html|docs/deployment-smoke.json; docs/evaluation.md; real model and baseline artifacts|implemented
P06.02|11|Leaderboard with error bars/SD|src/aster_gym/reporting.py|tests/test_reporting.py|implemented
P06.03|11|Tier and reward breakdowns|src/aster_gym/reporting.py|tests/test_reporting.py|implemented
P06.04|11|Task-by-model failure heatmap|src/aster_gym/reporting.py|tests/test_reporting.py|implemented
P06.05|11|Clickable saved transcript viewer|src/aster_gym/reporting.py|tests/test_reporting.py|implemented
P06.06|11|Adversarial baselines on same leaderboard|results/baseline-*|site/index.html|implemented
P06.07|11|Cost and latency per model|src/aster_gym/reporting.py|tests/test_reporting.py|implemented
P06.08|12|All five actual RL curves visible on dashboard|src/aster_gym/reporting.py|real training_metrics.jsonl|external_evidence_pending
P07.01|11|Online policy-gradient RL from shared Phase04 reward|src/aster_gym/cloud/train.py|actual Colab run|external_evidence_pending
P07.02|11|Implemented objective and algorithm justification|README.md|docs/colab.md|implemented
P07.03|11|Frozen initial reference, explicit beta/estimator, KL every step|src/aster_gym/cloud/train.py|reference_proof.json and actual metrics|external_evidence_pending
P07.04|11|Grouped method >=4 samples and equal-reward group fraction|src/aster_gym/cloud/train.py|tests/test_cloud.py and actual logs|external_evidence_pending
P07.05|12|Held-out before/after with disjoint generator seeds|src/aster_gym/cloud/train.py|actual heldout.json and split checks|external_evidence_pending
P07.06|12|Reward, KL, entropy, length and tier-pass logs/plots|src/aster_gym/cloud/train.py|actual training_metrics.jsonl|external_evidence_pending
P07.07|12|>=2 beta sweep incl deliberately low beta; observed recommendation|configs/train.json|actual validation and sweep artifacts|external_evidence_pending
P07.08|12|Gaming transcript and response or documented thorough search|docs/rl-report.md|actual training completions|external_evidence_pending
P07.09|13|Diagnose entropy/KL/length/format/group/leakage failures|src/aster_gym/cloud/train.py|actual metric/transcript review|external_evidence_pending
P07.10|13|Checkpoint and crash recovery|src/aster_gym/cloud/train.py|cloud resume smoke|external_evidence_pending
P07.11|13,15|Small feasible run, short completions and explicit scoping|configs/train.json|Colab feasibility gate|external_evidence_pending
P07.12|17-18|Preserve actual training transcripts; no fake rising curve|src/aster_gym/cloud/train.py|genuine import bundle|external_evidence_pending
P08.01|13|Live sandbox independent of applicant laptop|render.yaml|docs/render-sandbox-acceptance.json|implemented
P08.02|13|Batch fetch and externally generated answer submission|src/aster_gym/api.py|tests/test_api.py|implemented
P08.03|13-14|No oracle data across wire incl errors/debug/transcripts|src/aster_gym/api.py|tests/test_api.py|implemented
P08.04|13|Exactly shared rubric and pinned authority|src/aster_gym/api.py|tests/test_api.py|implemented
P08.05|13|Per-task score, component breakdown and clauses|src/aster_gym/api.py|tests/test_api.py|implemented
P08.06|13|Run id, private actual seed, versions and tier mix|src/aster_gym/store.py|tests/test_api.py|implemented
P08.07|13|One no-clone command under two minutes|README.md|docs/render-no-clone-proof.json; docs/render-no-clone-tier3-proof.json|implemented
P08.08|13|Task cap, caller limits, no caller API-key storage, cost story|src/aster_gym/store.py|tests/test_api.py|implemented
P08.09|14|Expected-value provenance leakage check and findings|docs/edge-cases.md|tests/test_api.py|implemented
P08.10|14|10000 tasks/tight submit loops rejected|src/aster_gym/api.py|tests/test_api.py|implemented
P08.11|14|Safe concurrent writes|src/aster_gym/store.py|tests/test_api.py|implemented
P08.12|14|Prevent silent cross-version comparisons|src/aster_gym/reporting.py|tests/test_reporting.py|implemented
P08.13|14|Cold starts documented and retried|docs/deployment.md|scripts/quickstart.py|implemented
P08.14|14|Reviewer Tier2/3 first-try external roundtrip|src/aster_gym/api.py|docs/render-tier2-proof.json; docs/render-tier3-proof.json|implemented
P08.15|14|GET /tasks POST /submit GET /runs/{id}|src/aster_gym/api.py|tests/test_api.py|implemented
ADV-B|15|Five handwritten-distribution realistic transfer tasks with ranking findings|data/transfer.jsonl|human approval plus real comparison|external_evidence_pending
D01|16|GitHub repo clean clone plus .env.example|README.md|docs/local-verification.json and public repo|implemented
D02|16|Deployed real-results dashboard URL|site/index.html|docs/deployment-smoke.json; real model and baseline artifacts|implemented
D03|16|Live sandbox URL and two-minute command|render.yaml|docs/render-sandbox-acceptance.json|implemented
D04|16|Loom 8-12min >=3min reward >=1min failure plus live sandbox|docs/loom-script.md|owner recorded video|human_action_pending
D05|16|README architecture stack schema weights run next steps|README.md|documentation review|implemented
D06|16|Eval report real configs, variance, tiers, baselines, judge, fidelity, spend|docs/evaluation.md|actual experiments and human audits|external_evidence_pending
D07|16|RL report five curves, heldout, beta sweep and gaming|docs/rl-report.md|actual Colab artifacts|external_evidence_pending
D08|16|Hand-read 5-10 failed transcripts; categories and counts|docs/failure-analysis.md|human signed review rows|human_review_pending
D09|16|Advanced track measured findings|docs/advanced-track.md|actual transfer results|external_evidence_pending
D10|16|Five actually encountered edges and responses|docs/edge-cases.md|regression tests|implemented
D11|16|At least three explicit assumptions/tradeoffs|docs/assumptions-and-tradeoffs.md|documentation review|implemented
D12|16,18|Specific honest AI usage disclosure|docs/ai-usage.md|owner final confirmation|implemented
D13|18|Four submission links and required email subject|docs/submission-checklist.md|owner email draft; not sent automatically|human_action_pending
U01|user|Python-first pinned deps types logs config modules no secrets|pyproject.toml|CI lint/type/startup|implemented
U02|user|CI schema/determinism/reference/reward/leakage/API/startup|.github/workflows/ci.yml|actual local check artifact|implemented
U03|user|Adversarial malformed/missing/conflict/timeout/cost/provider/reward/split checks|tests/|pytest result artifact|implemented
U04|user|Vertical slice before expansion; actual outputs; requirement audit|docs/submission-checklist.md|local smoke plus pending experiments|external_evidence_pending
U05|user|No local training, inference, model downloads or heavy computation|src/aster_gym/cloud/guard.py|tests/test_cloud.py|implemented
U06|user|Training blockers do not block remaining implementation|docs/rl-report.md|pending/failed statuses and notebook recovery|implemented
U07|user|Flexible Colab configuration and safe compact results handoff|src/aster_gym/bundles.py|tests/test_cloud.py|implemented"""


def main():
    rows = []
    for line in SPEC.splitlines():
        identifier, page, requirement, implementation, test, status = line.split("|")
        rows.append({"id": identifier, "source_page": page, "requirement": requirement,
                     "implementation": implementation, "test_or_manual_check": test,
                     "evidence": test, "status": status})
    status_definition = "implemented means software/doc exists, not completed external/human evidence"
    doc = {"source": "AI Product Operator - Niural AI Labs.pdf, 18 pages",
           "status_definition": status_definition,
           "optional_not_selected": ["A adversarial advanced track", "C scaffold advanced track",
                                     "D publish advanced track", "sandbox driving caller agents",
                                     "external runs on leaderboard"], "requirements": rows}
    (ROOT / "docs" / "traceability.json").write_text(json.dumps(doc, indent=2) + "\n")
    text = ["# Requirement traceability matrix", "", status_definition + ".", "",
            "Repository and dashboard URLs are verified. Genuine remote results are recorded; "
            "the three-model comparison, human labels and training curves remain pending. Public sandbox acceptance passed.", "",
            "| ID / PDF page | Requirement | Implementation | Test / evidence | Status |",
            "| --- | --- | --- | --- | --- |"]
    for row in rows:
        text.append(f"| {row['id']} / {row['source_page']} | {row['requirement']} | "
                    f"{row['implementation']} | {row['test_or_manual_check']} | {row['status']} |")
    (ROOT / "docs" / "traceability.md").write_text("\n".join(text) + "\n")
    print(json.dumps({"requirements": len(rows), "statuses": {
        status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})}}))


if __name__ == "__main__":
    main()
