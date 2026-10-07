"""Build an offline dashboard from genuine saved artifacts, without model calls."""

from __future__ import annotations

import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeGuard

from aster_gym.versions import stable_hash


def comparison_identity(config: dict[str, Any]) -> str:
    """Separate scoring and sampling contracts before ranking saved experiments."""
    return stable_hash({key: config.get(key) for key in (
        "taskset_hash", "mode", "ruleset_version", "rules_hash", "reward_version",
        "schema_version", "judge", "temperature", "max_tokens", "seed", "rollouts", "prompt_hashes",
    )})


def _number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summarize_records(records: list[dict[str, Any]], *, expected_tasks: int | None = None,
                      rollouts: int | None = None, task_ids: list[str] | None = None) -> dict[str, Any]:
    complete = [r for r in records if r.get("status") == "complete" and _number(r.get("score"))]
    by_rollout: dict[str, list[float]] = defaultdict(list)
    by_tier: dict[str, list[float]] = defaultdict(list)
    by_component: dict[str, list[float]] = defaultdict(list)
    all_by_rollout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        all_by_rollout[str(record.get("rollout", 0))].append(record)
    for record in complete:
        by_rollout[str(record.get("rollout", 0))].append(record["score"])
        by_tier[str(record.get("tier", "unknown"))].append(record["score"])
        for name, component in record.get("components", {}).items():
            value = component.get("score") if isinstance(component, dict) else component
            if _number(value):
                by_component[name].append(float(value))
    expected_ids: set[Any] = set(task_ids) if task_ids is not None else {record.get("task_id") for record in records}
    task_count = expected_tasks if expected_tasks is not None else len(expected_ids)
    rollout_count = rollouts if rollouts is not None else len(all_by_rollout)
    total = task_count * rollout_count
    replicate_means = [
        statistics.mean(values) for rollout, values in by_rollout.items()
        if len(values) == task_count == len(all_by_rollout[rollout])
        and {record.get("task_id") for record in all_by_rollout[rollout]} == expected_ids
        and (rollouts is None or 0 <= int(rollout) < rollouts)
    ]
    def avg(values: list[float]) -> float | None:
        return statistics.mean(values) if values else None
    latencies = [r["latency_s"] for r in records if _number(r.get("latency_s"))]
    return {
        "mean": avg(replicate_means),
        "sample_mean": avg([r["score"] for r in complete]),
        "sd": statistics.stdev(replicate_means) if len(replicate_means) >= 2 else None,
        "replicate_means": replicate_means,
        "complete": len(complete), "total": total, "observed": len(records),
        "coverage": len(complete) / total if total else 0,
        "rankable": total > 0 and rollout_count >= 3 and len(complete) == total and len(replicate_means) == rollout_count,
        "tiers": {key: avg(values) for key, values in sorted(by_tier.items())},
        "components": {key: avg(values) for key, values in sorted(by_component.items())},
        "cost_usd": sum(r["cost_usd"] for r in records if _number(r.get("cost_usd"))),
        "confirmed_cost_usd": sum(r.get("confirmed_cost_usd", 0) for r in records if _number(r.get("confirmed_cost_usd", 0))),
        "latency_mean_s": avg(latencies),
        "tokens": sum(r["tokens"] for r in records if _number(r.get("tokens"))),
        "operational_failures": len(records) - len(complete),
    }


def _json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line, parse_constant=_invalid_number) for line in path.read_text().splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL records must be objects")
    return rows


def _invalid_number(value: str) -> None:
    raise ValueError("Nonfinite JSON numbers are invalid evidence")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(), parse_constant=_invalid_number)


def _validate_records(records: Any) -> list[dict[str, Any]]:
    """Reject corrupted evidence before it can influence a displayed aggregate."""
    if not isinstance(records, list):
        raise ValueError("records must be an array")
    seen = set()
    statuses = {"complete", "pending", "provider_error", "cost_exhausted", "timeout", "internal_error",
                "error", "failed", "budget_exhausted"}
    for record in records:
        if not isinstance(record, dict) or record.get("status") not in statuses:
            raise ValueError("invalid record status")
        value = record.get("score")
        if value is not None and (not _number(value) or not 0 <= value <= 1):
            raise ValueError("invalid score")
        if (record["status"] == "complete") != (value is not None):
            raise ValueError("score/status mismatch")
        for field in ("cost_usd", "confirmed_cost_usd", "latency_s", "tokens"):
            value = record.get(field, 0)
            if not _number(value) or value < 0:
                raise ValueError("invalid operational metric")
        components = record.get("components", {})
        if not isinstance(components, dict):
            raise ValueError("components must be an object")
        for component in components.values():
            component_score = component.get("score") if isinstance(component, dict) else component
            if not _number(component_score) or not 0 <= component_score <= 1:
                raise ValueError("invalid component score")
        if "rollout" in record or "task_id" in record:
            if type(record.get("rollout")) is not int or not isinstance(record.get("task_id"), str):
                raise ValueError("invalid record identity")
            identity = (record["task_id"], record["rollout"])
            if identity in seen or record["rollout"] < 0:
                raise ValueError("duplicate or negative record identity")
            seen.add(identity)
    return records


def collect_results(results_dir: Path) -> dict[str, Any]:
    runs = []
    warnings = []
    if results_dir.exists():
        for config_path in sorted(results_dir.rglob("config.json")):
            directory = config_path.parent
            try:
                config = _load_json(config_path)
                if not isinstance(config, dict):
                    raise ValueError("config must be an object")
                scores_path = directory / "scores.json"
                scores = _load_json(scores_path) if scores_path.exists() else {"records": []}
                records = scores.get("records", []) if isinstance(scores, dict) else scores
                records = _validate_records(records)
                kind = config.get("evidence_kind", "unknown")
                is_evaluation = kind in {"model_run", "adversarial_baseline"} and config.get("run_type") != "training"
                task_count, rollout_count, ids = config.get("task_count"), config.get("rollouts"), config.get("task_ids")
                frozen = (type(task_count) is int and task_count > 0
                          and type(rollout_count) is int and rollout_count > 0
                          and isinstance(ids, list) and len(ids) == task_count
                          and all(isinstance(identity, str) for identity in ids) and len(set(ids)) == len(ids))
                if is_evaluation and not frozen:
                    warnings.append(f"Unranked {directory.name}: frozen task/rollout manifest missing")
                if is_evaluation and frozen:
                    frozen_ids = ids if isinstance(ids, list) else []
                    if any(r.get("task_id") not in frozen_ids or type(r.get("rollout")) is not int
                           or not 0 <= r["rollout"] < rollout_count for r in records):
                        raise ValueError("record outside frozen cohort")
                summary = summarize_records(records, expected_tasks=task_count if frozen else None,
                                             rollouts=rollout_count if frozen else None, task_ids=ids if frozen else None)
                summary["rankable"] = summary["rankable"] and frozen and is_evaluation
                budget_path = directory / "budget.json"
                if budget_path.exists():
                    budget = _load_json(budget_path)
                    if (not isinstance(budget, dict) or any(not _number(budget.get(name)) or budget[name] < 0
                            for name in ("accounted_usd", "confirmed_usd", "reserved_usd"))):
                        raise ValueError("invalid cost ledger")
                    # A process may stop after a billed call and before its transcript
                    # is committed. The durable budget remains the accounting source.
                    summary["cost_usd"] = budget["accounted_usd"] + budget["reserved_usd"]
                    summary["confirmed_cost_usd"] = budget["confirmed_usd"]
                elif kind != "adversarial_baseline" and not any("confirmed_cost_usd" in r for r in records):
                    summary["confirmed_cost_usd"] = None
                # Fixtures are excluded from rankings; still available for explicit inspection.
                runs.append({
                    "config": config, "records": records, "summary": summary,
                    "comparison_key": comparison_identity(config),
                    "transcripts": _json_lines(directory / "transcript.jsonl"),
                    "training": _json_lines(directory / "training_metrics.jsonl"),
                    "heldout": _load_json(directory / "heldout.json")
                    if (directory / "heldout.json").exists() else None,
                    "eligible": is_evaluation,
                    "path": str(directory.relative_to(results_dir)),
                })
            except (ValueError, KeyError, TypeError, OSError) as exc:
                warnings.append(f"Skipped {directory.name}: {type(exc).__name__}")
    analysis = None
    if (results_dir / "analysis.json").exists():
        try:
            analysis = _load_json(results_dir / "analysis.json")
            if not isinstance(analysis, dict):
                raise ValueError("analysis must be an object")
        except (ValueError, OSError):
            warnings.append("Analysis artifact is invalid; derived findings are pending")
            analysis = None
    return {"runs": runs, "warnings": warnings, "analysis": analysis}


def build_dashboard(results_dir: str | Path, output_dir: str | Path) -> Path:
    data = collect_results(Path(results_dir))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    page = _PAGE.replace("__RESULT_DATA__", payload)
    path = output / "index.html"
    path.write_text(page)
    (output / "dashboard-data.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    return path


def write_summary(results_dir: str | Path, output_path: str | Path) -> Path:
    """Plain review report; all aggregates are recomputed from score records."""
    data = collect_results(Path(results_dir))
    lines = ["# Saved-results summary", "", "No model inference is run by this report.", ""]
    if not any(r["config"].get("evidence_kind") == "model_run" for r in data["runs"]):
        lines.extend(["Genuine model evaluation and cloud training remain pending. The results below",
                      "are actual deterministic adversarial policy runs, with repeated rather than stochastic outputs.", ""])
    for run in data["runs"]:
        config, summary = run["config"], run["summary"]
        lines.extend([
            f"## {html.escape(str(config.get('model', run['path'])))}",
            "", f"Evidence kind: {config.get('evidence_kind', 'unknown')}.",
            f"Completed records: {summary['complete']}/{summary['total']}.",
            f"Mean reward: {summary['mean']}; replicate sample SD: {summary['sd']}.",
            f"Tier means: {json.dumps(summary['tiers'])}.",
            f"Cost: ${summary['cost_usd']:.6f}; mean latency: {summary['latency_mean_s']} seconds.", "",
        ])
    if not data["runs"]:
        lines.append("Model evaluations and cloud training are pending. No findings are fabricated.")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    return destination


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aster Payroll Gym — Evidence dashboard</title>
<style>
:root{color-scheme:light;--ink:#152c39;--muted:#576878;--line:#d9e0e4;--accent:#176b60;--bg:#f3f6f7}*{box-sizing:border-box}body{margin:0;font:15px/1.55 system-ui,sans-serif;color:var(--ink);background:var(--bg)}main{max-width:1200px;margin:auto;padding:34px 24px}h1{font-size:34px;letter-spacing:-1px;margin:0}h2{font-size:21px;margin:0 0 12px}p{margin:6px 0 18px;color:var(--muted)}.eyebrow{font:12px monospace;color:var(--accent);letter-spacing:2px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px}.card{background:white;border:1px solid var(--line);border-radius:12px;padding:22px;margin:18px 0}.badge{font:12px monospace;padding:4px 7px;background:#e3f2ed;border-radius:4px;display:inline-block}.pending{padding:18px;border-left:3px solid #a67a25;background:#fff8e8;color:#705520}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:10px;white-space:nowrap}th{color:var(--muted);font-size:12px}button,select{border:1px solid var(--line);background:white;border-radius:5px;padding:7px;color:var(--ink);cursor:pointer}button:hover{border-color:var(--accent)}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 ui-monospace,monospace;background:#f3f6f7;padding:15px;max-height:600px;overflow:auto}svg{width:100%;height:auto;display:block}.bar{height:9px;background:#e8ecee;border-radius:5px;width:110px;display:inline-block;vertical-align:middle;margin-right:8px}.bar span{display:block;height:9px;background:var(--accent);border-radius:5px}.heat{display:grid;grid-template-columns:repeat(auto-fill,minmax(64px,1fr));gap:5px;margin:12px 0}.cell{font:11px monospace;padding:8px;border:0;min-height:38px}footer{font-size:12px;color:var(--muted);margin:22px 0}.legend{font-size:12px;color:var(--muted)}@media(max-width:500px){main{padding:22px 12px}.grid{grid-template-columns:1fr}h1{font-size:28px}}
</style></head><body><main><div class="eyebrow">ASTER / ASTER-1.0</div><h1>Payroll Gym Evidence</h1>
<p>Auditable rewards, tool traces, and held-out learning. Each view reads saved outputs; this page never calls a model.</p>
<div id="status"></div><section class="card"><h2>Leaderboard and uncertainty</h2><p>Compare within identical task, judge, scoring and sampling contracts. Error bars use sample SD of rollout-replicate means. Fixtures are excluded.</p><div id="leaderboard" class="scroll"></div></section>
<div class="grid"><section class="card"><h2>Tier and reward breakdowns</h2><select id="run-select" aria-label="Select results run"></select><div id="breakdowns"></div></section><section class="card"><h2>Cost, latency and coverage</h2><div id="cost"></div></section></div>
<section class="card"><h2>Task-by-model failure heatmap</h2><p>Each model shows its task mean across rollouts. Click a cell to inspect a transcript. Amber means pending or operational failure.</p><div id="heatmap"></div></section>
<section class="card"><h2>Transcript viewer</h2><div id="transcript-meta"></div><pre id="transcript">Select a heatmap cell to see the saved messages, tool calls, attempts and clause-linked score.</pre></section>
<section class="card"><h2>RL: two KL sweeps</h2><p>Curves show actual logged steps only. Beta selection uses validation; final evaluation stays seed-separated.</p><div id="rl" class="grid"></div></section>
<section class="card"><h2>Held-out and transfer evidence</h2><div id="heldout"></div></section>
<footer>Synthetic fictional payroll. Ground truth is computed by independent Python rules. Absence of evidence is displayed explicitly. Deployable sandbox links and human review status belong in the repository README.</footer>
</main><script type="application/json" id="results">__RESULT_DATA__</script><script>
'use strict';const data=JSON.parse(document.getElementById('results').textContent),runs=data.runs;
const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e},fmt=x=>typeof x==='number'?x.toFixed(3):'pending';
function pending(target,text){target.append(el('div',text,'pending'))}function label(r){return r.config.model||r.config.run_id||r.path}
function table(headers,rows){const t=el('table'),h=el('tr');headers.forEach(x=>h.append(el('th',x)));t.append(h);rows.forEach(row=>{const tr=el('tr');row.forEach(v=>{const td=el('td');td.append(v instanceof Node?v:document.createTextNode(String(v)));tr.append(td)});t.append(tr)});return t}
function scorebar(mean,sd){const e=el('span');if(typeof mean==='number'){const ns='http://www.w3.org/2000/svg',g=document.createElementNS(ns,'svg');g.setAttribute('viewBox','0 0 130 22');g.style.cssText='width:130px;height:22px;display:inline-block;vertical-align:middle;margin-right:8px';const x=v=>8+Math.max(0,Math.min(1,v))*114;const base=document.createElementNS(ns,'path');base.setAttribute('d','M8 11H122');base.setAttribute('stroke','#d9e0e4');g.append(base);if(typeof sd==='number'){const whisker=document.createElementNS(ns,'path'),a=x(mean-sd),b=x(mean+sd);whisker.setAttribute('d',`M${a} 6V16M${a} 11H${b}M${b} 6V16`);whisker.setAttribute('stroke','#176b60');whisker.setAttribute('stroke-width','2');g.append(whisker)}const point=document.createElementNS(ns,'circle');point.setAttribute('cx',x(mean));point.setAttribute('cy',11);point.setAttribute('r',4);point.setAttribute('fill','#176b60');g.append(point);e.append(g)}e.append(document.createTextNode(fmt(mean)+(sd===null?' (SD pending)':' ± '+fmt(sd))));return e}
const genuine=runs.filter(r=>r.eligible),stat=document.getElementById('status');stat.append(el('span',genuine.length+' saved evaluation/baseline runs','badge'));if(!genuine.some(r=>r.config.evidence_kind==='model_run'))pending(stat,'Model evaluation pending. Programmatic baselines do not establish model performance.');if(runs.some(r=>!r.eligible))stat.append(el('p','Fixtures and unknown evidence kinds are inspectable but excluded from rankings.'));data.warnings.forEach(w=>stat.append(el('p',w)));
const leader=document.getElementById('leaderboard');if(!genuine.length)pending(leader,'Model evaluation pending. No leaderboard scores have been invented.');else{const cohorts=new Map();genuine.forEach(r=>{const key=r.comparison_key;if(!cohorts.has(key))cohorts.set(key,[]);cohorts.get(key).push(r)});cohorts.forEach((rs,key)=>{leader.append(el('p','Compatible cohort: '+key+' · judge: '+(rs[0].config.judge?.model||'none')+' · temperature: '+(rs[0].config.temperature??'n/a')+' · token limit: '+(rs[0].config.max_tokens??'n/a')+' · seed: '+(rs[0].config.seed??'unrecorded')));rs.sort((a,b)=>Number(b.summary.rankable)-Number(a.summary.rankable)||(b.summary.mean??-1)-(a.summary.mean??-1));leader.append(table(['Rank','Model / baseline','Reward ± SD','Coverage','Run status'],rs.map((r,i)=>[r.summary.rankable?i+1:'unranked',label(r)+(r.config.evidence_kind==='adversarial_baseline'?' [baseline]':''),scorebar(r.summary.mean,r.summary.sd),r.summary.complete+'/'+r.summary.total,r.config.status||'unknown'])))})}
const sel=document.getElementById('run-select');runs.forEach((r,i)=>{const opt=el('option',label(r)+' ['+r.config.evidence_kind+']');opt.value=i;sel.append(opt)});
function breakdown(){const r=runs[Number(sel.value)],box=document.getElementById('breakdowns'),cost=document.getElementById('cost');box.replaceChildren();cost.replaceChildren();if(!r){pending(box,'Awaiting saved results.');pending(cost,'Awaiting saved results.');return}box.append(el('p','Evidence: '+r.config.evidence_kind));box.append(table(['Tier','Mean reward'],Object.entries(r.summary.tiers).map(([k,v])=>[k,scorebar(v,null)])));box.append(table(['Component','Mean component score'],Object.entries(r.summary.components).map(([k,v])=>[k,fmt(v)])));cost.append(table(['Measure','Observed value'],[['Confirmed provider cost, USD',typeof r.summary.confirmed_cost_usd==='number'?r.summary.confirmed_cost_usd.toFixed(6):'unavailable'],['Accounted cost including uncertain calls, USD',r.summary.cost_usd.toFixed(6)],['Mean latency, seconds',fmt(r.summary.latency_mean_s)],['Tokens',r.summary.tokens],['Completed / total',r.summary.complete+' / '+r.summary.total],['Operational or pending records',r.summary.operational_failures]]))}sel.onchange=breakdown;breakdown();
function show(r,id){const records=r.records.filter(x=>x.task_id===id),traces=r.transcripts.filter(x=>x.task_id===id);document.getElementById('transcript-meta').textContent=label(r)+' · '+id+' · '+r.config.evidence_kind;document.getElementById('transcript').textContent=JSON.stringify({scores:records,transcripts:traces},null,2);document.getElementById('transcript').scrollIntoView({behavior:'smooth',block:'nearest'})}
const heat=document.getElementById('heatmap');if(!genuine.length)pending(heat,'Failure heatmap pending.');genuine.forEach(r=>{heat.append(el('h3',label(r)));const cells=el('div',undefined,'heat');[...new Set(r.records.map(x=>x.task_id))].forEach(id=>{const rs=r.records.filter(x=>x.task_id===id),sc=rs.filter(x=>x.status==='complete'&&typeof x.score==='number').map(x=>x.score),v=sc.length?sc.reduce((a,b)=>a+b)/sc.length:null,b=el('button',id.slice(-8)+' '+fmt(v),'cell');b.style.background=v===null?'#fff2cc':`hsl(${v*145},40%,${91-v*15}%)`;b.title=id;b.onclick=()=>show(r,id);cells.append(b)});heat.append(cells)});
function chart(target,title,series){target.append(el('h3',title));const usable=series.filter(s=>s.points.length);if(!usable.length){pending(target,'No logged observations.');return}const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 480 220');svg.setAttribute('role','img');svg.setAttribute('aria-label',title);const all=usable.flatMap(s=>s.points),mx=Math.max(...all.map(p=>p[0]),1),lo=Math.min(0,...all.map(p=>p[1])),hi=Math.max(...all.map(p=>p[1]),lo+0.01);for(const [y,value] of [[20,hi],[185,lo]]){const txt=document.createElementNS(ns,'text');txt.setAttribute('x',0);txt.setAttribute('y',y+4);txt.setAttribute('font-size','11');txt.textContent=value.toFixed(3);svg.append(txt)}const axis=document.createElementNS(ns,'path');axis.setAttribute('d','M48 20V185H462');axis.setAttribute('stroke','#cad3d9');axis.setAttribute('fill','none');svg.append(axis);usable.forEach((s,i)=>{const line=document.createElementNS(ns,'polyline');line.setAttribute('points',s.points.map(p=>(48+p[0]/mx*410)+','+(185-(p[1]-lo)/(hi-lo)*165)).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke',['#176b60','#ce7134','#645fc0'][i%3]);line.setAttribute('stroke-width','2');svg.append(line)});target.append(svg,el('div',usable.map(s=>s.name).join(' · ')+' | x: optimizer step','legend'))}
const rl=document.getElementById('rl'),trained=runs.filter(r=>r.training.length&&r.config.evidence_kind==='model_run'&&r.config.run_type==='training');if(!trained.length)pending(rl,'Colab execution pending. All five learning curves will populate from genuine training_metrics.jsonl records.');
const curves=[['Mean reward',['mean_reward','reward','reward_mean']],['KL divergence',['kl','kl_divergence']],['Policy entropy',['entropy','policy_entropy']],['Completion length',['completion_length','mean_completion_length']],['Pass rate by tier',[]]];
if(trained.length)curves.forEach(([title,keys])=>{const card=el('div');const series=[];trained.forEach(r=>{if(keys.length){series.push({name:label(r)+' β='+r.config.beta,points:r.training.map(x=>[x.step,keys.map(k=>x[k]).find(v=>typeof v==='number')]).filter(x=>x.every(v=>typeof v==='number'&&Number.isFinite(v)))})}else for(const tier of ['1','2','3'])series.push({name:label(r)+' β='+r.config.beta+' Tier '+tier,points:r.training.map(x=>[x.step,(x.pass_rate_by_tier||x.tier_pass_rates||{})[tier]]).filter(x=>x.every(v=>typeof v==='number'&&Number.isFinite(v)))})});chart(card,title,series);rl.append(card)});
const held=document.getElementById('heldout');if(data.analysis&&data.analysis.transfer){held.append(el('h3','Transfer ranking analysis'),el('pre',JSON.stringify(data.analysis.transfer,null,2)))}const hs=runs.filter(r=>r.heldout&&r.config.evidence_kind==='model_run'),transfers=genuine.filter(r=>r.config.split==='transfer'||r.config.track==='B');if(!hs.length&&!transfers.length)pending(held,'Held-out before/after evaluation and five-task transfer test pending cloud/API execution.');hs.forEach(r=>{held.append(el('h3',label(r)),el('pre',JSON.stringify(r.heldout,null,2)))});transfers.forEach(r=>held.append(el('p',label(r)+' transfer mean '+fmt(r.summary.mean)+' ± '+fmt(r.summary.sd)+' ('+r.summary.complete+' records). Small-sample ranking agreement is reported in the advanced-track write-up.')));
</script></body></html>'''
