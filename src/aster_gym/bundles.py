"""JSON-only interchange. Never deserializes tensors, code, or pickle artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from aster_gym.schemas import ComponentScore
from aster_gym.versions import REWARD_VERSION, RULESET_VERSION, SCHEMA_VERSION, rules_hash

MAX_FILE = 32 * 1024 * 1024
MAX_TOTAL = 100 * 1024 * 1024
MAX_FILES = 256
ALLOWED = {"config.json", "scores.json", "metrics.json", "transcript.jsonl", "training_metrics.jsonl",
           "splits.json", "reference_proof.json", "heldout.json", "checkpoint_metadata.json",
           "run_status.json", "training_completions.jsonl", "selection.json", "failure.json", "budget.json"}
REQUIRED = {"config.json", "scores.json", "metrics.json", "transcript.jsonl"}


class BundleError(ValueError):
    """An untrusted results bundle violated its data-only contract."""


def _bad_constant(value: str) -> None:
    raise BundleError(f"Non-finite number: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError("Duplicate JSON key")
        result[key] = value
    return result


def _finite_tree(value: Any, depth: int = 0) -> None:
    if depth > 80:
        raise BundleError("Excessive JSON nesting")
    if isinstance(value, float) and not math.isfinite(value):
        raise BundleError("Non-finite JSON number")
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item, depth + 1)
    if isinstance(value, list):
        for item in value:
            _finite_tree(item, depth + 1)


def _parse(data: bytes, name: str) -> Any:
    try:
        text = data.decode("utf-8")
        if name.endswith(".jsonl"):
            value = [json.loads(line, parse_constant=_bad_constant, object_pairs_hook=_unique_object)
                     for line in text.splitlines() if line.strip()]
        else:
            value = json.loads(text, parse_constant=_bad_constant, object_pairs_hook=_unique_object)
        _finite_tree(value)
        return value
    except (ValueError, UnicodeError, RecursionError) as error:
        raise BundleError(f"Invalid JSON in {name}") from error


def _safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or any(p in {".", "..", ""} for p in name.split("/"))
            or not re.fullmatch(r"[A-Za-z0-9_./-]+", name) or len(name) > 240):
        raise BundleError("Unsafe bundle path")
    return path


def recompute_metrics(records: list[dict[str, Any]], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Replace untrusted aggregates with the exact evaluation-harness summarizer."""
    from aster_gym.eval import summarize_records

    config = config or {}
    return summarize_records(records, expected_tasks=config.get("task_count"), rollouts=config.get("rollouts"))


def _validate_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"records"} or not isinstance(value["records"], list):
        raise BundleError("scores.json must contain records")
    seen: set[tuple[str, int]] = set()
    for record in value["records"]:
        if not isinstance(record, dict):
            raise BundleError("Score record must be an object")
        if not isinstance(record.get("task_id"), str) or type(record.get("rollout")) is not int:
            raise BundleError("Invalid task/rollout identity")
        key = (record["task_id"], record["rollout"])
        if key in seen or record["rollout"] < 0:
            raise BundleError("Duplicate or negative rollout")
        seen.add(key)
        if record.get("tier") not in (1, 2, 3) or isinstance(record.get("tier"), bool):
            raise BundleError("Invalid tier")
        if record.get("status") not in {"complete", "pending", "failed", "error", "budget_exhausted", "provider_error", "cost_exhausted", "timeout", "internal_error"}:
            raise BundleError("Invalid score status")
        score = record.get("score")
        if score is not None and (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
            raise BundleError("Invalid score range")
        if (record["status"] == "complete") != (score is not None):
            raise BundleError("A score must correspond to a completed result")
        if not isinstance(record.get("components"), dict):
            raise BundleError("Missing reward components")
        for name, component in record["components"].items():
            if not isinstance(name, str) or not isinstance(component, dict):
                raise BundleError("Invalid reward component")
            if any(type(component.get(field)) not in (int, float) for field in ("score", "weight")):
                raise BundleError("Invalid reward component number")
            try:
                parsed_component = ComponentScore.model_validate(component)
            except ValidationError as error:
                raise BundleError("Invalid reward component schema") from error
            if not all(clause in {f"R{i}" for i in range(1, 12)} for clause in parsed_component.clauses):
                raise BundleError("Unknown reward clause")
        for field in ("latency_s", "cost_usd", "tokens"):
            number = record.get(field, 0)
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise BundleError(f"Invalid {field}")
    return value["records"]


def validate_splits(value: Any) -> None:
    if not isinstance(value, dict) or not {"train", "validation", "evaluation"} <= value.keys():
        raise BundleError("Missing split manifest")
    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for split in ("train", "validation", "evaluation"):
        item = value[split]
        if not isinstance(item, dict):
            raise BundleError("Invalid split metadata")
        ids, hashes, seeds = item.get("ids"), item.get("input_hashes"), item.get("seeds")
        if not isinstance(ids, list) or not isinstance(hashes, list) or not isinstance(seeds, list) or not ids:
            raise BundleError("Empty or invalid split")
        if not (len(ids) == len(hashes) == len(seeds)):
            raise BundleError("Split metadata length mismatch")
        if not all(isinstance(v, str) for v in ids + hashes) or not all(type(v) is int for v in seeds):
            raise BundleError("Invalid split identity types")
        if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
            raise BundleError("Invalid input fingerprint")
        if (len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes) or
                seen_ids.intersection(ids) or seen_hashes.intersection(hashes) or seen_seeds.intersection(seeds)):
            raise BundleError("Duplicate inputs/IDs or cross-split seed leakage")
        seen_hashes.update(hashes)
        seen_ids.update(ids)
        seen_seeds.update(seeds)


def validate_payloads(payloads: dict[str, bytes], *, allow_fixtures: bool = False) -> dict[str, Any]:
    parsed = {name: _parse(data, name) for name, data in payloads.items()}
    configs = [name for name in parsed if PurePosixPath(name).name == "config.json"]
    if not configs:
        raise BundleError("No results runs")
    for config_name in configs:
        directory = str(PurePosixPath(config_name).parent)
        prefix = "" if directory == "." else directory + "/"
        if not {prefix + item for item in REQUIRED} <= parsed.keys():
            raise BundleError("Incomplete results artifact contract")
        config = parsed[config_name]
        if not isinstance(config, dict):
            raise BundleError("Invalid run config")
        for key, expected in {"reward_version": REWARD_VERSION, "ruleset_version": RULESET_VERSION,
                              "schema_version": SCHEMA_VERSION, "rules_hash": rules_hash()}.items():
            if config.get(key) != expected:
                raise BundleError(f"Incompatible {key}")
        if config.get("evidence_kind") not in {"model_run", "adversarial_baseline", "fixture"}:
            raise BundleError("Evidence must be explicitly classified")
        if config["evidence_kind"] == "fixture" and not allow_fixtures:
            raise BundleError("Fixture bundles require explicit test-only allowance")
        if config.get("status") not in {"running", "complete", "partial", "failed", "pending"}:
            raise BundleError("Invalid run status")
        for key in ("run_id", "model", "taskset_hash"):
            if not isinstance(config.get(key), str) or not config[key]:
                raise BundleError(f"Missing config identity: {key}")
        if not re.fullmatch(r"[0-9a-f]{64}", config["taskset_hash"]):
            raise BundleError("Invalid taskset fingerprint")
        if type(config.get("seed")) is not int or not isinstance(config.get("prompt_hashes"), dict):
            raise BundleError("Missing seed or prompt provenance")
        for task_id, digest in config["prompt_hashes"].items():
            if not isinstance(task_id, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise BundleError("Invalid prompt fingerprints")
        records = _validate_records(parsed[prefix + "scores.json"])
        if any(r["task_id"] not in config["prompt_hashes"] for r in records):
            raise BundleError("Scored task absent from frozen prompt manifest")
        if config["status"] == "complete" and (not records or any(r["status"] != "complete" for r in records)):
            raise BundleError("Complete run has absent/unfinished results")
        if config.get("run_type") != "training" and ("task_count" in config or "rollouts" in config):
            task_count, rollouts = config.get("task_count"), config.get("rollouts")
            if (type(task_count) is not int or type(rollouts) is not int or task_count < 1
                    or not 1 <= rollouts <= 20 or task_count != len(config["prompt_hashes"])):
                raise BundleError("Invalid evaluation cohort contract")
            if any(record["rollout"] >= rollouts for record in records):
                raise BundleError("Rollout exceeds frozen evaluation contract")
            if config["status"] == "complete" and len(records) != task_count * rollouts:
                raise BundleError("Complete evaluation is missing frozen task/rollout samples")
        transcripts = parsed[prefix + "transcript.jsonl"]
        if not isinstance(transcripts, list) or any(not isinstance(t, dict) or
                not isinstance(t.get("messages"), list) for t in transcripts):
            raise BundleError("Invalid transcript schema")
        if any(not isinstance(t.get("task_id"), str) or type(t.get("rollout")) is not int
               or t["rollout"] < 0 for t in transcripts):
            raise BundleError("Invalid transcript identity")
        if len(transcripts) != len(records):
            raise BundleError("Duplicate or missing transcripts")
        if { (t.get("task_id"), t.get("rollout")) for t in transcripts} != {
                (r["task_id"], r["rollout"]) for r in records}:
            raise BundleError("Scores/transcript identities differ")
        parsed[prefix + "metrics.json"] = recompute_metrics(records, config)
        if config.get("run_type") == "training":
            split_key = prefix + "splits.json"
            if split_key not in parsed:
                raise BundleError("Training run lacks split provenance")
            validate_splits(parsed[split_key])
            curves = parsed.get(prefix + "training_metrics.jsonl", [])
            if config["status"] == "complete" and not curves:
                raise BundleError("Completed training requires actual curves")
            if not isinstance(curves, list):
                raise BundleError("Invalid training curve schema")
            training_ids = set(parsed[split_key]["train"]["ids"])
            if set(config["prompt_hashes"]) != training_ids or any(r["task_id"] not in training_ids for r in records):
                raise BundleError("Training prompt/score identities differ from sealed train split")
            for row in curves:
                if not isinstance(row, dict) or not {"step", "beta", "reward", "kl", "kl_penalty", "entropy",
                         "completion_length", "pass_rate_by_tier", "identical_reward_group_fraction"} <= row.keys():
                    raise BundleError("Missing training metric fields")
                if type(row["step"]) is not int or row["step"] < 1:
                    raise BundleError("Invalid optimizer step")
                for key in ("beta", "reward", "kl", "kl_penalty", "entropy", "completion_length"):
                    if type(row[key]) not in (int, float) or not math.isfinite(row[key]):
                        raise BundleError("Invalid training metric")
                if (row["beta"] <= 0 or not 0 <= row["reward"] <= 1 or row["kl"] < -1e-6
                        or row["entropy"] < 0 or row["completion_length"] < 0
                        or not math.isclose(row["kl_penalty"], row["beta"] * row["kl"], rel_tol=1e-6, abs_tol=1e-8)):
                    raise BundleError("Inconsistent training metric or KL penalty")
                fraction = row["identical_reward_group_fraction"]
                if type(fraction) not in (int, float) or not 0 <= fraction <= 1:
                    raise BundleError("Invalid identical-reward group fraction")
                tier_rates = row["pass_rate_by_tier"]
                if not isinstance(tier_rates, dict) or set(tier_rates) != {"1", "2", "3"} or any(
                        rate is not None and (type(rate) not in (int, float) or not 0 <= rate <= 1)
                        for rate in tier_rates.values()):
                    raise BundleError("Invalid per-tier training metric")
            if curves and [row["step"] for row in curves] != sorted({row["step"] for row in curves}):
                raise BundleError("Training metric steps must be unique and increasing")
            if config["status"] == "complete":
                proof = parsed.get(prefix + "reference_proof.json", {})
                if not isinstance(proof, dict) or not all(proof.get(k) is True for k in
                        ("initial_logits_equal", "reference_weights_unchanged")):
                    raise BundleError("Missing frozen-reference proof")
                before, after = proof.get("reference_sha256_before"), proof.get("reference_sha256_after")
                if (not isinstance(before, str) or not re.fullmatch(r"[0-9a-f]{64}", before)
                        or before != after):
                    raise BundleError("Invalid frozen-reference parameter hashes")
                steps = config.get("optimizer_steps")
                if type(steps) is not int or steps < 1 or [row["step"] for row in curves] != list(range(1, steps + 1)):
                    raise BundleError("Missing curves for an optimizer step")
    for name, data in parsed.items():
        if PurePosixPath(name).name == "splits.json":
            validate_splits(data)
    return parsed


def export_bundle(run_dir: str | Path, output_path: str | Path) -> Path:
    """Export allowlisted JSON files only; model/optimizer state stays in cloud storage."""
    source, destination = Path(run_dir), Path(output_path)
    payloads: dict[str, bytes] = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise BundleError("Symlinks cannot be exported")
        if path.is_file() and path.name in ALLOWED and not any(p.startswith("checkpoint-") for p in path.parts):
            name = path.relative_to(source).as_posix()
            _safe_name(name)
            if path.stat().st_size > MAX_FILE:
                raise BundleError("A result file exceeds 32 MiB; export shorter runs separately")
            payloads[name] = path.read_bytes()
    if len(payloads) > MAX_FILES or sum(map(len, payloads.values())) > MAX_TOTAL:
        raise BundleError("Results exceed compact-bundle limits")
    validate_payloads(payloads)
    manifest = {"bundle_version": "1.0", "files": {name: {"sha256": hashlib.sha256(data).hexdigest(),
                 "bytes": len(data)} for name, data in payloads.items()}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        for name, data in payloads.items():
            archive.writestr(name, data)
    return destination


def import_bundle(archive_path: str | Path, output_dir: str | Path, *, allow_fixtures: bool = False) -> Path:
    """Validate before writing; never extract archive members or overwrite existing runs."""
    source, destination = Path(archive_path), Path(output_dir)
    if source.stat().st_size > MAX_TOTAL or destination.exists():
        raise BundleError("Archive too large or destination already exists")
    try:
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES + 1 or len({i.filename for i in infos}) != len(infos):
                raise BundleError("Too many or duplicate members")
            total = 0
            payloads: dict[str, bytes] = {}
            for info in infos:
                member_path = _safe_name(info.filename)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode) or info.is_dir() or info.flag_bits & 1:
                    raise BundleError("Symlink/directory/encrypted member forbidden")
                if info.filename != "manifest.json" and member_path.name not in ALLOWED:
                    raise BundleError("Only allowlisted JSON artifacts are accepted; no weights/pickle/code")
                if info.file_size > MAX_FILE or info.file_size > max(1, info.compress_size) * 500:
                    raise BundleError("Oversized or suspicious compressed member")
                total += info.file_size
                if total > MAX_TOTAL:
                    raise BundleError("Bundle exceeds uncompressed limit")
                payloads[info.filename] = archive.read(info)
    except (zipfile.BadZipFile, RuntimeError) as error:
        raise BundleError("Invalid ZIP archive") from error
    manifest = _parse(payloads.pop("manifest.json", b"{}"), "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("bundle_version") != "1.0":
        raise BundleError("Unsupported bundle manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(payloads):
        raise BundleError("Manifest members differ from archive")
    for name, data in payloads.items():
        if files[name] != {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}:
            raise BundleError("Checksum or byte-length mismatch")
    parsed = validate_payloads(payloads, allow_fixtures=allow_fixtures)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".aster-import-", dir=destination.parent))
    try:
        for name, value in parsed.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(".jsonl"):
                path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in value))
            else:
                path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        (staging / "import_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
