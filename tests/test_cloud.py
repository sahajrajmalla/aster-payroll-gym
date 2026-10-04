"""No test here imports or installs a training/model library."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from aster_gym.bundles import BundleError, export_bundle, import_bundle, validate_payloads, validate_splits
from aster_gym.cloud.common import TrainConfig, split_manifest
from aster_gym.cloud.guard import CloudOnlyError, require_colab, require_cuda
from aster_gym.versions import REWARD_VERSION, RULESET_VERSION, SCHEMA_VERSION, rules_hash


def sample_run(path: Path, **overrides: object) -> Path:
    path.mkdir(parents=True)
    config = {"run_id": "example", "model": "synthetic-test-provider", "seed": 8,
              "ruleset_version": RULESET_VERSION, "reward_version": REWARD_VERSION,
              "schema_version": SCHEMA_VERSION, "rules_hash": rules_hash(),
              "taskset_hash": "a" * 64, "prompt_hashes": {"t1": "b" * 64},
              "evidence_kind": "model_run", "status": "complete", "task_count": 1, "rollouts": 3,
              **overrides}
    # Test-only artifact construction; these ephemeral values never become submission evidence.
    records = [{"task_id": "t1", "tier": 1, "rollout": i, "status": "complete", "score": score,
                "components": {}, "latency_s": 1.0, "tokens": 3, "cost_usd": 0, "mode": "single"}
               for i, score in enumerate([0.0, 0.5, 1.0])]
    (path / "config.json").write_text(json.dumps(config))
    (path / "scores.json").write_text(json.dumps({"records": records}))
    (path / "metrics.json").write_text(json.dumps({"aggregate": {"mean": 999999}}))
    (path / "transcript.jsonl").write_text("".join(json.dumps({"task_id": "t1", "rollout": i,
              "messages": [{"role": "assistant", "content": "test fixture"}], "status": "complete"}) + "\n"
              for i in range(3)))
    return path


def make_zip(path: Path, files: dict[str, bytes]) -> Path:
    manifest = {"bundle_version": "1.0", "files": {name: {"sha256": hashlib.sha256(data).hexdigest(),
                                         "bytes": len(data)} for name, data in files.items()}}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def test_local_guard_blocks_before_torch_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aster_gym.cloud.guard.platform.system", lambda: "Darwin")
    for explicit in (False, True):
        with pytest.raises(CloudOnlyError):
            require_cuda(explicit=explicit)
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_importing_all_cloud_entrypoints_does_not_import_ml() -> None:
    script = """import sys
import aster_gym.cloud.train
import aster_gym.cloud.evaluate
import aster_gym.bundles
assert not any(k in sys.modules for k in ('torch','transformers','trl','peft','accelerate','datasets'))
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_no_implicit_consent() -> None:
    with pytest.raises(CloudOnlyError, match="Explicit"):
        require_colab(explicit=False)


def test_cli_refuses_locally_without_reading_config() -> None:
    for module, flag in (("train", "--start-training"), ("evaluate", "--start-inference")):
        args = [sys.executable, "-m", f"aster_gym.cloud.{module}", flag, "--config", "/does/not/exist"]
        if module == "evaluate":
            args.extend(["--output", "/does/not/exist"])
        result = subprocess.run(args, capture_output=True, text=True, env={**os.environ})
        assert result.returncode != 0
        assert "Cloud-only" in result.stderr
        assert "FileNotFoundError" not in result.stderr


def test_training_config_is_pinned_and_grouped() -> None:
    config = TrainConfig.model_validate_json(Path("configs/train.json").read_text())
    assert len(config.model_revision) == 40
    assert config.num_generations == 4
    assert config.beta_values == [0.001, 0.1]
    for bad in ({"num_generations": 2}, {"beta_values": [0.0, 0.1]}, {"model_revision": "main"},
                {"batch_size": 1, "gradient_accumulation_steps": 3},
                {"beta_values": [float("nan"), 0.1]}, {"beta_values": [float("inf"), 0.1]},
                {"lora_alpha": 0}):
        with pytest.raises(ValueError):
            TrainConfig.model_validate({**config.model_dump(), **bad})


def test_frozen_repository_splits_pass() -> None:
    config = TrainConfig.model_validate_json(Path("configs/train.json").read_text())
    if not Path(config.train_path).exists():
        pytest.skip("Frozen data not generated yet")
    validate_splits(split_manifest(config))
    duplicate = config.model_copy(update={"validation_path": config.train_path})
    with pytest.raises(ValueError, match="leakage"):
        split_manifest(duplicate)


def test_data_only_roundtrip_recomputes_aggregates(tmp_path: Path) -> None:
    run = sample_run(tmp_path / "run")
    (run / "model.safetensors").write_text("must not be exported")
    archive = export_bundle(run, tmp_path / "results.zip")
    with zipfile.ZipFile(archive) as source:
        assert all(not name.endswith("safetensors") for name in source.namelist())
    result = import_bundle(archive, tmp_path / "imported")
    metrics = json.loads((result / "metrics.json").read_text())
    assert metrics["aggregate"]["mean"] == 0.5
    assert metrics["aggregate"]["sd"] == 0.5
    with pytest.raises(BundleError, match="already exists"):
        import_bundle(archive, result)


@pytest.mark.parametrize("name", ["../config.json", "/config.json", "a/../../config.json", "a\\config.json",
                                  "run.pkl", "train.py", "a//config.json"])
def test_unsafe_members_rejected(tmp_path: Path, name: str) -> None:
    archive = make_zip(tmp_path / "bad.zip", {name: b"{}"})
    with pytest.raises(BundleError):
        import_bundle(archive, tmp_path / "imported")
    assert not (tmp_path / "imported").exists()


def test_symlink_and_encrypted_like_member_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    with zipfile.ZipFile(archive, "w") as source:
        member = zipfile.ZipInfo("config.json")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        source.writestr(member, "elsewhere")
    with pytest.raises(BundleError, match="Symlink"):
        import_bundle(archive, tmp_path / "imported")


def test_checksum_and_scorer_version_rejected(tmp_path: Path) -> None:
    run = sample_run(tmp_path / "run")
    files = {p.name: p.read_bytes() for p in run.iterdir()}
    archive = make_zip(tmp_path / "mismatch.zip", files)
    with zipfile.ZipFile(archive, "a") as target:
        with pytest.warns(UserWarning):
            target.writestr("scores.json", "{}")
    with pytest.raises(BundleError, match="duplicate"):
        import_bundle(archive, tmp_path / "imported")
    config = json.loads(files["config.json"])
    config["reward_version"] = "OLD"
    files["config.json"] = json.dumps(config).encode()
    archive = make_zip(tmp_path / "old.zip", files)
    with pytest.raises(BundleError, match="reward_version"):
        import_bundle(archive, tmp_path / "old")


@pytest.mark.parametrize("raw", [b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'{"x":1,"x":2}'])
def test_nonfinite_and_duplicate_json_rejected(tmp_path: Path, raw: bytes) -> None:
    run = sample_run(tmp_path / "run")
    files = {p.name: p.read_bytes() for p in run.iterdir()}
    files["metrics.json"] = raw
    archive = make_zip(tmp_path / "bad.zip", files)
    with pytest.raises(BundleError, match="JSON"):
        import_bundle(archive, tmp_path / "imported")


def test_fixture_rejected_and_cannot_look_complete_without_evidence(tmp_path: Path) -> None:
    run = sample_run(tmp_path / "run", evidence_kind="fixture")
    files = {p.name: p.read_bytes() for p in run.iterdir()}
    archive = make_zip(tmp_path / "fixture.zip", files)
    with pytest.raises(BundleError, match="Fixture"):
        import_bundle(archive, tmp_path / "imported")
    import_bundle(archive, tmp_path / "test-only", allow_fixtures=True)
    assert json.loads((tmp_path / "test-only/config.json").read_text())["evidence_kind"] == "fixture"


def test_split_leakage_is_rejected() -> None:
    value = {name: {"ids": [str(i)], "input_hashes": [str(i) * 64], "seeds": [i]}
             for i, name in enumerate(("train", "validation", "evaluation"))}
    validate_splits(value)
    value["evaluation"]["seeds"] = [0]
    with pytest.raises(BundleError, match="leakage"):
        validate_splits(value)


def test_training_requires_actual_curves_and_reference_proof(tmp_path: Path) -> None:
    run = sample_run(tmp_path / "run", run_type="training")
    splits = {name: {"ids": [str(i)], "input_hashes": [str(i) * 64], "seeds": [i]}
              for i, name in enumerate(("train", "validation", "evaluation"))}
    (run / "splits.json").write_text(json.dumps(splits))
    with pytest.raises(BundleError, match="curves"):
        export_bundle(run, tmp_path / "bad.zip")


def test_notebook_has_no_saved_execution_or_results() -> None:
    notebook = json.loads(Path("notebooks/aster_colab.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
    code_cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert "COLAB_ONLY" in code_cells[0]
    assert all("assert COLAB_ONLY" in cell for cell in code_cells[1:])


def test_complete_evaluation_cannot_omit_or_duplicate_replicates(tmp_path: Path) -> None:
    run = sample_run(tmp_path / "run")
    payloads = {p.name: p.read_bytes() for p in run.iterdir()}
    records = json.loads(payloads["scores.json"])
    records["records"].pop()
    payloads["scores.json"] = json.dumps(records).encode()
    with pytest.raises(BundleError, match="missing frozen"):
        validate_payloads(payloads)
    payloads["scores.json"] = (run / "scores.json").read_bytes()
    first = payloads["transcript.jsonl"].splitlines()[0]
    payloads["transcript.jsonl"] += first + b"\n"
    with pytest.raises(BundleError, match="Duplicate or missing transcripts"):
        validate_payloads(payloads)


@pytest.mark.parametrize("component", [
    {"score": True, "weight": 1.0, "clauses": ["R1"], "code": "bad"},
    {"score": 1.0, "weight": 1.0, "clauses": ["R999"], "code": "bad"},
    {"score": 2.0, "weight": 1.0, "clauses": ["R1"], "code": "bad"},
    {"score": 1.0, "weight": 1.0, "clauses": ["R1"], "code": "bad", "extra": "bad"},
])
def test_import_validates_clause_level_components(tmp_path: Path, component: dict[str, object]) -> None:
    run = sample_run(tmp_path / "run")
    payloads = {p.name: p.read_bytes() for p in run.iterdir()}
    records = json.loads(payloads["scores.json"])
    records["records"][0]["components"] = {"correctness": component}
    payloads["scores.json"] = json.dumps(records).encode()
    with pytest.raises(BundleError):
        validate_payloads(payloads)


def test_malformed_transcript_identity_fails_safely(tmp_path: Path) -> None:
    run = sample_run(tmp_path / "run")
    payloads = {p.name: p.read_bytes() for p in run.iterdir()}
    rows = [json.loads(line) for line in payloads["transcript.jsonl"].splitlines()]
    rows[0]["task_id"] = {"unhashable": "value"}
    payloads["transcript.jsonl"] = b"\n".join(json.dumps(row).encode() for row in rows)
    with pytest.raises(BundleError, match="transcript identity"):
        validate_payloads(payloads)


def test_resume_completed_beta_cannot_change_configuration(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    # Test only metadata orchestration. The CUDA check is stubbed; no ML package is imported.
    from aster_gym.cloud import train

    manifest = {name: {"ids": [name], "input_hashes": [str(i) * 64], "seeds": [i]}
                for i, name in enumerate(("train", "validation", "evaluation"))}
    config = TrainConfig(model_revision="a" * 40, output_dir=str(tmp_path / "cloud"))
    folder = Path(config.output_dir) / "beta-0.001"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"status": "complete", "experiment_hash": "changed"}))
    monkeypatch.setattr(train, "require_cuda", lambda **kwargs: None)
    monkeypatch.setattr(train, "split_manifest", lambda unused: manifest)
    monkeypatch.setattr(train, "train_beta", lambda *args, **kwargs: pytest.fail("Must reject before training"))
    with pytest.raises(ValueError, match="changed configuration"):
        train.run_sweep(config, explicit=True, resume=True)
    assert "torch" not in sys.modules


def training_contract_fixture(path: Path) -> dict[str, bytes]:
    """Ephemeral importer-contract data, never an experiment or reported learning curve."""
    run = sample_run(path, run_type="training", optimizer_steps=3)
    payloads = {p.name: p.read_bytes() for p in run.iterdir()}
    splits = {name: {"ids": ["t1" if i == 0 else name], "input_hashes": [str(i) * 64], "seeds": [i]}
              for i, name in enumerate(("train", "validation", "evaluation"))}
    rows = [{"step": step, "beta": 0.001, "reward": 0.5, "kl": 0.1, "kl_penalty": 0.0001,
             "entropy": 1.0, "completion_length": 100.0, "identical_reward_group_fraction": 0.5,
             "pass_rate_by_tier": {"1": 0.5, "2": None, "3": None}} for step in range(1, 4)]
    proof = {"initial_logits_equal": True, "reference_weights_unchanged": True,
             "reference_sha256_before": "e" * 64, "reference_sha256_after": "e" * 64}
    payloads["splits.json"] = json.dumps(splits).encode()
    payloads["training_metrics.jsonl"] = b"\n".join(json.dumps(row).encode() for row in rows)
    payloads["reference_proof.json"] = json.dumps(proof).encode()
    return payloads


def test_training_contract_accepts_consistent_step_and_reference_metadata(tmp_path: Path) -> None:
    payloads = training_contract_fixture(tmp_path / "contract-only")
    assert validate_payloads(payloads)["config.json"]["optimizer_steps"] == 3


@pytest.mark.parametrize("change", [
    {"kl_penalty": 0.5}, {"identical_reward_group_fraction": 2.0},
    {"pass_rate_by_tier": {"1": True, "2": None, "3": None}}, {"beta": -0.01},
])
def test_training_import_rejects_inconsistent_metrics(tmp_path: Path, change: dict[str, object]) -> None:
    payloads = training_contract_fixture(tmp_path / "contract-only")
    rows = [json.loads(row) for row in payloads["training_metrics.jsonl"].splitlines()]
    rows[0].update(change)
    payloads["training_metrics.jsonl"] = b"\n".join(json.dumps(row).encode() for row in rows)
    with pytest.raises(BundleError):
        validate_payloads(payloads)


def test_training_import_requires_every_optimizer_step_and_matching_hashes(tmp_path: Path) -> None:
    payloads = training_contract_fixture(tmp_path / "contract-only")
    original = payloads["training_metrics.jsonl"]
    rows = original.splitlines()
    payloads["training_metrics.jsonl"] = b"\n".join([rows[0], rows[2]])
    with pytest.raises(BundleError, match="optimizer step"):
        validate_payloads(payloads)
    payloads["training_metrics.jsonl"] = original
    proof = json.loads(payloads["reference_proof.json"])
    proof["reference_sha256_after"] = "f" * 64
    payloads["reference_proof.json"] = json.dumps(proof).encode()
    with pytest.raises(BundleError, match="parameter hashes"):
        validate_payloads(payloads)
