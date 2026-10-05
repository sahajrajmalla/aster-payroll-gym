"""Real GRPO/LoRA sweeps; every heavyweight import is below the Colab/CUDA guard."""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from aster_gym.cloud.common import (
    TrainConfig,
    append_json,
    code_revision,
    file_hash,
    load_tasks,
    split_manifest,
    versions,
    write_json,
)
from aster_gym.cloud.guard import require_cuda
from aster_gym.judge import configured_judge_identity, judge_identity
from aster_gym.versions import implementation_hash, stable_hash


def _hash_parameters(model: Any, *, trainable: bool, torch: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.named_parameters():
        if tensor.requires_grad == trainable:
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _experiment_hash(config: TrainConfig, beta: float, smoke: bool,
                     manifest: dict[str, Any], judge_settings: dict[str, object] | None = None) -> str:
    """Bind resumed training to the same configuration, rules, and sealed splits."""
    return stable_hash({"training_config": config.model_dump(), "beta": beta, "smoke": smoke,
                        "splits": manifest, "judge": judge_settings if judge_settings is not None
                        else configured_judge_identity(), "implementation_hash": implementation_hash(), **versions()})


def train_beta(config: TrainConfig, beta: float, *, explicit: bool, resume: bool = False,
               smoke: bool = False) -> Path:
    torch = require_cuda(explicit=explicit)
    # These modules are deliberately absent from the default installation and module-level imports.
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    from aster_gym.eval import summarize_records
    from aster_gym.generator import task_prompt
    from aster_gym.scoring import make_judge, score

    manifest = split_manifest(config)
    tasks = load_tasks(config.train_path)
    task_index = {task.id: task for task in tasks}
    judge = make_judge()
    if judge is None:
        raise RuntimeError("Configure the shared remote judge before training; a replacement reward is forbidden")
    destination = Path(config.output_dir) / ("smoke" if smoke else f"beta-{beta:g}")
    metadata_path = destination / "config.json"
    fingerprint = _experiment_hash(config, beta, smoke, manifest, judge_identity(judge))
    if destination.exists() and not resume:
        raise FileExistsError("Run exists; resume it explicitly or choose a new cloud output directory")
    destination.mkdir(parents=True, exist_ok=True)
    prior = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    if prior and prior.get("experiment_hash") != fingerprint:
        raise ValueError("Cannot resume a changed configuration, rules, or split manifest")
    metadata = {"run_id": destination.name, "run_type": "training", "model": config.model,
                "model_revision": config.model_revision, "seed": config.seed,
                "taskset_hash": manifest["train"]["taskset_hash"], "code_revision": code_revision(),
                "implementation_hash": implementation_hash(),
                "judge": judge_identity(judge),
                "prompt_hashes": {task.id: stable_hash(task_prompt(task)) for task in tasks},
                "evidence_kind": "model_run", "status": "running", "experiment_hash": fingerprint,
                "training_config": config.model_dump(), "beta": beta, "smoke": smoke, **versions()}
    checkpoints = sorted(destination.glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    checkpoint = checkpoints[-1] if resume and checkpoints else None
    if resume and prior and checkpoint is None:
        raise ValueError("No saved optimizer checkpoint exists; choose a fresh output directory")
    write_json(metadata_path, metadata)
    write_json(destination / "splits.json", manifest)
    resume_step = int(checkpoint.name.rsplit("-", 1)[1]) if checkpoint else 0
    # Discard uncommitted completion/metric rows after the last durable optimizer checkpoint.
    retained = [r for r in _read_jsonl(destination / "training_completions.jsonl") if r["step"] <= resume_step]
    for filename in ("training_completions.jsonl", "training_metrics.jsonl", "transcript.jsonl"):
        path = destination / filename
        rows = [r for r in _read_jsonl(path) if r.get("step", 0) <= resume_step] if resume else []
        path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    records = [r["record"] for r in retained]
    write_json(destination / "scores.json", {"records": records})
    write_json(destination / "metrics.json", summarize_records(records))
    set_seed(config.seed)
    tokenizer = None
    policy = None
    trainer = None
    loop = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.model_revision,
                                                  trust_remote_code=False, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        eos_token_id = tokenizer.eos_token_id
        prompts = [{"prompt": [{"role": "user", "content": task_prompt(task)}], "task_id": task.id}
                   for task in tasks]
        for row in prompts:
            tokenized = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
            if len(tokenized) > config.max_prompt_length:
                raise ValueError("A training prompt exceeds max_prompt_length; refusing evidence truncation")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        backbone = AutoModelForCausalLM.from_pretrained(
            config.model, revision=config.model_revision, trust_remote_code=False,
            torch_dtype=dtype, attn_implementation="sdpa").to("cuda")
        policy = get_peft_model(backbone, LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha,
                                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        policy.config.use_cache = False
        if any(parameter.requires_grad and "lora_" not in name for name, parameter in policy.named_parameters()):
            raise RuntimeError("A non-adapter parameter is trainable; frozen-reference invariant violated")
        policy.eval()
        probe = tokenizer("Synthetic Aster payroll rules", return_tensors="pt").to("cuda")
        with torch.no_grad():
            initial_logits = policy(**probe).logits.detach()
            with policy.disable_adapter():
                reference_logits = policy(**probe).logits.detach()
        logits_equal = bool(torch.allclose(initial_logits, reference_logits, atol=1e-5, rtol=1e-5))
        if not logits_equal:
            raise RuntimeError("Adapter-disabled backbone is not the initial reference policy")
        del initial_logits, reference_logits, probe
        frozen_before = _hash_parameters(policy, trainable=False, torch=torch)
        adapter_before = _hash_parameters(policy, trainable=True, torch=torch)
        proof = {"initial_logits_equal": logits_equal, "initial_atol": 1e-5, "initial_rtol": 1e-5,
                 "reference_weights_unchanged": False, "reference_sha256_before": frozen_before,
                 "reference_mechanism": "immutable adapter-disabled initial backbone",
                 "model_revision": config.model_revision, "adapter_sha256_before": adapter_before}
        write_json(destination / "reference_proof.json", proof)
        recent: list[dict[str, Any]] = []
        loop = asyncio.Runner()

        def shared_reward(prompts: Any, completions: list[Any], task_id: list[str],
                          completion_ids: list[list[int]], trainer_state: Any, **kwargs: Any) -> list[float]:
            del kwargs
            rewards: list[float] = []
            step = int(trainer_state.global_step) + 1
            group_start = len(records)
            for index, (task_name, completion) in enumerate(zip(task_id, completions, strict=True)):
                task = task_index[task_name]
                raw = completion[0].get("content", "") if isinstance(completion, list) else completion
                started = time.monotonic()
                ledger_start = len(getattr(judge, "ledger", []))
                report = loop.run(score(task, raw, judge=judge))
                judge_calls = getattr(judge, "ledger", [])[ledger_start:]
                report_data = report.model_dump(mode="json")
                record = {"task_id": task_name, "tier": task.difficulty, "rollout": len(records),
                          "status": report.status, "score": report.score, "components": report_data["components"],
                          "latency_s": time.monotonic() - started, "tokens": len(completion_ids[index]),
                          "cost_usd": sum(c.get("cost_usd", 0) for c in judge_calls),
                          "mode": "single", "step": step}
                records.append(record)
                recent.append(record)
                append_json(destination / "training_completions.jsonl", {
                    "step": step, "group": (group_start + index) // config.num_generations,
                    "task_id": task_name, "completion": raw, "record": record,
                    "truncated": not completion_ids[index] or completion_ids[index][-1] != eos_token_id})
                append_json(destination / "transcript.jsonl", {"task_id": task_name,
                    "rollout": record["rollout"], "step": step, "messages": list(prompts[index]) +
                    [{"role": "assistant", "content": raw}], "attempts": [], "status": report.status,
                    "judge_calls": judge_calls})
                if report.status != "complete" or report.score is None:
                    raise RuntimeError("Shared judge/scorer is pending; stopped without substituting a reward")
                rewards.append(report.score)
            write_json(destination / "scores.json", {"records": records})
            return rewards

        class MetricsCallback(TrainerCallback):
            def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
                if not logs or "kl" not in logs:
                    return
                fields = ("reward", "kl", "entropy", "completions/mean_length",
                          "frac_reward_zero_std", "completions/clipped_ratio")
                if any(not math.isfinite(float(logs.get(k, float("nan")))) for k in fields):
                    raise RuntimeError("Missing/non-finite required training metrics")
                components: dict[str, list[float]] = defaultdict(list)
                for record in recent:
                    for name, component in record["components"].items():
                        components[name].append(component["score"])
                row = {"step": int(state.global_step), "beta": beta, "reward": float(logs["reward"]),
                       "mean_reward": float(logs["reward"]), "kl": float(logs["kl"]),
                       "kl_penalty": beta * float(logs["kl"]), "entropy": float(logs["entropy"]),
                       "completion_length": float(logs["completions/mean_length"]),
                       "identical_reward_group_fraction": float(logs["frac_reward_zero_std"]),
                       "truncated_fraction": float(logs["completions/clipped_ratio"]),
                       "pass_rate_by_tier": {str(t): mean([r["score"] >= 0.975 for r in recent if r["tier"] == t])
                           if any(r["tier"] == t for r in recent) else None for t in (1, 2, 3)},
                       "reward_components": {k: mean(v) for k, v in components.items()},
                       "raw_logs": {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))
                                    and math.isfinite(v)}}
                append_json(destination / "training_metrics.jsonl", row)
                recent.clear()

        arguments = GRPOConfig(
            output_dir=str(destination), max_steps=3 if smoke else config.steps,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_generations=config.num_generations, temperature=config.temperature,
            max_completion_length=config.max_completion_length, learning_rate=config.learning_rate,
            beta=beta, loss_type="grpo", scale_rewards="group", num_iterations=1,
            epsilon=0.2, sync_ref_model=False, use_bias_correction_kl=False,
            logging_steps=1, save_steps=1 if smoke else config.save_steps, save_total_limit=3,
            save_only_model=False, report_to="none", bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(), gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}, optim="adamw_torch",
            seed=config.seed, data_seed=config.seed, dataloader_num_workers=0,
            use_vllm=False, mask_truncated_completions=False, remove_unused_columns=False)
        trainer = GRPOTrainer(model=policy, args=arguments, reward_funcs=shared_reward,
                              train_dataset=Dataset.from_list(prompts), processing_class=tokenizer,
                              callbacks=[MetricsCallback()])
        if trainer.ref_model is not None or trainer.beta != beta:
            raise RuntimeError("Unexpected reference or beta configuration")
        trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        logged_steps = [row["step"] for row in _read_jsonl(destination / "training_metrics.jsonl")]
        if logged_steps != list(range(1, int(trainer.state.global_step) + 1)):
            raise RuntimeError("Required metrics were not recorded for every optimizer step")
        trainer.save_model(str(destination / "final_adapter"))
        trainer.save_state()
        tokenizer.save_pretrained(destination / "final_adapter")
        frozen_after = _hash_parameters(policy, trainable=False, torch=torch)
        adapter_after = _hash_parameters(policy, trainable=True, torch=torch)
        proof.update({"reference_sha256_after": frozen_after,
                      "reference_weights_unchanged": frozen_after == frozen_before,
                      "adapter_sha256_after": adapter_after,
                      "adapter_weights_changed": adapter_before != adapter_after})
        write_json(destination / "reference_proof.json", proof)
        if frozen_before != frozen_after:
            raise RuntimeError("Reference changed during training")
        checkpoint_files = [p for p in destination.rglob("*") if p.is_file() and p.suffix in
                            {".safetensors", ".pt", ".pth", ".bin"}]
        write_json(destination / "checkpoint_metadata.json", {"storage": "cloud_only",
            "files": [{"cloud_path": str(p), "bytes": p.stat().st_size, "sha256": file_hash(p)}
                      for p in checkpoint_files], "final_adapter": str(destination / "final_adapter")})
        metadata["status"] = "partial" if smoke and adapter_before == adapter_after else "complete"
        metadata["optimizer_steps"] = int(trainer.state.global_step)
        metadata["adapter_weights_changed"] = adapter_before != adapter_after
        write_json(destination / "metrics.json", summarize_records(records))
        write_json(destination / "scores.json", {"records": records})
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["failure_type"] = type(error).__name__
        write_json(destination / "scores.json", {"records": records})
        write_json(destination / "metrics.json", summarize_records(records))
        write_json(destination / "failure.json", {"type": type(error).__name__, "status": "failed",
                     "action": "Inspect Colab traceback; resume a matching durable checkpoint or start a new run"})
        raise
    finally:
        if loop is not None:
            if hasattr(judge, "aclose"):
                loop.run(judge.aclose())
            loop.close()
        write_json(metadata_path, metadata)
        del trainer, policy, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    return destination


def run_sweep(config: TrainConfig, *, explicit: bool, resume: bool = False, smoke: bool = False) -> None:
    require_cuda(explicit=explicit)
    from aster_gym.cloud.evaluate import evaluate_policy

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = split_manifest(config)
    master_split_path = output / "splits.json"
    if master_split_path.exists() and json.loads(master_split_path.read_text()) != manifest:
        raise ValueError("Cannot mix changed sealed splits in an existing experiment directory")
    write_json(output / "splits.json", manifest)
    if smoke:
        train_beta(config, config.beta_values[0], explicit=True, resume=resume, smoke=True)
        return
    candidates = []
    for beta in config.beta_values:
        folder = output / f"beta-{beta:g}"
        done = json.loads((folder / "config.json").read_text()) if (folder / "config.json").exists() else {}
        if resume and done and done.get("experiment_hash") != _experiment_hash(config, beta, False, manifest):
            raise ValueError("Cannot reuse a completed beta run with changed configuration, rules, or splits")
        if not (resume and done.get("status") == "complete"):
            folder = train_beta(config, beta, explicit=True, resume=resume and folder.exists())
        metrics = asyncio.run(evaluate_policy(config, task_path=config.validation_path,
                     output_dir=output / f"validation-{beta:g}", adapter=str(folder / "final_adapter"), explicit=True))
        score_records = json.loads((output / f"validation-{beta:g}" / "scores.json").read_text())["records"]
        if not score_records or any(r["status"] != "complete" for r in score_records):
            raise RuntimeError("Incomplete validation cannot select beta")
        candidates.append((mean([r["score"] for r in score_records]), beta, folder, metrics))
    # Highest validation reward; predetermined tie-break favors stronger KL preservation.
    _, beta, folder, _ = max(candidates, key=lambda item: (item[0], item[1]))
    write_json(output / "selection.json", {"selected_beta": beta, "selection_split": "validation",
          "tie_break": "larger beta", "candidates": [{"beta": b, "mean_reward": reward}
             for reward, b, _, _ in candidates], "evaluation_scored_before_selection": False,
          "heldout_manifest_validated_before_selection": True})
    before = asyncio.run(evaluate_policy(config, task_path=config.evaluation_path,
                                         output_dir=output / "heldout-before", explicit=True))
    after = asyncio.run(evaluate_policy(config, task_path=config.evaluation_path,
                                        output_dir=output / "heldout-after", explicit=True,
                                        adapter=str(folder / "final_adapter")))
    heldout_status = "complete" if before.get("coverage") == after.get("coverage") == 1 else "partial"
    heldout = {"status": heldout_status, "selected_beta": beta,
          "selection_split": "validation", "evaluation_taskset_hash": manifest["evaluation"]["taskset_hash"],
          "before": {"run_dir": "heldout-before", "metrics": before},
          "after": {"run_dir": "heldout-after", "metrics": after}, "evidence_kind": "model_run", **versions()}
    write_json(output / "heldout.json", heldout)
    write_json(output / "heldout-after" / "heldout.json", heldout)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Colab GPU-only GRPO; never executes on this laptop")
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--start-training", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    require_cuda(explicit=args.start_training)
    config = TrainConfig.model_validate_json(Path(args.config).read_text())
    run_sweep(config, explicit=True, resume=args.resume, smoke=args.smoke)


if __name__ == "__main__":
    main()
