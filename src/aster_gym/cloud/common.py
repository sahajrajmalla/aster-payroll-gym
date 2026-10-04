"""Lightweight experiment metadata shared by Colab scripts and local import tests."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aster_gym.versions import REWARD_VERSION, RULESET_VERSION, SCHEMA_VERSION, rules_hash, stable_hash


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    seed: int = 20261002
    train_path: str = "data/train.jsonl"
    validation_path: str = "data/validation.jsonl"
    evaluation_path: str = "data/evaluation.jsonl"
    output_dir: str = "/content/drive/MyDrive/aster-gym-results"
    beta_values: list[float] = [0.001, 0.10]
    steps: int = Field(default=80, ge=3, le=10000)
    num_generations: int = Field(default=4, ge=4, le=16)
    temperature: float = Field(default=0.8, gt=0, le=2)
    eval_temperature: float = Field(default=0.7, gt=0, le=2)
    max_completion_length: int = Field(default=256, ge=64, le=1024)
    max_prompt_length: int = Field(default=6000, ge=512, le=16000)
    learning_rate: float = Field(default=5e-5, gt=0, le=0.01)
    batch_size: int = Field(default=1, ge=1, le=8)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=64)
    lora_rank: int = Field(default=8, ge=1, le=64)
    lora_alpha: int = Field(default=16, ge=1, le=256)
    save_steps: int = Field(default=20, ge=1)
    eval_rollouts: int = Field(default=3, ge=3, le=10)

    @model_validator(mode="after")
    def valid_group(self) -> TrainConfig:
        if self.batch_size * self.gradient_accumulation_steps % self.num_generations:
            raise ValueError("Effective batch size must be divisible by grouped completions")
        if (len(self.beta_values) != 2 or len(set(self.beta_values)) != 2
                or any(not math.isfinite(beta) or beta <= 0 for beta in self.beta_values)):
            raise ValueError("Specify exactly two distinct positive KL coefficients")
        return self


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def load_tasks(path: str | Path) -> list[Any]:
    from aster_gym.schemas import Task

    return [Task.model_validate_json(line) for line in Path(path).read_text().splitlines() if line.strip()]


def split_manifest(config: TrainConfig) -> dict[str, Any]:
    from aster_gym.generator import validate_taskset

    manifest: dict[str, Any] = {}
    seen: set[str] = set()
    seen_seeds: set[int] = set()
    seen_ids: set[str] = set()
    for name, path in (("train", config.train_path), ("validation", config.validation_path),
                       ("evaluation", config.evaluation_path)):
        tasks = load_tasks(path)
        validate_taskset(tasks)
        hashes = [task.input_hash for task in tasks]
        task_seeds = {task.seed for task in tasks}
        ids = [task.id for task in tasks]
        if (not hashes or len(hashes) != len(set(hashes)) or seen.intersection(hashes)
                or seen_seeds.intersection(task_seeds) or seen_ids.intersection(ids) or len(set(ids)) != len(ids)):
            raise ValueError("Empty/duplicate dataset or cross-split input/seed/ID leakage")
        seen.update(hashes)
        seen_seeds.update(task_seeds)
        seen_ids.update(ids)
        manifest[name] = {"ids": [task.id for task in tasks], "input_hashes": hashes,
                          "seeds": [task.seed for task in tasks],
                          "taskset_hash": stable_hash([task.model_dump(mode="json") for task in tasks])}
    return manifest


def code_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def versions() -> dict[str, str]:
    return {"ruleset_version": RULESET_VERSION, "reward_version": REWARD_VERSION,
            "schema_version": SCHEMA_VERSION, "rules_hash": rules_hash()}


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()
