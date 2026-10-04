"""CUDA inference provider. This module is safe to import locally; execution is guarded."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from aster_gym.cloud.common import TrainConfig, load_tasks, write_json
from aster_gym.cloud.guard import require_cuda


class ColabProvider:
    """Open-weight provider implementing the same harness contract as remote APIs."""

    def __init__(self, config: TrainConfig, *, explicit: bool, adapter: str | None = None) -> None:
        torch = require_cuda(explicit=explicit)
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        from aster_gym.providers import CostBudget

        self.budget = CostBudget(0)
        self.model = config.model + (f":{Path(adapter).name}" if adapter else ":initial")
        self.config = config
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.model_revision,
                                                       trust_remote_code=False, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.policy = AutoModelForCausalLM.from_pretrained(
            config.model, revision=config.model_revision, trust_remote_code=False,
            torch_dtype=dtype, attn_implementation="sdpa").to("cuda")
        if adapter:
            from peft import PeftModel

            self.policy = PeftModel.from_pretrained(self.policy, adapter, is_trainable=False)
        self.policy.eval()
        set_seed(config.seed)

    async def chat(self, messages: list[dict[str, Any]], *, tools: Any = None,
                   max_tokens: int = 512, temperature: float = 0.7) -> Any:
        from aster_gym.providers import ChatResult

        start = time.monotonic()
        model_messages = messages
        if tools:
            # Explicit adapter makes the small instruct model's tool surface auditable.
            # It performs no planning, rule lookup, or answer repair for the model.
            instruction = ("Use these tools when evidence is needed. To call one tool output exactly "
                '{"tool_call":{"name":"TOOL_NAME","arguments":{...}}}. '
                "After obtaining evidence, return only the required final answer JSON. Tools: "
                + json.dumps(tools, sort_keys=True))
            model_messages = [{"role": "system", "content": instruction}]
            for message in messages:
                if message["role"] == "tool":
                    model_messages.append({"role": "user", "content": "Tool response: " + message["content"]})
                elif message.get("tool_calls"):
                    call = message["tool_calls"][0]["function"]
                    model_messages.append({"role": "assistant", "content": json.dumps({"tool_call": {
                        "name": call["name"], "arguments": json.loads(call["arguments"])}})})
                else:
                    model_messages.append(message)
        text = self.tokenizer.apply_chat_template(model_messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        if inputs["input_ids"].shape[1] > self.config.max_prompt_length:
            raise ValueError("Prompt exceeds configured limit; refusing silent truncation of rule evidence")
        with self.torch.inference_mode():
            generated = self.policy.generate(**inputs, do_sample=True, temperature=temperature,
                                             max_new_tokens=max_tokens, top_p=1.0, max_time=100.0,
                                             pad_token_id=self.tokenizer.pad_token_id,
                                             eos_token_id=self.tokenizer.eos_token_id)
        prompt_tokens = inputs["input_ids"].shape[1]
        completion = generated[0, prompt_tokens:]
        content = self.tokenizer.decode(completion, skip_special_tokens=True)
        answer_message: dict[str, Any] = {"role": "assistant", "content": content}
        if tools:
            try:
                payload = json.loads(content)
                call = payload["tool_call"]
                if set(payload) == {"tool_call"} and set(call) == {"name", "arguments"}:
                    if isinstance(call["name"], str) and isinstance(call["arguments"], dict):
                        answer_message = {"role": "assistant", "content": None, "tool_calls": [{
                            "id": f"call_{int(start * 1000000)}", "type": "function", "function": {
                                "name": call["name"], "arguments": json.dumps(call["arguments"])}}]}
            except (ValueError, KeyError, TypeError):
                pass  # Invalid tool JSON is scored as the model's final output; never repaired.
        return ChatResult(message=answer_message,
                          attempts=[{"attempt": 1, "status": "complete", "provider": "colab_cuda"}],
                          prompt_tokens=int(prompt_tokens), completion_tokens=int(completion.shape[0]),
                          cost_usd=0.0, latency_s=time.monotonic() - start)

    async def aclose(self) -> None:
        del self.policy
        self.torch.cuda.empty_cache()


async def evaluate_policy(config: TrainConfig, *, task_path: str, output_dir: Path,
                          explicit: bool, adapter: str | None = None, judge: Any = None,
                          mode: str = "single") -> dict[str, Any]:
    require_cuda(explicit=explicit)
    from aster_gym.eval import run_evaluation
    from aster_gym.scoring import make_judge

    provider = ColabProvider(config, explicit=explicit, adapter=adapter)
    scoring_judge = judge or make_judge()
    try:
        metrics = await run_evaluation(load_tasks(task_path), output_dir, model=provider.model,
                                      rollouts=config.eval_rollouts, mode=mode, concurrency=1, base_url="colab://cuda",
                                      judge=scoring_judge, provider=provider, seed=config.seed,
                                      max_tokens=config.max_completion_length,
                                      temperature=config.eval_temperature, max_cost=0.0)
        metadata = json.loads((output_dir / "config.json").read_text())
        metadata.update({"model_revision": config.model_revision, "adapter": adapter,
                         "execution_environment": "google_colab_cuda",
                         "tool_scaffold": "explicit-json-tool-envelope-v1" if mode == "tool" else None})
        write_json(output_dir / "config.json", metadata)
        return metrics
    finally:
        await provider.aclose()
        if scoring_judge is not None and hasattr(scoring_judge, "aclose"):
            await scoring_judge.aclose()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Open-weight evaluation; Google Colab CUDA only")
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--tasks", default="data/evaluation.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--mode", choices=["single", "tool"], default="single")
    parser.add_argument("--start-inference", action="store_true")
    args = parser.parse_args(argv)
    require_cuda(explicit=args.start_inference)
    config = TrainConfig.model_validate_json(Path(args.config).read_text())
    asyncio.run(evaluate_policy(config, task_path=args.tasks, output_dir=Path(args.output),
                               explicit=True, adapter=args.adapter, mode=args.mode))


if __name__ == "__main__":
    main()
