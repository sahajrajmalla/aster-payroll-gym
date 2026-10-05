"""Optional verifiers adapters. Importing this module does not import an ML stack."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from aster_gym.generator import generate_taskset, task_prompt
from aster_gym.scoring import score
from aster_gym.tools import ToolSession, is_reference_read, parse_tool_arguments

if TYPE_CHECKING:
    import verifiers as vf


def final_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list):
        return ""
    for message in reversed(completion):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role == "assistant":
            content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
            return content if isinstance(content, str) else ""
    return ""


def load_environment(mode: str = "single", split: str = "development", seed: int = 7,
                     n: int = 12, tier: str | int = "all", **kwargs: Any) -> vf.Environment:
    """Load ASTER with a real shared scorer; tools resolve exclusively from rollout state.

    Install the environment extra explicitly (preferably in Colab). The ordinary
    package/API/test path does not load verifiers, datasets, torch, or model weights.
    """
    if mode not in {"single", "tool"}:
        raise ValueError("INVALID_MODE")
    try:
        import verifiers as vf
        from datasets import Dataset
        from verifiers.types import ToolMessage
    except ImportError as exc:
        raise RuntimeError("ENVIRONMENT_EXTRA_REQUIRED: install aster-payroll-gym[environment] in Colab") from exc

    tasks = kwargs.pop("tasks", None)
    if tasks is None:
        tasks = generate_taskset(n=n, seed=seed, tier=tier, split=split)
    task_map = {task.id: task for task in tasks}
    judge = kwargs.pop("judge", None)
    max_turns = kwargs.pop("max_turns", 6)
    timeout_seconds = kwargs.pop("timeout_seconds", 120)
    rows = [{"prompt": [{"role": "user", "content": task_prompt(task, mode=mode)}],
             "answer": task.id, "info": {"task_id": task.id}} for task in tasks]

    class StrictAnswerParser(vf.Parser):
        def parse_answer(self, completion: Any) -> str:
            # No regex salvage or code-fence repair. The shared scorer owns strict JSON validation.
            return final_text(completion)

    class SharedRubric(vf.Rubric):
        async def score_rollout(self, state: Any) -> None:
            task = task_map[state["info"]["task_id"]]
            if state.get("timed_out") or state.get("error") is not None:
                state["aster_score_status"] = "operational_failure"
                raise RuntimeError("ROLLOUT_INFRASTRUCTURE_FAILURE")
            if state.get("stop_condition") == "max_turns_reached" and mode == "tool":
                last = state.get("completion", [])
                last = last[-1] if isinstance(last, list) and last else None
                calls = last.get("tool_calls") if isinstance(last, dict) else getattr(last, "tool_calls", None)
                if calls:
                    state["reward"] = 0.0
                    state["metrics"] = {"MAX_TURNS": 1.0}
                    return
            if mode == "tool" and not state.get("aster_reference_reads", 0):
                state["reward"] = 0.0
                state["metrics"] = {"TOOLS_NOT_USED": 1.0}
                return
            report = await score(task, final_text(state["completion"]), judge=judge)
            state["aster_score_report"] = report.model_dump(mode="json")
            if report.status != "complete":
                # Do not silently substitute a zero for a failed judge.
                raise RuntimeError("JUDGE_PENDING")
            state["reward"] = report.score
            state["metrics"] = {name: part.score for name, part in report.components.items()}

        async def score_group(self, states: list[Any]) -> None:
            # Group dispatch must preserve the exact same audited score semantics.
            for state in states:
                await self.score_rollout(state)

    parser = StrictAnswerParser()
    rubric = SharedRubric(parser=parser)
    common = {"dataset": Dataset.from_list(rows), "parser": parser, "rubric": rubric,
              "pass_threshold": .975, "max_workers": 2, "timeout_seconds": timeout_seconds, **kwargs}
    if mode == "single":
        return vf.SingleTurnEnv(**common)

    # These typed declarations supply the ToolEnv schemas. Execution below always
    # uses the task in that rollout's state, never an environment-global active task.
    def read_document(document_id: str) -> str:
        """Read one source document named in the current task."""
        raise RuntimeError("ROLLOUT_CONTEXT_REQUIRED")

    def lookup_rules(pay_date: str) -> str:
        """Read authoritative clauses for an ISO payroll date."""
        raise RuntimeError("ROLLOUT_CONTEXT_REQUIRED")

    def calculate(expression: str) -> str:
        """Calculate decimal arithmetic using +, -, *, / and parentheses."""
        raise RuntimeError("ROLLOUT_CONTEXT_REQUIRED")

    class IsolatedToolEnv(vf.ToolEnv):
        async def env_response(self, messages: Any, state: Any, **unused: Any) -> list[Any]:
            task = task_map[state["info"]["task_id"]]
            session = ToolSession(task)
            session.calls = state.get("aster_tool_calls", 0)
            message = messages[-1]
            calls = message.get("tool_calls", []) if isinstance(message, dict) else message.tool_calls or []
            outputs = []
            for call in calls[:8]:
                if isinstance(call, dict):
                    name = call.get("function", {}).get("name", call.get("name", ""))
                    arguments = call.get("function", {}).get("arguments", call.get("arguments", ""))
                    call_id = call.get("id", "invalid")
                else:
                    name, arguments, call_id = call.name, call.arguments, call.id
                parsed_arguments = parse_tool_arguments(arguments)
                result = session.call(name, parsed_arguments)
                if is_reference_read(name, parsed_arguments, result):
                    state["aster_reference_reads"] = state.get("aster_reference_reads", 0) + 1
                outputs.append(ToolMessage(role="tool", content=json.dumps(result), tool_call_id=call_id))
            state["aster_tool_calls"] = session.calls
            return outputs

    return IsolatedToolEnv(tools=[read_document, lookup_rules, calculate], max_turns=max_turns,
                           error_formatter=lambda exc: '{"error_code":"TOOL_ERROR"}', **common)
