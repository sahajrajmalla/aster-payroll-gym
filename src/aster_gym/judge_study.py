"""Bounded uncached reliability study using the production eligibility/scorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evidence import judge_reliability
from .generator import read_taskset
from .judge import judge_identity
from .scoring import make_judge, score
from .versions import REWARD_VERSION, RULESET_VERSION, rules_hash


async def run_study(packet_path: str | Path) -> dict[str, Any]:
    path = Path(packet_path)
    packet = json.loads(path.read_text())
    if (packet.get("rules_hash") != rules_hash() or packet.get("reward_version") != REWARD_VERSION
            or packet.get("ruleset_version") != RULESET_VERSION):
        raise ValueError("JUDGE_REVIEW_PACKET_VERSION_MISMATCH")
    examples = packet["examples"]
    if len(examples) != 15 or len({r["example_id"] for r in examples}) != 15:
        raise ValueError("EXACTLY_FIFTEEN_DISTINCT_EXAMPLES_REQUIRED")
    tasks = {t.id: t for t in read_taskset(packet["task_source"])}
    judge = make_judge()
    if judge is None:
        raise ValueError("JUDGE_API_KEY_REQUIRED; no ratings have been fabricated")
    if not judge.verified_free:
        await judge.aclose()
        raise ValueError("VERIFIED_ZERO_DOLLAR_JUDGE_REQUIRED")

    class UncachedJudge:
        async def grade(self, task, answer):
            return await judge.grade(task, answer, bypass_cache=True)

    try:
        if packet.get("judge_prompt_hash") not in (None, judge.prompt_hash) or packet.get("judge_model") not in (
                None, judge.model):
            raise ValueError("JUDGE_CONFIGURATION_CHANGED; prepare a new review packet")
        identity = judge_identity(judge)
        has_ratings = any(row.get("ratings") for row in examples)
        if packet.get("judge_settings") != identity and (has_ratings or packet.get("judge_settings") is not None):
            raise ValueError("JUDGE_CONFIGURATION_CHANGED; preserve prior ratings and prepare a new review packet")
        packet.update(judge_prompt_hash=judge.prompt_hash, judge_model=judge.model, judge_settings=identity)
        for row in examples:
            task = tasks[row["task_id"]]
            if row["task_input_hash"] != task.input_hash:
                raise ValueError("JUDGE_TASK_FINGERPRINT_MISMATCH")
            ratings = row.setdefault("ratings", [])
            # A malformed judge response is an observed abstention, not a reason to
            # sample again until a valid response conceals the reliability failure.
            complete_repeats = {r["repeat"] for r in ratings if r.get("status") == "complete"
                                or r.get("error_code") == "JUDGE_INVALID_OUTPUT"}
            for repeat in range(3):
                if repeat in complete_repeats:
                    continue
                began = len(judge.ledger)
                graded = await score(task, row["candidate"], judge=UncachedJudge())
                component = graded.components.get("judge")
                label = int(component.score * 2) if component and component.code == "JUDGE_GRADED" else None
                prior_ledger: list[dict[str, Any]] = next(
                    (r.get("ledger", []) for r in ratings if r.get("repeat") == repeat), [])
                ratings[:] = [r for r in ratings if r.get("repeat") != repeat]
                ratings.append({"repeat": repeat, "status": "complete" if label is not None else "pending",
                                "label": label, "scorer_status": graded.status, "gate": graded.gate,
                                "error_code": graded.error_code, "ledger": prior_ledger + judge.ledger[began:]})
                packet["reliability"] = judge_reliability(examples)
                path.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
                if graded.status == "pending" and graded.error_code != "JUDGE_INVALID_OUTPUT":
                    return packet["reliability"]
        return packet["reliability"]
    finally:
        await judge.aclose()
