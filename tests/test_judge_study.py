"""Malformed live judge responses remain observations, not resampled successes."""

import json
from pathlib import Path

import pytest

from aster_gym.judge_study import run_study
from aster_gym.schemas import JudgeResult


@pytest.mark.asyncio
@pytest.mark.parametrize("first_error", ["JUDGE_INVALID_OUTPUT", "PROVIDER_RATE_LIMIT"])
async def test_study_retains_malformed_repeat_and_resumes_without_resampling(tmp_path, monkeypatch, first_error):
    class FakeJudge:
        verified_free = True
        model = "test-only"
        base_url = "https://example.invalid"
        prompt_hash = "test-prompt"
        max_tokens = 512
        reasoning_effort = None

        def __init__(self):
            self.ledger = []
            self.calls = 0

        async def grade(self, task, answer, *, bypass_cache=False):
            assert bypass_cache
            self.calls += 1
            self.ledger.append({"test_call": self.calls})
            if self.calls == 1:
                return JudgeResult(score=0, status="pending", code=first_error)
            return JudgeResult(score=1, status="complete", code="JUDGE_GRADED")

        async def aclose(self):
            pass

    judge = FakeJudge()
    monkeypatch.setattr("aster_gym.judge_study.make_judge", lambda: judge)
    packet = json.loads(Path("reviews/judge-review-packet.json").read_text())
    source = tmp_path / "study.json"
    source.write_text(json.dumps(packet))
    result = await run_study(source)
    invalid = first_error == "JUDGE_INVALID_OUTPUT"
    if not invalid:
        assert judge.calls == 1
        result = await run_study(source)
    calls = 45 if invalid else 46
    assert judge.calls == calls
    assert result["attempted_repeats"] == 45
    assert result["invalid_output_repeats"] == int(invalid)
    assert result["invalid_output_fraction"] == pytest.approx(int(invalid) / 45)
    assert result["repeat_measurement_status"] == "complete"
    assert result["status"] == "partial_or_pending"
    assert result["human_exact_agreement"] is None
    assert result["completed_three_repeat_examples"] == (14 if invalid else 15)
    await run_study(source)
    assert judge.calls == calls
    preserved = json.loads(source.read_text())["examples"][0]["ratings"][0]
    if invalid:
        assert preserved["error_code"] == "JUDGE_INVALID_OUTPUT"
    else:
        assert preserved["ledger"] == [{"test_call": 1}, {"test_call": 2}]
