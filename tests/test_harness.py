"""All outputs below are fixtures; no model inference or network traffic occurs."""
import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest

from aster_gym.environment import final_text
from aster_gym.eval import run_evaluation, summarize_records
from aster_gym.generator import generate_taskset
from aster_gym.providers import (
    AsyncOpenAIProvider,
    ChatResult,
    CostBudget,
    Pricing,
    ProviderError,
    resolve_reasoning_effort,
)
from aster_gym.tools import ToolSession, calculate, parse_tool_arguments


async def test_dispatch_interval_applies_across_concurrent_rollouts(tmp_path):
    starts = []

    class FixtureProvider:
        budget = None
        reasoning_effort = None

        async def chat(self, messages, **kwargs):
            starts.append(time.monotonic())
            return ChatResult({"role": "assistant", "content": "{}"})

    await run_evaluation(generate_taskset(2, seed=808), tmp_path, model="fixture",
                         provider=FixtureProvider(), evidence_kind="fixture", rollouts=3,
                         concurrency=3, request_interval_s=.02)
    assert len(starts) == 6
    assert all(b - a >= .015 for a, b in zip(starts, starts[1:]))
    assert json.loads((tmp_path / "config.json").read_text())["request_interval_s"] == .02


async def test_invalid_dispatch_interval_fails_before_provider_call(tmp_path):
    with pytest.raises(ValueError, match="INVALID_REQUEST_INTERVAL"):
        await run_evaluation(generate_taskset(1), tmp_path, model="fixture", provider=object(),
                             evidence_kind="fixture", request_interval_s=float("nan"))


@pytest.mark.parametrize("expression", ["__import__('os').system('echo bad')", "10**10000000", "1/0",
                                        "[1]", "True+2", "1e100000", "1 " * 1000, "(1).__class__"])
def test_calculator_rejects_unsafe(expression):
    assert "error_code" in calculate(expression)


def test_calculator_decimal_and_tool_isolation():
    assert calculate("0.1+0.2")["value"] == "0.3"
    assert calculate("(9 + 1) * 2 / 4")["value"] == "5"
    first = SimpleNamespace(context_files={"a.txt": "first payroll"})
    second = SimpleNamespace(context_files={"a.txt": "second payroll"})
    a, b = ToolSession(first), ToolSession(second)
    assert a.call("read_document", {"document_id": "a.txt"})["content"] == "first payroll"
    assert b.call("read_document", {"document_id": "a.txt"})["content"] == "second payroll"
    assert a.call("read_document", {"document_id": "../ground_truth.json"}) == {"error_code": "DOCUMENT_NOT_FOUND"}
    assert "rules" in a.call("lookup_rules", {"pay_date": "2030-01-01"})
    assert a.call("lookup_rules", {"pay_date": "20300101"}) == {"error_code": "INVALID_DATE"}
    assert a.call("calculate", {"expression": "2", "secret": "bad"}) == {"error_code": "INVALID_ARGUMENTS"}
    limited = ToolSession(first, max_calls=1)
    limited.call("calculate", {"expression": "1"})
    assert limited.call("calculate", {"expression": "1"}) == {"error_code": "TOOL_LIMIT"}


@pytest.mark.parametrize("value", ['{"a":1,"a":2}', '[]', '{"x":NaN}', 'invalid', 'x'*4096])
def test_strict_tool_argument_parsing(value):
    assert parse_tool_arguments(value) is None


async def test_concurrent_budget_reservations():
    budget = CostBudget(.10)
    reservations = await asyncio.gather(*(budget.reserve(.04) for _ in range(3)), return_exceptions=True)
    assert sum(isinstance(value, ProviderError) for value in reservations) == 1
    await budget.settle(.04, .01)
    await budget.settle(.04)
    assert budget.spent == pytest.approx(.05)
    assert budget.reserved == pytest.approx(0)


def response(content="{}"):
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}],
                                     "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


async def test_provider_retry_and_no_secret_in_error():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={"secret": "do not expose"}) if len(calls) == 1 else response()
    provider = AsyncOpenAIProvider(model="fixture", base_url="https://fixture.invalid/v1", api_key="TEST_SECRET",
                                   budget=CostBudget(0), pricing=Pricing(0, 0, True), backoff_s=0,
                                   transport=httpx.MockTransport(handler))
    try:
        result = await provider.chat([{"role": "user", "content": "fixture"}])
        assert len(result.attempts) == 2
        assert result.prompt_tokens == 10
        assert "TEST_SECRET" not in json.dumps(result.attempts)
        assert result.attempts[0]["status"] == "PROVIDER_RATE_LIMIT"
    finally:
        await provider.aclose()


async def test_provider_timeout_consumes_reserved_budget():
    def handler(request):
        raise httpx.ReadTimeout("PRIVATE_PROVIDER_INFORMATION", request=request)
    budget = CostBudget(1)
    provider = AsyncOpenAIProvider(model="fixture", base_url="https://fixture.invalid", api_key="fixture",
                                   budget=budget, pricing=Pricing(1, 1), max_retries=0,
                                   transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ProviderError) as caught:
            await provider.chat([{"role": "user", "content": "fixture"}])
        assert str(caught.value) == "PROVIDER_TIMEOUT"
        assert budget.spent > 0 and budget.reserved == 0
    finally:
        await provider.aclose()


def test_unknown_pricing_and_local_endpoint_fail_closed():
    with pytest.raises(ValueError, match="UNKNOWN_PRICING"):
        AsyncOpenAIProvider(model="x", base_url="https://example.invalid", api_key="x",
                            budget=CostBudget(0), pricing=None)
    with pytest.raises(ValueError, match="VERIFIED_FREE"):
        Pricing(0, 0)
    with pytest.raises(ValueError, match="REMOTE_HTTPS"):
        AsyncOpenAIProvider(model="x", base_url="http://localhost:1234", api_key="x",
                            budget=CostBudget(0), pricing=Pricing(0, 0, True))


@pytest.mark.parametrize("effort", [True, 1, [], "extreme", "LOW"])
def test_invalid_reasoning_effort_rejected_before_http_client(effort):
    with pytest.raises(ValueError, match="INVALID_REASONING_EFFORT"):
        AsyncOpenAIProvider(model="fixture", base_url="https://example.invalid", api_key="fixture",
                            budget=CostBudget(0), pricing=Pricing(0, 0, True), reasoning_effort=effort)


def test_reasoning_defaults_are_scoped_to_exact_google_endpoint_and_model():
    endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert resolve_reasoning_effort("gemini-3.8-flash", endpoint) == "low"
    assert resolve_reasoning_effort("gemini-3.5-flash-lite", endpoint) == "low"
    assert resolve_reasoning_effort("other-model", endpoint) is None
    assert resolve_reasoning_effort("gemini-3.8-flash", "https://example.invalid/v1") is None
    assert resolve_reasoning_effort("gemini-3.8-flash", endpoint + "different-path") is None
    for effort in ("none", "minimal"):
        with pytest.raises(ValueError, match="UNSUPPORTED_GEMINI"):
            resolve_reasoning_effort("gemini-3.8-flash", endpoint, effort)


async def test_provider_explicit_effort_is_transmitted_and_cost_accounting_preserved():
    calls = []
    budget = CostBudget(1)

    def handler(request):
        calls.append(json.loads(request.content))
        return response()

    provider = AsyncOpenAIProvider(model="fixture", base_url="https://example.invalid", api_key="fixture",
        budget=budget, pricing=Pricing(1, 1), reasoning_effort="low", transport=httpx.MockTransport(handler))
    try:
        result = await provider.chat([{"role": "user", "content": "fixture"}], max_tokens=512)
        assert calls[0]["reasoning_effort"] == "low" and calls[0]["max_tokens"] == 512
        assert budget.reserved == 0 and budget.actual_spend == pytest.approx(result.cost_usd)
    finally:
        await provider.aclose()


async def test_unrelated_provider_omits_unrequested_effort_field():
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return response()

    provider = AsyncOpenAIProvider(model="fixture", base_url="https://example.invalid", api_key="fixture",
        budget=CostBudget(0), pricing=Pricing(0, 0, True), transport=httpx.MockTransport(handler))
    try:
        await provider.chat([{"role": "user", "content": "fixture"}])
        assert "reasoning_effort" not in calls[0]
    finally:
        await provider.aclose()


@pytest.mark.parametrize("endpoint", ["https://127.0.0.2/v1", "https://0.0.0.0", "https://[::1]",
                                      "https://foo.localhost", "https://localhost./v1",
                                      "https://10.0.0.1", "https://user:secret@example.invalid"])
def test_local_address_variants_and_embedded_credentials_are_rejected(endpoint):
    with pytest.raises(ValueError, match="REMOTE_HTTPS"):
        AsyncOpenAIProvider(model="x", base_url=endpoint, api_key="x",
                            budget=CostBudget(0), pricing=Pricing(0, 0, True))


class FixtureProvider:
    def __init__(self, content="malformed"):
        self.content = content
        self.calls = 0
        self.budget = CostBudget(0)

    async def chat(self, messages, **kwargs):
        self.calls += 1
        return ChatResult({"role": "assistant", "content": self.content}, [{"status": "complete", "cost_usd": 0}],
                          10, 2, 0, .001)


async def test_evaluation_resume_never_resamples_wrong_answer(tmp_path):
    task = generate_taskset(n=1, seed=9, tier=1)[0]
    provider = FixtureProvider()
    first = await run_evaluation([task], tmp_path, model="fixture", rollouts=3,
                                 evidence_kind="fixture", provider=provider)
    assert provider.calls == 3 and first["aggregate"]["mean"] == 0
    assert first["aggregate"]["sd"] == 0
    await run_evaluation([task], tmp_path, model="fixture", rollouts=3,
                         evidence_kind="fixture", provider=provider)
    assert provider.calls == 3
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["evidence_kind"] == "fixture"
    transcript = (tmp_path / "transcript.jsonl").read_text()
    assert "ground_truth" not in transcript
    with pytest.raises(ValueError, match="RESUME_CONFIGURATION"):
        await run_evaluation([task], tmp_path, model="other", evidence_kind="fixture", provider=provider)


async def test_no_judge_evaluation_preserves_pending_traps_without_resampling(tmp_path):
    """Fixture transport only: missing credentials must not rewrite the rubric or coverage."""
    tasks = [generate_taskset(n=1, seed=9, tier=1)[0], generate_taskset(n=1, seed=10, tier=3)[0]]
    assert tasks[1].ground_truth.decision == "needs_information"

    class ExactFixture(FixtureProvider):
        async def chat(self, messages, **kwargs):
            task = tasks[self.calls]
            self.calls += 1
            return ChatResult({"role": "assistant", "content": task.ground_truth.model_dump_json()},
                              [{"status": "complete", "cost_usd": 0}], 10, 2, 0, .001)

    provider = ExactFixture()
    metrics = await run_evaluation(tasks, tmp_path, model="fixture-no-judge", rollouts=1,
                                   evidence_kind="fixture", provider=provider, judge=None)
    records = json.loads((tmp_path / "scores.json").read_text())["records"]
    ordinary = next(row for row in records if row["tier"] == 1)
    trap = next(row for row in records if row["tier"] == 3)
    assert ordinary["status"] == "complete" and ordinary["score"] == 1.0
    assert trap["status"] == "pending" and trap["score"] is None
    assert trap["error_code"] == "JUDGE_UNAVAILABLE"
    assert metrics["coverage"] == .5 and metrics["aggregate"]["mean"] is None
    assert json.loads((tmp_path / "config.json").read_text())["status"] == "partial"
    trace = (tmp_path / "transcript.jsonl").read_text()
    await run_evaluation(tasks, tmp_path, model="fixture-no-judge", rollouts=1,
                         evidence_kind="fixture", provider=provider, judge=None)
    assert provider.calls == 2
    assert (tmp_path / "transcript.jsonl").read_text() == trace


async def test_missing_resume_journal_refuses_resampling_before_mutation(tmp_path):
    task = generate_taskset(n=1, seed=9, tier=1)[0]
    provider = FixtureProvider()
    await run_evaluation([task], tmp_path, model="fixture", rollouts=1,
                         evidence_kind="fixture", provider=provider)
    original_config = (tmp_path / "config.json").read_bytes()
    original_scores = (tmp_path / "scores.json").read_bytes()
    next((tmp_path / ".records").glob("*.json")).unlink()
    with pytest.raises(ValueError, match="RESUME_JOURNAL_MISSING"):
        await run_evaluation([task], tmp_path, model="fixture", rollouts=1,
                             evidence_kind="fixture", provider=provider)
    assert provider.calls == 1
    assert (tmp_path / "config.json").read_bytes() == original_config
    assert (tmp_path / "scores.json").read_bytes() == original_scores


async def test_tool_use_required_and_provider_failures_unscored(tmp_path):
    task = generate_taskset(n=1, seed=9, tier=1)[0]
    provider = FixtureProvider(task.ground_truth.model_dump_json())
    await run_evaluation([task], tmp_path / "tool", model="fixture", mode="tool", rollouts=1,
                         evidence_kind="fixture", provider=provider)
    row = json.loads((tmp_path / "tool" / "scores.json").read_text())["records"][0]
    assert row["score"] == 0 and row["gate"] == "TOOLS_NOT_USED"

    class BrokenProvider(FixtureProvider):
        async def chat(self, messages, **kwargs):
            raise ProviderError("PROVIDER_TIMEOUT")
    result = await run_evaluation([task], tmp_path / "error", model="fixture", rollouts=1,
                                  evidence_kind="fixture", provider=BrokenProvider())
    assert result["completed_samples"] == 0
    assert result["aggregate"]["mean"] is None
    assert result["operational_failures"] == 1


@pytest.mark.parametrize("document,expected_score", [
    ("request.json", 0), ("unrelated-0.txt", 0), ("schedules.json", 1), ("rules.md", 1),
])
async def test_request_or_distractor_read_cannot_unlock_tool_reward(tmp_path, document, expected_score):
    """Even a correct memorized fixture answer must read actual source evidence."""
    task = generate_taskset(n=1, seed=19, tier=1)[0]

    class ReadThenAnswer(FixtureProvider):
        async def chat(self, messages, **kwargs):
            self.calls += 1
            message = ({"role": "assistant", "content": None, "tool_calls": [{
                "id": "call", "function": {"name": "read_document",
                "arguments": json.dumps({"document_id": document})}}]} if self.calls == 1
                else {"role": "assistant", "content": task.ground_truth.model_dump_json()})
            return ChatResult(message, [{"status": "complete", "cost_usd": 0}], 10, 2, 0, .001)

    await run_evaluation([task], tmp_path, model="fixture", mode="tool", rollouts=1,
                         evidence_kind="fixture", provider=ReadThenAnswer())
    row = json.loads((tmp_path / "scores.json").read_text())["records"][0]
    trace = json.loads((tmp_path / "transcript.jsonl").read_text())
    assert row["score"] == expected_score
    assert trace["reference_reads"] == expected_score
    if not expected_score:
        assert row["gate"] == "TOOLS_NOT_USED"


async def test_real_tool_roundtrip_with_fixture_transport(tmp_path):
    task = generate_taskset(n=1, seed=18, tier=1)[0]

    class ToolProvider(FixtureProvider):
        async def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return ChatResult({"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call1", "type": "function", "function": {"name": "lookup_rules",
                     "arguments": '{"pay_date":"2030-01-01"}'}}]})
            assert messages[-1]["role"] == "tool"
            return ChatResult({"role": "assistant", "content": task.ground_truth.model_dump_json()})
    result = await run_evaluation([task], tmp_path, model="fixture", mode="tool", rollouts=1,
                                  evidence_kind="fixture", provider=ToolProvider())
    assert result["sample_summary"]["mean"] == 1


def test_statistics_do_not_report_partial_replicate_as_complete():
    records = [{"task_id": "a", "tier": 1, "rollout": 0, "status": "complete", "score": 1}]
    metrics = summarize_records(records, expected_tasks=2, rollouts=3)
    assert metrics["aggregate"]["mean"] is None
    assert metrics["coverage"] == pytest.approx(1/6)
    assert final_text([{"role": "assistant", "content": "raw"}, {"role": "tool", "content": "tool"}]) == "raw"


async def test_tool_turn_limit_and_timeout(tmp_path):
    task = generate_taskset(n=1, seed=19, tier=1)[0]

    class LoopProvider(FixtureProvider):
        async def chat(self, messages, **kwargs):
            return ChatResult({"role": "assistant", "content": "", "tool_calls": [
                {"id": "call", "type": "function", "function": {"name": "calculate", "arguments": '{"expression":"1"}'}}]})

    await run_evaluation([task], tmp_path / "loop", model="fixture", mode="tool", rollouts=1,
                         max_turns=2, evidence_kind="fixture", provider=LoopProvider())
    row = json.loads((tmp_path / "loop" / "scores.json").read_text())["records"][0]
    assert row["error_code"] == "MAX_TURNS" and row["score"] == 0

    class SlowProvider(FixtureProvider):
        async def chat(self, messages, **kwargs):
            await asyncio.sleep(1)

    result = await run_evaluation([task], tmp_path / "timeout", model="fixture", rollouts=1,
                                  timeout_s=.01, evidence_kind="fixture", provider=SlowProvider())
    assert result["status_counts"] == {"timeout": 1}


async def test_pending_judge_reuses_original_answer(tmp_path):
    from aster_gym.schemas import JudgeResult
    task = generate_taskset(n=1, seed=90, tier=3)[0]
    provider = FixtureProvider(task.ground_truth.model_dump_json())

    class FixtureJudge:
        def __init__(self):
            self.ready = False
        async def grade(self, task, answer):
            return JudgeResult(score=1 if self.ready else 0, status="complete" if self.ready else "pending")
    judge = FixtureJudge()
    first = await run_evaluation([task], tmp_path, model="fixture", rollouts=1,
                                 evidence_kind="fixture", provider=provider, judge=judge)
    assert first["status_counts"] == {"pending": 1}
    judge.ready = True
    second = await run_evaluation([task], tmp_path, model="fixture", rollouts=1,
                                  evidence_kind="fixture", provider=provider, judge=judge)
    assert second["sample_summary"]["mean"] == 1
    assert provider.calls == 1


async def test_judge_deadline_resumes_scoring_without_policy_resampling(tmp_path):
    from aster_gym.schemas import JudgeResult

    task = generate_taskset(n=1, seed=93, tier=3)[0]
    provider = FixtureProvider(task.ground_truth.model_dump_json())

    class SlowJudge:
        def __init__(self):
            self.ready = False

        async def grade(self, task, answer):
            if not self.ready:
                await asyncio.sleep(1)
            return JudgeResult(score=1)

    judge = SlowJudge()
    result = await run_evaluation([task], tmp_path, model="fixture", rollouts=1, timeout_s=.01,
                                  evidence_kind="fixture", provider=provider, judge=judge)
    assert result["status_counts"] == {"pending": 1}
    judge.ready = True
    resumed = await run_evaluation([task], tmp_path, model="fixture", rollouts=1, timeout_s=.01,
                                   evidence_kind="fixture", provider=provider, judge=judge)
    assert resumed["sample_summary"]["mean"] == 1
    assert provider.calls == 1


async def test_cancelled_http_reservation_is_not_released_free():
    async def handler(request):
        await asyncio.sleep(1)
        return response()
    budget = CostBudget(1)
    provider = AsyncOpenAIProvider(model="fixture", base_url="https://fixture.invalid", api_key="fixture",
                                   budget=budget, pricing=Pricing(1, 1),
                                   transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(.01):
                await provider.chat([{"role": "user", "content": "fixture"}])
        assert budget.spent > 0 and budget.reserved == 0
    finally:
        await provider.aclose()


async def test_timeout_transcript_preserves_conservative_cost(tmp_path):
    task = generate_taskset(n=1, seed=91, tier=1)[0]

    async def handler(request):
        await asyncio.sleep(1)
        return response()

    provider = AsyncOpenAIProvider(model="fixture", base_url="https://fixture.invalid", api_key="fixture",
                                   budget=CostBudget(1), pricing=Pricing(1, 1),
                                   transport=httpx.MockTransport(handler))
    try:
        metrics = await run_evaluation([task], tmp_path, model="fixture", rollouts=1, max_cost=1,
                                        timeout_s=.01, evidence_kind="fixture", provider=provider)
        row = json.loads((tmp_path / "scores.json").read_text())["records"][0]
        assert row["status"] == "timeout" and row["score"] is None
        assert row["cost_usd"] > 0
        assert row["confirmed_cost_usd"] == 0
        assert row["cost_usd"] == pytest.approx(metrics["budget"]["accounted_usd"])
        assert row["attempts"][0]["cost_kind"] == "conservative"
        assert metrics["retries"] == 0
    finally:
        await provider.aclose()


async def test_cost_exhaustion_prevents_paid_dispatch(tmp_path):
    task = generate_taskset(n=1, seed=92, tier=1)[0]
    calls = []

    def handler(request):
        calls.append(request)
        return response()

    provider = AsyncOpenAIProvider(model="fixture", base_url="https://fixture.invalid", api_key="fixture",
                                   budget=CostBudget(0), pricing=Pricing(1, 1),
                                   transport=httpx.MockTransport(handler))
    try:
        metrics = await run_evaluation([task], tmp_path, model="fixture", rollouts=3,
                                        evidence_kind="fixture", provider=provider)
        assert calls == []
        assert metrics["status_counts"] == {"cost_exhausted": 3}
        assert metrics["aggregate"]["mean"] is None
    finally:
        await provider.aclose()


def test_pass_rate_threshold_is_distinct_from_exact_score():
    metrics = summarize_records([{"task_id": "a", "tier": 3, "rollout": 0,
                                  "status": "complete", "score": .975}])
    assert metrics["by_tier"]["3"]["pass_rate"] == 1
    assert metrics["by_tier"]["3"]["exact_score_rate"] == 0


async def test_durable_reservation_and_run_exclusivity(tmp_path):
    task = generate_taskset(n=1, seed=9, tier=1)[0]
    started = asyncio.Event()
    resume = asyncio.Event()

    class PausedProvider(FixtureProvider):
        async def chat(self, messages, **kwargs):
            started.set()
            await resume.wait()
            return await super().chat(messages, **kwargs)
    provider = PausedProvider()
    first = asyncio.create_task(run_evaluation([task], tmp_path, model="fixture", rollouts=1,
                                               evidence_kind="fixture", provider=provider, max_cost=.1))
    await started.wait()
    with pytest.raises(ValueError, match="RUN_ALREADY_ACTIVE"):
        await run_evaluation([task], tmp_path, model="fixture", rollouts=1,
                             evidence_kind="fixture", provider=FixtureProvider(), max_cost=.1)
    resume.set()
    await first
    # An interrupted in-flight reservation is charged on restart, not silently discarded.
    budget_file = tmp_path / "budget.json"
    budget_file.write_text(json.dumps({"accounted_usd": .04, "reserved_usd": .01, "confirmed_usd": .04}))
    metrics = await run_evaluation([task], tmp_path, model="fixture", rollouts=1,
                                   evidence_kind="fixture", provider=FixtureProvider(), max_cost=.1)
    assert metrics["budget"]["reserved_usd"] == 0
    assert metrics["budget"]["accounted_usd"] == pytest.approx(.05)


def test_verifiers_adapter_contract_without_optional_imports(monkeypatch):
    """Contract double only: pinned verifiers runtime smoke test remains Colab-only."""
    import sys
    import types

    class Parser:
        pass
    class Rubric:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    class SingleTurnEnv:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    class ToolEnv(SingleTurnEnv):
        pass
    class Dataset:
        @staticmethod
        def from_list(rows):
            return rows
    class ToolMessage:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    fake_vf = types.ModuleType("verifiers")
    fake_vf.Parser, fake_vf.Rubric = Parser, Rubric
    fake_vf.SingleTurnEnv, fake_vf.ToolEnv = SingleTurnEnv, ToolEnv
    fake_types = types.ModuleType("verifiers.types")
    fake_types.ToolMessage = ToolMessage
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = Dataset
    monkeypatch.setitem(sys.modules, "verifiers", fake_vf)
    monkeypatch.setitem(sys.modules, "verifiers.types", fake_types)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    from aster_gym.environment import load_environment
    single = load_environment(n=1, tier=1)
    assert isinstance(single, SingleTurnEnv)
    assert single.timeout_seconds == 120
    assert "ground_truth" not in json.dumps(single.dataset)
    task = generate_taskset(n=1, seed=91, tier=1)[0]
    tool = load_environment(mode="tool", tasks=[task])
    assert isinstance(tool, ToolEnv)
    assert len(tool.tools) == 3
    state = {"info": {"task_id": task.id}}
    messages = [{"role": "assistant", "tool_calls": [{"id": "call", "function": {
        "name": "read_document", "arguments": json.dumps({"document_id": "request.json"})}}]}]
    outputs = asyncio.run(tool.env_response(messages, state))
    assert outputs[0].role == "tool" and state.get("aster_reference_reads", 0) == 0
    messages[0]["tool_calls"][0]["function"]["arguments"] = json.dumps({"document_id": "unrelated-0.txt"})
    asyncio.run(tool.env_response(messages, state))
    assert state.get("aster_reference_reads", 0) == 0
    messages[0]["tool_calls"][0]["function"]["arguments"] = json.dumps({"document_id": "schedules.json"})
    outputs = asyncio.run(tool.env_response(messages, state))
    assert outputs[0].role == "tool" and state["aster_reference_reads"] == 1
    with pytest.raises(RuntimeError, match="JUDGE_PENDING"):
        trap = generate_taskset(n=1, seed=92, tier=3)[0]
        trap_env = load_environment(tasks=[trap])
        asyncio.run(trap_env.rubric.score_rollout({"info": {"task_id": trap.id},
                    "completion": [{"role": "assistant", "content": trap.ground_truth.model_dump_json()}]}))


async def test_cost_exhaustion_prevents_http_dispatch():
    calls = []
    provider = AsyncOpenAIProvider(model="fixture", base_url="https://fixture.invalid", api_key="fixture",
                                   budget=CostBudget(0), pricing=Pricing(1, 1),
                                   transport=httpx.MockTransport(lambda request: calls.append(request) or response()))
    try:
        with pytest.raises(ProviderError, match="COST_EXHAUSTED"):
            await provider.chat([{"role": "user", "content": "fixture"}])
        assert not calls
    finally:
        await provider.aclose()


async def test_5xx_and_invalid_usage_are_accounted():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, json={"choices": []})
    budget = CostBudget(1)
    provider = AsyncOpenAIProvider(model="fixture", base_url="https://fixture.invalid", api_key="fixture",
                                   budget=budget, pricing=Pricing(1, 1), backoff_s=0,
                                   transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ProviderError, match="INVALID_PROVIDER_RESPONSE") as caught:
            await provider.chat([{"role": "user", "content": "fixture"}])
        assert len(caught.value.attempts) == 2
        assert budget.spent > 0 and budget.reserved == 0
    finally:
        await provider.aclose()


async def test_evaluation_resume_rejects_changed_judge_execution_settings(tmp_path):
    from aster_gym.judge import RemoteJudge

    tasks = generate_taskset(1, seed=12, tier=2)
    first = RemoteJudge(model="fixture", base_url="https://example.invalid", api_key="fixture", max_tokens=512)
    changed = RemoteJudge(model="fixture", base_url="https://example.invalid", api_key="rotated-fixture", max_tokens=1024)
    provider = FixtureProvider()
    await run_evaluation(tasks, tmp_path, model="fixture", rollouts=3, provider=provider,
                         evidence_kind="fixture", judge=first)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["judge"]["max_tokens"] == 512
    with pytest.raises(ValueError, match="RESUME_CONFIGURATION_MISMATCH"):
        await run_evaluation(tasks, tmp_path, model="fixture", rollouts=3, provider=provider,
                             evidence_kind="fixture", judge=changed)
    await first.aclose()
    await changed.aclose()
