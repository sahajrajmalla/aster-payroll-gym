"""Small remote-only chat client with explicit prices and concurrent reservations."""
from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

import httpx


class ProviderError(RuntimeError):
    def __init__(self, code: str, attempts: list[dict[str, Any]] | None = None):
        super().__init__(code)
        self.code = code
        self.attempts = attempts or []


@dataclass(frozen=True)
class Pricing:
    input_per_million: float
    output_per_million: float
    verified_free: bool = False

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or v < 0 for v in (self.input_per_million, self.output_per_million)):
            raise ValueError("INVALID_PRICING")
        if self.input_per_million == self.output_per_million == 0 and not self.verified_free:
            raise ValueError("ZERO_PRICING_REQUIRES_VERIFIED_FREE_TIER")

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.input_per_million + completion_tokens * self.output_per_million) / 1_000_000


class CostBudget:
    """Reserve worst-case call costs before dispatch; ambiguous calls consume reservation."""

    def __init__(self, maximum: float, spent: float = 0.0):
        if not math.isfinite(maximum) or maximum < 0 or not 0 <= spent <= maximum + 1e-12:
            raise ValueError("INVALID_BUDGET")
        self.maximum = maximum
        self.spent = spent
        self.actual_spend = 0.0
        self.reserved = 0.0
        self._lock = asyncio.Lock()
        self.on_change: Callable[[dict[str, float]], None] | None = None

    async def reserve(self, amount: float) -> float:
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("INVALID_RESERVATION")
        async with self._lock:
            if self.spent + self.reserved + amount > self.maximum + 1e-12:
                raise ProviderError("COST_EXHAUSTED")
            self.reserved += amount
            if self.on_change:
                self.on_change(self.snapshot())
        return amount

    async def settle(self, reservation: float, actual: float | None = None) -> None:
        async with self._lock:
            self.reserved = max(0.0, self.reserved - reservation)
            self.spent += reservation if actual is None else actual
            if actual is not None:
                self.actual_spend += actual
            if self.on_change:
                self.on_change(self.snapshot())
            if actual is not None and actual > reservation + 1e-12:
                # Provider violated a documented bound; stop future calls, never hide it.
                self.spent = max(self.spent, self.maximum)
                raise ProviderError("PROVIDER_USAGE_EXCEEDED_RESERVATION")

    def snapshot(self) -> dict[str, float]:
        return {"maximum_usd": self.maximum, "accounted_usd": self.spent,
                "confirmed_usd": self.actual_spend, "reserved_usd": self.reserved}


@dataclass
class ChatResult:
    message: dict[str, Any]
    attempts: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0


def resolve_reasoning_effort(model: str, base_url: str, requested: str | None = None) -> str | None:
    """Use a documented Google default; leave unrelated providers unmodified.

    Other providers receive this field only when explicitly configured. Support
    there is account/model specific and any rejection remains an operational error.
    """
    if requested is not None and (type(requested) is not str
                                  or requested not in {"none", "minimal", "low", "medium", "high"}):
        raise ValueError("INVALID_REASONING_EFFORT")
    parsed = urlparse(base_url)
    google = (parsed.hostname == "generativelanguage.googleapis.com"
              and parsed.path.rstrip("/") == "/v1beta/openai" and model.startswith("gemini-"))
    if requested is None:
        return "low" if google else None
    if google and requested == "none" and model.startswith(("gemini-3", "gemini-2.5-pro")):
        raise ValueError("UNSUPPORTED_GEMINI_REASONING_EFFORT")
    if google and requested == "minimal" and model == "gemini-3.8-flash":
        raise ValueError("UNSUPPORTED_GEMINI_REASONING_EFFORT")
    return requested


class AsyncOpenAIProvider:
    """OpenAI-compatible HTTPS client. This class never loads local model weights."""

    def __init__(self, *, model: str, base_url: str, api_key: str, budget: CostBudget,
                 pricing: Pricing | None, timeout_s: float = 120, max_retries: int = 2,
                 backoff_s: float = 1, transport: httpx.AsyncBaseTransport | None = None,
                 reasoning_effort: str | None = None):
        if pricing is None:
            raise ValueError("UNKNOWN_PRICING")
        parsed = urlparse(base_url)
        if transport is None:
            hostname = (parsed.hostname or "").lower().rstrip(".")
            try:
                address = ipaddress.ip_address(hostname)
                local_address = not address.is_global
            except ValueError:
                local_address = hostname == "localhost" or hostname.endswith(".localhost")
            if (parsed.scheme != "https" or not hostname or local_address
                    or parsed.username is not None or parsed.password is not None):
                raise ValueError("REMOTE_HTTPS_ENDPOINT_REQUIRED")
        if (not 0 <= max_retries <= 5 or not math.isfinite(timeout_s) or timeout_s <= 0
                or not math.isfinite(backoff_s) or backoff_s < 0):
            raise ValueError("INVALID_PROVIDER_LIMITS")
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key
        self.reasoning_effort = resolve_reasoning_effort(model, base_url, reasoning_effort)
        self.budget, self.pricing = budget, pricing
        self.timeout_s, self.max_retries, self.backoff_s = timeout_s, max_retries, backoff_s
        self._client = httpx.AsyncClient(timeout=timeout_s, transport=transport, follow_redirects=False)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
                   max_tokens: int = 512, temperature: float = 0.7) -> ChatResult:
        if (type(max_tokens) is not int or not 1 <= max_tokens <= 4096
                or not math.isfinite(temperature) or not 0 <= temperature <= 2):
            raise ValueError("INVALID_TOKEN_LIMIT")
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                   "temperature": temperature}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if tools:
            payload["tools"] = tools
        # UTF-8 bytes + per-message overhead upper-bounds ordinary text BPE tokenization.
        # No image/audio content is supported. Output is bounded by max_tokens.
        prompt_bound = len(json.dumps(payload, ensure_ascii=False).encode()) + 256 * (len(messages) + 1)
        reservation_cost = self.pricing.cost(prompt_bound, max_tokens)
        attempts: list[dict[str, Any]] = []
        start = time.monotonic()
        total_cost = 0.0
        for attempt in range(self.max_retries + 1):
            try:
                reservation = await self.budget.reserve(reservation_cost)
            except ProviderError as exc:
                exc.attempts = attempts
                raise
            began = time.monotonic()
            try:
                response = await self._client.post(self.base_url + "/chat/completions", json=payload,
                                                   headers={"Authorization": "Bearer " + self.api_key})
            except asyncio.CancelledError as exc:
                await self.budget.settle(reservation)
                attempts.append({"attempt": attempt + 1, "status": "PROVIDER_CANCELLED",
                                 "cost_usd": reservation, "cost_kind": "conservative",
                                 "latency_s": time.monotonic() - began})
                # asyncio.timeout preserves this as TimeoutError.__cause__; retain
                # accounting without mistaking cancellation for a free request.
                exc.attempts = attempts  # type: ignore[attr-defined]
                raise
            except httpx.TransportError as exc:
                await self.budget.settle(reservation)
                total_cost += reservation
                code = "PROVIDER_TIMEOUT" if isinstance(exc, httpx.TimeoutException) else "PROVIDER_NETWORK"
                attempts.append({"attempt": attempt + 1, "status": code, "cost_usd": reservation,
                                 "cost_kind": "conservative", "latency_s": time.monotonic() - began})
                if attempt == self.max_retries:
                    raise ProviderError(code, attempts) from None
                await asyncio.sleep(min(8.0, self.backoff_s * 2 ** attempt))
                continue
            retryable = response.status_code == 429 or response.status_code >= 500
            if response.is_error or not 200 <= response.status_code < 300:
                # 4xx is rejected/unbilled; uncertain 5xx consumes the reserved maximum.
                uncertain = response.status_code >= 500
                await self.budget.settle(reservation, None if uncertain else 0.0)
                cost = reservation if uncertain else 0.0
                total_cost += cost
                code = "PROVIDER_RATE_LIMIT" if response.status_code == 429 else "PROVIDER_HTTP_ERROR"
                attempts.append({"attempt": attempt + 1, "status": code, "http_status": response.status_code,
                                 "cost_usd": cost, "cost_kind": "conservative" if uncertain else "confirmed",
                                 "latency_s": time.monotonic() - began})
                if not retryable or attempt == self.max_retries:
                    raise ProviderError(code, attempts)
                await asyncio.sleep(min(8.0, self.backoff_s * 2 ** attempt))
                continue
            try:
                if len(response.content) > 1_000_000:
                    raise ValueError
                body = response.json()
                usage = body["usage"]
                prompt_tokens, output_tokens = usage["prompt_tokens"], usage["completion_tokens"]
                if any(type(v) is not int or v < 0 for v in (prompt_tokens, output_tokens)):
                    raise ValueError
                incoming = body["choices"][0]["message"]
                if not isinstance(incoming, dict) or incoming.get("role") != "assistant":
                    raise ValueError
                message: dict[str, Any] = {"role": "assistant", "content": incoming.get("content") or ""}
                if not isinstance(message["content"], str):
                    raise ValueError
                if incoming.get("tool_calls"):
                    calls = incoming["tool_calls"]
                    if not isinstance(calls, list) or len(calls) > 8:
                        raise ValueError
                    if any(not isinstance(call, dict) or not isinstance(call.get("id"), str)
                           or not isinstance(call.get("function"), dict)
                           or not isinstance(call["function"].get("name"), str)
                           or not isinstance(call["function"].get("arguments"), str)
                           for call in calls):
                        raise ValueError
                    message["tool_calls"] = calls
            except (ValueError, KeyError, IndexError, TypeError):
                await self.budget.settle(reservation)
                attempts.append({"attempt": attempt + 1, "status": "INVALID_PROVIDER_RESPONSE",
                                 "cost_usd": reservation, "cost_kind": "conservative"})
                raise ProviderError("INVALID_PROVIDER_RESPONSE", attempts) from None
            actual = self.pricing.cost(prompt_tokens, output_tokens)
            attempts.append({"attempt": attempt + 1, "status": "complete", "cost_usd": actual,
                             "cost_kind": "confirmed", "latency_s": time.monotonic() - began,
                             "prompt_tokens": prompt_tokens, "completion_tokens": output_tokens})
            try:
                await self.budget.settle(reservation, actual)
            except ProviderError as exc:
                exc.attempts = attempts
                raise
            total_cost += actual
            return ChatResult(message, attempts, prompt_tokens, output_tokens, total_cost, time.monotonic() - start)
        raise ProviderError("PROVIDER_RETRIES_EXHAUSTED", attempts)
