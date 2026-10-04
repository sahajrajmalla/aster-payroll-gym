"""R10 parser: one JSON object, no coercion, duplicate keys, or non-finite tokens."""

import json

from .schemas import Answer


class AnswerParseError(ValueError):
    """A stable public failure, never includes the rejected payload or oracle."""


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def parse_answer(raw: str | dict | Answer) -> Answer:
    try:
        if isinstance(raw, Answer):
            return Answer.model_validate(raw.model_dump())
        if isinstance(raw, str):
            if len(raw.encode()) > 8192:
                raise ValueError("oversized answer")
            raw = json.loads(raw, object_pairs_hook=_pairs,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite")))
        if not isinstance(raw, dict):
            raise ValueError("one object required")
        return Answer.model_validate(raw)
    except (ValueError, TypeError, RecursionError) as exc:
        raise AnswerParseError("INVALID_ANSWER_CONTRACT") from exc
