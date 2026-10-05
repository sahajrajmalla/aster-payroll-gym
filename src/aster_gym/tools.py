"""Bounded, deterministic tools. They expose source evidence, never the oracle."""
from __future__ import annotations

import ast
import json
from datetime import date
from decimal import Decimal, DecimalException, localcontext
from typing import TYPE_CHECKING, Any

from aster_gym.versions import rules_text

if TYPE_CHECKING:
    from aster_gym.schemas import Task

TOOL_DEFINITIONS = [
    {"type": "function", "function": {"name": name, "description": description,
     "parameters": {"type": "object", "properties": {argument: {"type": "string"}},
                    "required": [argument], "additionalProperties": False}}}
    for name, argument, description in [
        ("read_document", "document_id", "Read one source document identified in the task. No answer keys."),
        ("lookup_rules", "pay_date", "Read the authoritative payroll clauses for an ISO payment date."),
        ("calculate", "expression", "Calculate bounded decimal arithmetic using +, -, *, / and parentheses."),
    ]
]


def is_reference_read(name: str, arguments: Any, response: dict[str, Any]) -> bool:
    """Count source evidence, excluding request echoes and irrelevant planning notes."""
    if "error_code" in response or not isinstance(arguments, dict):
        return False
    if name == "lookup_rules":
        return (response.get("pay_date") == arguments.get("pay_date")
                and isinstance(response.get("rules"), str) and bool(response["rules"].strip()))
    if name != "read_document":
        return False
    document_id = arguments.get("document_id")
    references = {"rules.md", "schedules.json", "payroll-records.json"} | {
        f"record-{field}.json" for field in ("salary_records", "period_days", "paid_days",
                                            "bonus_cents", "ytd_pensionable_cents", "ytd_year")
    }
    return (isinstance(document_id, str) and document_id in references
            and response.get("document_id") == document_id
            and isinstance(response.get("content"), str) and bool(response["content"].strip()))


def calculate(expression: str) -> dict[str, str]:
    """Evaluate only arithmetic AST nodes; never execute Python or resolve names."""
    try:
        if not isinstance(expression, str) or not 1 <= len(expression) <= 256:
            raise ValueError
        tree = ast.parse(expression, mode="eval")
        if len(list(ast.walk(tree))) > 64:
            raise ValueError

        def visit(node: ast.AST) -> Decimal:
            if isinstance(node, ast.Expression):
                value = visit(node.body)
            elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
                literal = ast.get_source_segment(expression, node) or ""
                if len(literal) > 24 or not all(c in "0123456789.eE+-" for c in literal):
                    raise ValueError
                value = Decimal(literal)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = visit(node.operand)
                value = -value if isinstance(node.op, ast.USub) else value
            elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add):
                    value = left + right
                elif isinstance(node.op, ast.Sub):
                    value = left - right
                elif isinstance(node.op, ast.Mult):
                    value = left * right
                else:
                    value = left / right
            else:
                raise ValueError
            if not value.is_finite() or abs(value) > Decimal("1e15"):
                raise ValueError
            return value

        with localcontext() as context:
            context.prec = 28
            context.Emax = 100
            context.Emin = -100
            value = visit(tree)
            rendered = format(value.normalize(), "f")
            if len(rendered) > 128:
                raise ValueError
            return {"value": rendered, "precision": "28 decimal digits; no payroll rounding applied"}
    except (ValueError, SyntaxError, DecimalException, OverflowError, RecursionError):
        return {"error_code": "INVALID_EXPRESSION"}


class ToolSession:
    """One session per rollout; documents cannot bleed into another task."""

    def __init__(self, task: Task, max_calls: int = 24):
        self._documents = dict(task.context_files)
        self.max_calls = max_calls
        self.calls = 0

    def call(self, name: str, arguments: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls > self.max_calls:
            return {"error_code": "TOOL_LIMIT"}
        if not isinstance(arguments, dict):
            return {"error_code": "INVALID_ARGUMENTS"}
        expected = {"read_document": "document_id", "lookup_rules": "pay_date", "calculate": "expression"}
        key = expected.get(name)
        if key is None:
            return {"error_code": "UNKNOWN_TOOL"}
        if set(arguments) != {key} or not isinstance(arguments[key], str):
            return {"error_code": "INVALID_ARGUMENTS"}
        arg = arguments[key]
        if len(arg) > 256:
            return {"error_code": "INVALID_ARGUMENTS"}
        if name == "calculate":
            return calculate(arg)
        if name == "read_document":
            content = self._documents.get(arg)
            if content is None:
                return {"error_code": "DOCUMENT_NOT_FOUND"}
            if len(content.encode()) > 32_768:
                return {"error_code": "DOCUMENT_TOO_LARGE"}
            return {"document_id": arg, "content": content}
        try:
            parsed = date.fromisoformat(arg)
            if parsed.isoformat() != arg:
                raise ValueError
        except ValueError:
            return {"error_code": "INVALID_DATE"}
        # Return written authority. Schedule selection remains the agent's job.
        return {"pay_date": arg, "rules": rules_text()}


def parse_tool_arguments(raw: str) -> dict[str, Any] | None:
    """Reject duplicate keys, non-objects, and oversized arguments."""
    if not isinstance(raw, str) or len(raw.encode()) > 2048:
        return None

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        return value if isinstance(value, dict) else None
    except (ValueError, RecursionError):
        return None
