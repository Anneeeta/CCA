from __future__ import annotations

import json
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def extract_json_text(text: str | None) -> str:
    """
    Extract JSON substring from text if wrapped with extra prose.
    """
    if not isinstance(text, str):
        return ""
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    return text


def _sanitize_unescaped_quotes(text: str) -> str:
    """
    Escape unescaped double quotes inside JSON string values.

    Handles cases like: "use of terminology (e.g., "ramp forecasting," "CRPS")"
    where inner quotes should be escaped.
    """
    out = []
    in_string = False
    escaped = False
    JSON_STRUCTURAL = set(",}] :")
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                # A structural character after optional whitespace closes the string.
                # Otherwise this quote is an unescaped quote within the value.
                j = i + 1
                while j < len(text) and text[j] in ' \t\n\r':
                    j += 1
                next_ch = text[j] if j < len(text) else ""
                if next_ch in JSON_STRUCTURAL or next_ch == "":
                    out.append(ch)
                    in_string = False
                else:
                    # Escape the embedded quote.
                    out.append('\\"')
                continue
            out.append(ch)
        else:
            if ch == '"':
                out.append(ch)
                in_string = True
                continue
            out.append(ch)
    return "".join(out)


def _sanitize_unescaped_newlines(text: str) -> str:
    """
    Escape raw control characters inside JSON strings (e.g., newlines).
    """
    out = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
            continue
        else:
            if ch == '"':
                out.append(ch)
                in_string = True
                continue
            out.append(ch)
    return "".join(out)


def _repair_json_chain(text: str) -> str:
    """Apply all sanitizers in order."""
    result = _sanitize_unescaped_newlines(text)
    result = _sanitize_unescaped_quotes(result)
    return result


def parse_with_schema(text: str | None, schema: Type[T]) -> T:
    cleaned = extract_json_text(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        repaired = _repair_json_chain(cleaned)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError as e2:
            raise ValueError(f"JSON decode failed: {e2}") from e2
    try:
        return schema.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e}") from e


def load_json_lenient(text: str | None) -> dict | None:
    """
    Best-effort JSON loader for fallbacks.
    """
    cleaned = extract_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = _repair_json_chain(cleaned)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None
