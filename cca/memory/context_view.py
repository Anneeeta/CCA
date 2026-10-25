"""Deterministic context projections; no embeddings, model calls, or storage writes."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence


def content_key(text: str) -> str:
    # Preserve case, punctuation, numbers, negation and units. Do not merge by
    # semantic similarity or lowercase case-sensitive scientific identifiers.
    return " ".join(unicodedata.normalize("NFC", text).split())


def relevance(query: str, text: str) -> int:
    words = set(re.findall(r"\w+", query.casefold()))
    return len(words & set(re.findall(r"\w+", text.casefold())))


def excerpt(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    marker = " [excerpt; full text retained]"
    if max_chars <= len(marker):
        return marker[:max_chars]
    prefix = text[:max_chars - len(marker)]
    boundary = prefix.rfind(" ")
    if boundary > len(prefix) // 2:
        prefix = prefix[:boundary]
    return prefix.rstrip() + marker


def project_lines(lines: Sequence[str], max_chars: int | None) -> str:
    """Keep complete entries where possible and visibly mark omitted material."""
    if max_chars is None:
        return "\n".join(lines)
    complete = "\n".join(lines)
    if len(complete) <= max_chars:
        return complete
    marker = "[Additional entries omitted from context; full records retained.]"
    output: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) + (1 if output else 0) <= max_chars - len(marker) - 1:
            output.append(line)
            used += len(line) + (1 if len(output) > 1 else 0)
        else:
            if not output:
                return excerpt(line, max_chars)
            output.append(marker)
            break
    return "\n".join(output)


def ranked_items(items: Sequence, query: str, render: Callable, max_items: int | None):
    indexed = list(enumerate(items))
    if query:
        indexed.sort(key=lambda pair: (-relevance(query, render(pair[1])), pair[0]))
    if max_items is not None:
        indexed = indexed[:max_items]
    return [item for _, item in indexed]
