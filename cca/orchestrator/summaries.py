"""Round notes returned by the existing Arbiter call, with an offline fallback."""
from __future__ import annotations

import re

from ..memory.context_view import content_key, excerpt, project_lines
from ..schemas import RoundSummary


def source_urls(text: str) -> list[str]:
    return list(dict.fromkeys(url.rstrip(".,;:)]}") for url in re.findall(r"https?://[^\s<>\"']+", text)))


def prepare_summary(proposed: RoundSummary | None, *, round_id: int, question: str,
                    answer: str, expert: str | None = None, guidance: str | None = None,
                    evidence_refs: list[str] | None = None, max_chars: int = 2200) -> RoundSummary:
    allowed = list(dict.fromkeys(evidence_refs or []))
    if proposed is None or not any((proposed.key_points, proposed.unresolved_questions, proposed.conflicts, proposed.resolved_questions)):
        points = []
        for text in (answer, expert or ""):
            if text and text not in points:
                points.append(excerpt(text, max(180, max_chars // 3)))
        return RoundSummary(round_id=round_id, questions=[question], key_points=points,
                            evidence_refs=allowed, guidance=guidance,
                            origin="extractive_fallback")
    summary = proposed.model_copy(deep=True)
    summary.round_id = round_id
    summary.questions = [question]  # Questions and user intent come from the run, not a rewrite.
    summary.guidance = guidance
    summary.origin = "arbiter"
    summary.evidence_refs = [url for url in summary.evidence_refs if url in allowed]
    for field in ("key_points", "unresolved_questions", "conflicts", "resolved_questions"):
        cleaned = []
        for value in getattr(summary, field):
            text = content_key(value)
            for url in source_urls(text):
                if url not in allowed:
                    text = text.replace(url, "[unverified reference omitted]")
            if text and text not in cleaned:
                cleaned.append(text)
        setattr(summary, field, cleaned)
    return summary


def render_summary(summary: RoundSummary, max_chars: int) -> str:
    lines = [f"Round {summary.round_id} ({summary.origin}):"]
    lines.extend("Q: " + q for q in summary.questions)
    if summary.guidance:
        lines.append("User guidance: " + summary.guidance)
    # Put unresolved/contradictory information before longer explanatory notes.
    for label, values in (
        ("Unresolved", summary.unresolved_questions),
        ("Conflict", summary.conflicts),
        ("Resolved/revised", summary.resolved_questions),
    ):
        lines.extend(label + ": " + text for text in values)
    lines.extend("Source: " + url for url in summary.evidence_refs)
    lines.extend("Finding: " + text for text in summary.key_points)
    return project_lines(lines, max_chars)
