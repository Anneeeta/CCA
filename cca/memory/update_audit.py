"""Keep pool revisions outside model context, without altering old run artifacts."""
from datetime import datetime, timezone
from pathlib import Path

from ..storage.persistence import append_jsonl


def record_revision(path: Path, before, after, stats: dict) -> None:
    if before == after:
        return
    append_jsonl(path.with_name(path.stem + ".changes.jsonl"), {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "before": before,
        "after": after,
        "stats": stats,
    })
