from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import uuid

from ..schemas import StudentEntry, StudentMsUpdate
from ..storage.persistence import read_json_file, write_json_file
from .context_view import content_key, project_lines, relevance
from .update_audit import record_revision

BUCKETS = ("known", "known_unknown", "conflict", "unknown_unknown")


class StudentPool:
    """
    Student-owned knowledge pool Ms.
    Student may read/write; others cannot access.
    """

    def __init__(self, path: Path, *, deduplicate: bool = True):
        self.path = path
        self.deduplicate = deduplicate
        self.last_update_stats: dict = {}

    def _raw(self):
        return read_json_file(self.path, default={name: [] for name in BUCKETS})

    def load(self) -> Dict[str, List[StudentEntry]]:
        # Reading an old pool provides a deduplicated view, but never rewrites it.
        active: dict[str, StudentEntry] = {}
        raw = self._raw()
        for bucket in BUCKETS:
            for item in raw.get(bucket, []):
                entry = StudentEntry.model_validate(item).model_copy(deep=True)
                entry.type = bucket
                if not entry.id:
                    entry.id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(self.path.resolve()) + bucket + content_key(entry.content)))
                prior = active.get(entry.id)
                if prior and prior.round_id > entry.round_id:
                    continue
                duplicate = next((old for old in active.values() if old.id != entry.id and old.type == bucket and content_key(old.content) == content_key(entry.content)), None) if self.deduplicate else None
                if duplicate:
                    if prior is not None:
                        # A revision of an active id can converge on another
                        # entry. Keep the revised id and remove the duplicate.
                        del active[duplicate.id]
                    else:
                        entry.id = duplicate.id
                        if duplicate.round_id > entry.round_id:
                            continue
                active[entry.id] = entry
        data = {bucket: [] for bucket in BUCKETS}
        for entry in active.values():
            data[entry.type].append(entry)
        return data

    def save(self, data: Dict[str, List[StudentEntry]]) -> None:
        serialised = {bucket: [item.model_dump() for item in items] for bucket, items in data.items()}
        write_json_file(self.path, serialised)

    def apply_update(self, update: StudentMsUpdate) -> Dict[str, List[StudentEntry]]:
        before = self._raw()
        data = self.load()
        active = {entry.id: entry for entries in data.values() for entry in entries}
        stats = {key: 0 for key in ("added", "updated", "moved", "removed", "deduplicated")}
        stats["deduplicated"] = sum(len(before.get(b, [])) for b in BUCKETS) - len(active)
        for entry_id in update.remove_ids:
            if entry_id in active:
                del active[entry_id]
                stats["removed"] += 1
        for bucket_name in BUCKETS:
            entries: List[StudentEntry] = getattr(update, bucket_name)
            for entry in entries:
                incoming = entry.model_copy(deep=True)
                incoming.type = bucket_name
                prior = active.get(incoming.id) if incoming.id else None
                duplicate = next((old for old in active.values() if old.type == bucket_name and content_key(old.content) == content_key(incoming.content) and old.id != incoming.id), None) if self.deduplicate else None
                if prior is None and duplicate is not None:
                    prior = duplicate
                    incoming.id = duplicate.id
                    stats["deduplicated"] += 1
                elif prior is not None and duplicate is not None:
                    del active[duplicate.id]
                    stats["deduplicated"] += 1
                if prior is not None:
                    incoming.id = prior.id
                    incoming.round_id = max(prior.round_id, incoming.round_id)
                    if incoming.rationale is None:
                        incoming.rationale = prior.rationale
                    if incoming.priority is None:
                        incoming.priority = prior.priority
                    if prior.type != incoming.type:
                        stats["moved"] += 1
                    elif prior.model_dump() != incoming.model_dump():
                        stats["updated"] += 1
                else:
                    incoming.ensure_id()
                    stats["added"] += 1
                entry.id = incoming.id  # Retain canonical identity in round records.
                entry.type = bucket_name
                active[incoming.id] = incoming
        data = {bucket: [] for bucket in BUCKETS}
        for entry in active.values():
            data[entry.type].append(entry)
        self.save(data)
        self.last_update_stats = stats
        record_revision(self.path, before, {b: [e.model_dump() for e in items] for b, items in data.items()}, stats)
        return data

    def format_for_context(self, *, query: str = "", max_chars: int | None = None, max_items: int | None = None) -> str:
        data = self.load()
        lines = []
        per_bucket = max(1, max_items // len(BUCKETS)) if max_items is not None else None
        section_budget = max(1, max_chars // len(BUCKETS) - 50) if max_chars is not None else None
        for bucket, entries in data.items():
            lines.append(f"=== {bucket.upper()} ===")
            if not entries:
                lines.append("(empty)")
                continue
            ranked = sorted(enumerate(entries), key=lambda pair: (-relevance(query, pair[1].content), -pair[1].round_id, pair[0]))
            selected = [entry for _, entry in ranked[:per_bucket]]
            section = []
            for item in selected:
                prio = f" priority={item.priority}" if item.priority is not None else ""
                section.append(f"- [{item.id}] {item.content}{prio}")
            if len(selected) < len(entries):
                section.append(f"({len(entries) - len(selected)} other stored entries not shown)")
            lines.append(project_lines(section, section_budget))
        return "\n".join(lines)
