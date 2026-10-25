from __future__ import annotations

from pathlib import Path
from typing import List
import uuid

from ..schemas import KnowledgeItem, KnowledgePoolUpdate
from ..storage.persistence import read_json_file, write_json_file
from .context_view import content_key, project_lines, ranked_items
from .update_audit import record_revision


class SharedKnowledgePool:
    """
    Persistent storage for shared knowledge K.
    Only Arbiter is allowed to write; Instructor/Expert read.
    """

    def __init__(self, path: Path, *, deduplicate: bool = True):
        self.path = path
        self.deduplicate = deduplicate
        self.last_update_stats: dict = {}

    def load(self) -> List[KnowledgeItem]:
        raw = read_json_file(self.path, default=[])
        active = {}
        for row in raw:
            item = KnowledgeItem.model_validate(row).model_copy(deep=True)
            if not item.id:
                item.id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(self.path.resolve()) + content_key(item.content)))
            prior = active.get(item.id)
            if prior and prior.round_id > item.round_id:
                prior.source_urls = list(dict.fromkeys(prior.source_urls + item.source_urls))
                continue
            duplicate = next((old for old in active.values() if content_key(old.content) == content_key(item.content) and old.id != item.id), None) if self.deduplicate else None
            if duplicate:
                item.source_urls = list(dict.fromkeys(duplicate.source_urls + item.source_urls))
                if prior is not None:
                    del active[duplicate.id]
                else:
                    item.id = duplicate.id
                    prior = duplicate
                    if duplicate.round_id > item.round_id:
                        duplicate.source_urls = item.source_urls
                        continue
                    if item.conflict_tag is None:
                        item.conflict_tag = duplicate.conflict_tag
            if prior is not None:
                item.created_at = prior.created_at or item.created_at
                item.source_urls = list(dict.fromkeys(prior.source_urls + item.source_urls))
            active[item.id] = item
        return list(active.values())

    def save(self, items: List[KnowledgeItem]) -> None:
        data = [item.model_dump() for item in items]
        write_json_file(self.path, data)

    def apply_update(self, update: KnowledgePoolUpdate) -> List[KnowledgeItem]:
        before = read_json_file(self.path, default=[])
        active = {item.id: item for item in self.load()}
        stats = {key: 0 for key in ("added", "updated", "removed", "deduplicated")}
        stats["deduplicated"] = len(before) - len(active)
        for item_id in update.remove_ids:
            if item_id in active:
                del active[item_id]
                stats["removed"] += 1
        for item in update.items:
            incoming = item.model_copy(deep=True)
            prior = active.get(incoming.id) if incoming.id else None
            conflict_was_explicit = prior is not None and "conflict_tag" in item.model_fields_set
            duplicate = next((old for old in active.values() if content_key(old.content) == content_key(incoming.content) and old.id != incoming.id), None) if self.deduplicate else None
            if prior is None and duplicate is not None:
                prior = duplicate
                incoming.id = duplicate.id
                stats["deduplicated"] += 1
            elif prior is not None and duplicate is not None:
                del active[duplicate.id]
                incoming.source_urls = list(dict.fromkeys(duplicate.source_urls + incoming.source_urls))
                stats["deduplicated"] += 1
            if prior is not None:
                incoming.id = prior.id
                incoming.created_at = prior.created_at
                incoming.round_id = max(prior.round_id, incoming.round_id)
                incoming.source_urls = list(dict.fromkeys(prior.source_urls + incoming.source_urls))
                if incoming.conflict_tag is None and not conflict_was_explicit:
                    incoming.conflict_tag = prior.conflict_tag
                stats["updated"] += 1
            else:
                stats["added"] += 1
            incoming.ensure_ids()
            item.id = incoming.id
            active[incoming.id] = incoming
        existing = list(active.values())
        self.save(existing)
        self.last_update_stats = stats
        record_revision(self.path, before, [item.model_dump() for item in existing], stats)
        return existing

    def format_for_context(self, *, query: str = "", max_chars: int | None = None, max_items: int | None = None) -> str:
        items = self.load()
        if not items:
            return "(empty)"
        lines = []
        selected = ranked_items(items, query, lambda item: item.content, max_items)
        for item in selected:
            conflict = ""
            if item.conflict_tag:
                conflict = f" | conflict: {item.conflict_tag.conflict_type} vs {item.conflict_tag.conflict_with} ({item.conflict_tag.note})"
            sources = f" sources: {', '.join(item.source_urls)}" if item.source_urls else ""
            lines.append(f"- [{item.id}] ({item.label}) {item.content}{conflict}{sources}")
        if len(selected) < len(items):
            lines.append(f"({len(items) - len(selected)} other stored entries not shown)")
        return project_lines(lines, max_chars)
