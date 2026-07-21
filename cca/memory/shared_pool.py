from __future__ import annotations

from pathlib import Path
from typing import List

from ..schemas import KnowledgeItem, KnowledgePoolUpdate
from ..storage.persistence import read_json_file, write_json_file


class SharedKnowledgePool:
    """
    Persistent storage for shared knowledge K.
    Only Arbiter is allowed to write; Instructor/Expert read.
    """

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> List[KnowledgeItem]:
        raw = read_json_file(self.path, default=[])
        return [KnowledgeItem.model_validate(item) for item in raw]

    def save(self, items: List[KnowledgeItem]) -> None:
        data = [item.model_dump() for item in items]
        write_json_file(self.path, data)

    def apply_update(self, update: KnowledgePoolUpdate) -> List[KnowledgeItem]:
        existing = self.load()
        existing_ids = {item.id for item in existing if item.id}
        for item in update.items:
            item.ensure_ids()
            if item.id in existing_ids:
                # update existing entry
                for idx, ex in enumerate(existing):
                    if ex.id == item.id:
                        existing[idx] = item
                        break
            else:
                existing.append(item)
        self.save(existing)
        return existing

    def format_for_context(self) -> str:
        items = self.load()
        if not items:
            return "(empty)"
        lines = []
        for item in items:
            conflict = ""
            if item.conflict_tag:
                conflict = f" | conflict: {item.conflict_tag.conflict_type} vs {item.conflict_tag.conflict_with} ({item.conflict_tag.note})"
            lines.append(
                f"- [{item.id}] ({item.label}, {item.source_role}, round {item.round_id}) {item.content}{conflict}"
            )
        return "\n".join(lines)

