from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..schemas import StudentEntry, StudentMsUpdate
from ..storage.persistence import read_json_file, write_json_file


class StudentPool:
    """
    Student-owned knowledge pool Ms.
    Student may read/write; others cannot access.
    """

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> Dict[str, List[StudentEntry]]:
        raw = read_json_file(
            self.path,
            default={
                "known": [],
                "known_unknown": [],
                "conflict": [],
                "unknown_unknown": [],
            },
        )
        return {bucket: [StudentEntry.model_validate(item) for item in items] for bucket, items in raw.items()}

    def save(self, data: Dict[str, List[StudentEntry]]) -> None:
        serialised = {bucket: [item.model_dump() for item in items] for bucket, items in data.items()}
        write_json_file(self.path, serialised)

    def apply_update(self, update: StudentMsUpdate) -> Dict[str, List[StudentEntry]]:
        data = self.load()
        for bucket_name in ["known", "known_unknown", "conflict", "unknown_unknown"]:
            entries: List[StudentEntry] = getattr(update, bucket_name)
            for entry in entries:
                entry.ensure_id()
                data[bucket_name].append(entry)
        self.save(data)
        return data

    def format_for_context(self) -> str:
        data = self.load()
        lines = []
        for bucket, entries in data.items():
            lines.append(f"=== {bucket.upper()} ===")
            if not entries:
                lines.append("(empty)")
                continue
            for item in entries:
                prio = f" priority={item.priority}" if item.priority is not None else ""
                rationale = f" | rationale: {item.rationale}" if item.rationale else ""
                lines.append(f"- [{item.id}] {item.content}{prio}{rationale}")
        return "\n".join(lines)

