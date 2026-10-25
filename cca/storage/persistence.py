from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List
import uuid
import time


def replace_with_retry(source: Path, target: Path) -> None:
    """Windows readers may briefly deny replacement; never truncate the target."""
    for attempt in range(8):
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (None, 5, 32, 33) or attempt == 7:
                raise
            time.sleep(min(0.025 * 2 ** attempt, 0.4))


def write_text_file(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding=encoding) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    # Only a missing file means an empty initial state. Existing malformed or
    # truncated storage must fail visibly instead of erasing accumulated state.
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")
