import logging
import threading
from pathlib import Path
from typing import Optional


class _ThreadIdFilter(logging.Filter):
    def __init__(self, thread_id: int) -> None:
        super().__init__()
        self._thread_id = thread_id

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread == self._thread_id


_LOG_CONFIG_LOCK = threading.Lock()


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "cca.log"
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    thread_id = threading.get_ident()
    root = logging.getLogger()

    # Bind one file handler per thread so parallel batch workers do not overwrite
    # each other's log targets.
    with _LOG_CONFIG_LOCK:
        root.setLevel(level)
        for handler in list(root.handlers):
            if (
                getattr(handler, "_cca_thread_handler", False)
                and getattr(handler, "_cca_thread_id", None) == thread_id
            ):
                root.removeHandler(handler)
                handler.close()

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt))
        file_handler.addFilter(_ThreadIdFilter(thread_id))
        file_handler._cca_thread_handler = True  # type: ignore[attr-defined]
        file_handler._cca_thread_id = thread_id  # type: ignore[attr-defined]
        root.addHandler(file_handler)


def redact(text: str) -> str:
    if not text:
        return text
    return text.replace("DEEPSEEK_API_KEY", "****")
