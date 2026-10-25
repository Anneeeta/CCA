from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# Model loading touches process-wide Transformers/PyTorch initialization state.
# Load once, under a lock, rather than constructing one model in each task thread.
_TRANSFORMER_LOAD_LOCK = threading.Lock()
_TRANSFORMER_BACKENDS: dict[str, tuple[object, threading.Lock]] = {}


def _model_cache_key(model_name: str) -> str:
    path = Path(model_name).expanduser()
    if path.is_dir():
        return os.path.normcase(str(path.resolve()))
    return model_name


def _shared_transformer(model_name: str) -> tuple[object, threading.Lock]:
    key = _model_cache_key(model_name)
    with _TRANSFORMER_LOAD_LOCK:
        if key not in _TRANSFORMER_BACKENDS:
            from sentence_transformers import SentenceTransformer  # type: ignore

            backend = SentenceTransformer(model_name)
            _TRANSFORMER_BACKENDS[key] = (backend, threading.Lock())
            logger.info("Loaded embedding model: %s", model_name)
        return _TRANSFORMER_BACKENDS[key]


class EmbeddingProvider:
    """
    Provides local embeddings; falls back to hashing if sentence-transformers is unavailable.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.backend = None
        self._encode_lock = None
        self.initialization_error_type = None
        try:
            self.backend, self._encode_lock = _shared_transformer(model_name)
            self.mode = "sentence-transformers"
        except Exception as exc:
            self.initialization_error_type = type(exc).__name__
            logger.warning("Sentence-transformer initialization failed: %s", self.initialization_error_type)
            try:
                from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore

                self.backend = HashingVectorizer(n_features=256, alternate_sign=False, norm=None)
                self.mode = "hashing"
                logger.warning("Falling back to HashingVectorizer embeddings.")
            except Exception as e:  # pragma: no cover - highly unlikely
                logger.error("No embedding backend available: %s", type(e).__name__)
                self.backend = None
                self.mode = "none"

    @property
    def available(self) -> bool:
        return self.backend is not None

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self.available:
            raise RuntimeError("No embedding backend available.")
        if self.mode == "sentence-transformers":
            # encode() changes inference/device state internally. Serialize calls
            # on the shared model while model API requests remain concurrent.
            with self._encode_lock:
                return np.array(
                    self.backend.encode(
                        texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                )
        if self.mode == "hashing":
            # HashingVectorizer expects iterable of texts
            mat = self.backend.transform(texts)
            return mat.toarray()
        raise RuntimeError("Unknown embedding backend.")

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
