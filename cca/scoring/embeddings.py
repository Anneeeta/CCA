from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """
    Provides local embeddings; falls back to hashing if sentence-transformers is unavailable.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.backend = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.backend = SentenceTransformer(model_name)
            self.mode = "sentence-transformers"
            logger.info("Loaded embedding model: %s", model_name)
        except Exception:
            try:
                from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore

                self.backend = HashingVectorizer(n_features=256, alternate_sign=False, norm=None)
                self.mode = "hashing"
                logger.warning("Falling back to HashingVectorizer embeddings.")
            except Exception as e:  # pragma: no cover - highly unlikely
                logger.error("No embedding backend available: %s", e)
                self.backend = None
                self.mode = "none"

    @property
    def available(self) -> bool:
        return self.backend is not None

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self.available:
            raise RuntimeError("No embedding backend available.")
        if self.mode == "sentence-transformers":
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
