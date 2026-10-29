"""Embedding backends. All return L2-normalised float32 matrices (n, dim)."""

import hashlib
import re
from typing import Protocol

import numpy as np

from backend.config import Settings

_TOKEN = re.compile(r"\w+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1, norms)).astype(np.float32)


class HashingEmbedder:
    """Deterministic lexical (bag-of-words hashing) embedder.

    NOT semantic: it matches shared vocabulary only. Provided for offline use,
    unit tests, and environments where model weights cannot be downloaded.
    """

    def __init__(self, dim: int = 384):
        self.dim, self.name = dim, f"hashing-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in _TOKEN.findall(text.lower()):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
                m[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalise(m)


class SentenceTransformerEmbedder:
    """Semantic embeddings via sentence-transformers (model downloaded on first use)."""

    def __init__(self, model_name: str, batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed. Run: pip install -r requirements.txt") from exc
        self._model = SentenceTransformer(model_name)
        self.name, self.batch_size = model_name, batch_size
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False)
        return _normalise(np.asarray(vecs, dtype=np.float32))


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    if settings.embedding_backend == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {settings.embedding_backend}")
