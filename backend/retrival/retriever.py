"""Semantic retrieval, independent of the LLM and the web layer."""

import time
from dataclasses import dataclass

from backend.embeddings.embedding_service import Embedder
from backend.indexing.vector_store import SearchHit, VectorStore


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[SearchHit]
    latency_ms: float


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, top_k: int = 5, threshold: float = 0.0):
        self.store, self.embedder = store, embedder
        self.top_k, self.threshold = top_k, threshold

    def search(self, query: str, top_k: int | None = None, owner: str | None = None) -> RetrievalResult:
        t0 = time.perf_counter()
        vec = self.embedder.embed([query])[0]
        hits = self.store.search(vec, top_k or self.top_k, self.threshold, owner)
        return RetrievalResult(hits, (time.perf_counter() - t0) * 1000)
