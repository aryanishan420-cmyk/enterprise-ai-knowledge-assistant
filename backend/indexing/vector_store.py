"""Persistent FAISS vector store (exact inner-product search over normalised vectors).

Layout in `path`: vectors.npy (float32 matrix), meta.json (chunks + document registry).
The FAISS index is rebuilt in memory on load; writes are atomic (temp file + replace).
"""

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from backend.processing.chunking import Chunk


class DuplicateDocumentError(ValueError):
    pass


class DimensionMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchHit:
    chunk: dict
    score: float


class VectorStore:
    def __init__(self, path: str | Path, dim: int, embedder_name: str = ""):
        self.path, self.dim, self.embedder_name = Path(path), dim, embedder_name
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._chunks: list[dict] = []
        self._docs: dict[str, dict] = {}
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._index = faiss.IndexFlatIP(dim)
        self._load()

    def _load(self) -> None:
        meta_f, vec_f = self.path / "meta.json", self.path / "vectors.npy"
        if not meta_f.exists() or not vec_f.exists():
            return
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        vecs = np.load(vec_f)
        stored = meta.get("embedder", "")
        if stored and self.embedder_name and stored != self.embedder_name:
            raise DimensionMismatchError(
                f"The index was built with embedder '{stored}' but the active embedder is "
                f"'{self.embedder_name}'. Vectors from different models are not comparable: delete "
                f"{self.path} and re-upload your documents, or switch EMBEDDING_* back."
            )
        if vecs.size and vecs.shape[1] != self.dim:
            raise DimensionMismatchError(
                f"Stored index has dim {vecs.shape[1]} (embedder: {meta.get('embedder')}) but "
                f"the active embedder has dim {self.dim}. Re-ingest documents or change EMBEDDING_*."
            )
        self._chunks, self._docs, self._vectors = meta["chunks"], meta["documents"], vecs
        self._rebuild()

    def _rebuild(self) -> None:
        self._index = faiss.IndexFlatIP(self.dim)
        if len(self._vectors):
            self._index.add(self._vectors)

    def _save(self) -> None:
        tmp_v, tmp_m = self.path / "vectors.tmp.npy", self.path / "meta.tmp.json"
        np.save(tmp_v, self._vectors)
        tmp_m.write_text(
            json.dumps({"embedder": self.embedder_name, "chunks": self._chunks, "documents": self._docs}),
            encoding="utf-8",
        )
        os.replace(tmp_v, self.path / "vectors.npy")
        os.replace(tmp_m, self.path / "meta.json")

    def add_document(
        self, document_id: str, filename: str, chunks: list[Chunk], vectors: np.ndarray, owner: str = "default"
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        with self._lock:
            if document_id in self._docs:
                raise DuplicateDocumentError(f"Document already indexed: {self._docs[document_id]['filename']}")
            vecs = vectors.astype(np.float32)
            self._chunks.extend({**c.to_dict(), "owner": owner} for c in chunks)
            self._vectors = np.vstack([self._vectors, vecs])
            self._docs[document_id] = {
                "document_id": document_id,
                "filename": filename,
                "chunk_count": len(chunks),
                "owner": owner,
                "pages": len({c.page for c in chunks}),
                "added_at": time.time(),
            }
            self._index.add(vecs)
            self._save()

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            if document_id not in self._docs:
                return False
            keep = [i for i, c in enumerate(self._chunks) if c["document_id"] != document_id]
            self._chunks = [self._chunks[i] for i in keep]
            self._vectors = self._vectors[keep] if keep else np.zeros((0, self.dim), np.float32)
            del self._docs[document_id]
            self._rebuild()
            self._save()
            return True

    def list_documents(self, owner: str | None = None) -> list[dict]:
        with self._lock:
            docs = [dict(d) for d in self._docs.values() if owner in (None, d["owner"])]
        return sorted(docs, key=lambda d: d["added_at"])

    def get_document(self, document_id: str) -> dict | None:
        with self._lock:
            d = self._docs.get(document_id)
            return dict(d) if d else None

    def search(
        self, query_vec: np.ndarray, top_k: int = 5, threshold: float = 0.0, owner: str | None = None
    ) -> list[SearchHit]:
        with self._lock:
            if not self._chunks:
                return []
            q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
            k = len(self._chunks) if owner else min(top_k, len(self._chunks))
            scores, ids = self._index.search(q, k)
            hits: list[SearchHit] = []
            for s, i in zip(scores[0], ids[0], strict=True):
                if i < 0 or s < threshold:
                    continue
                c = self._chunks[i]
                if owner and c["owner"] != owner:
                    continue
                hits.append(SearchHit(dict(c), float(s)))
                if len(hits) == top_k:
                    break
            return hits

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)
