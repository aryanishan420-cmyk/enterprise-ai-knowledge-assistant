"""Ingestion service: validate -> extract -> chunk -> embed -> index -> record."""

import hashlib
import logging
import time

from backend.config import Settings
from backend.database.repository import Repository
from backend.document_loading.pdf_reader import extract_pages
from backend.document_loading.validation import PDFValidationError, validate_upload
from backend.embeddings.embedding_service import Embedder
from backend.indexing.vector_store import DuplicateDocumentError, VectorStore
from backend.monitoring.metrics import Metrics
from backend.processing.chunking import chunk_pages
from backend.processing.text_cleaner import clean_text

log = logging.getLogger(__name__)


class IngestionService:
    def __init__(self, settings: Settings, embedder: Embedder, store: VectorStore, repo: Repository, metrics: Metrics):
        self.s, self.embedder, self.store, self.repo, self.metrics = settings, embedder, store, repo, metrics

    def ingest(self, filename: str, data: bytes, owner: str = "default") -> dict:
        t0 = time.perf_counter()
        try:
            up = validate_upload(filename, data, self.s.max_upload_mb * 1024 * 1024)
            pages = extract_pages(data)
            # Hash the cleaned text (not raw bytes) so a re-saved copy of the same PDF is a duplicate.
            fingerprint = "\n".join(clean_text(p.text) for p in pages)
            doc_id = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
            if self.store.get_document(doc_id):
                raise DuplicateDocumentError(f"Document already indexed: {up.filename}")
            chunks = chunk_pages(doc_id, up.filename, pages, self.s.chunk_size, self.s.chunk_overlap)
            if not chunks:
                raise PDFValidationError("No usable text after cleaning.", "no_text")
            vectors = self.embedder.embed([c.text for c in chunks])
            self.store.add_document(doc_id, up.filename, chunks, vectors, owner)
            ms = (time.perf_counter() - t0) * 1000
            try:
                self.repo.add_document(
                    document_id=doc_id,
                    filename=up.filename,
                    owner=owner,
                    size_bytes=up.size_bytes,
                    pages=len(pages),
                    chunk_count=len(chunks),
                    ingestion_ms=ms,
                )
            except Exception:
                self.store.delete_document(doc_id)  # keep vector store and DB consistent
                raise
        except Exception:
            self.metrics.inc("ingestion_failures")
            raise
        self.metrics.observe("ingestion_ms", ms)
        self.metrics.inc("documents_ingested")
        log.info("document_ingested", extra={"fields": {"document_id": doc_id, "chunks": len(chunks)}})
        return {
            "document_id": doc_id,
            "filename": up.filename,
            "pages": len(pages),
            "chunk_count": len(chunks),
            "ingestion_ms": round(ms, 1),
        }

    def delete(self, document_id: str) -> bool:
        found = self.store.delete_document(document_id)
        self.repo.delete_document(document_id)
        return found
