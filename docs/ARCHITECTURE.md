# Architecture

## Components

| Layer | Module | Responsibility |
|---|---|---|
| API | `backend/api/` | FastAPI app factory (`main.create_app`), routes, Pydantic schemas, shared `AppState` |
| Validation | `document_loading/validation.py` | Extension, size, empty file, `%PDF-` signature |
| Extraction | `document_loading/pdf_reader.py` | Per-page text with PyPDF; rejects encrypted/corrupt/no-text PDFs |
| Processing | `processing/text_cleaner.py`, `chunking.py` | Whitespace cleanup; overlapping chunks that never cross a page |
| Embeddings | `embeddings/embedding_service.py` | `SentenceTransformerEmbedder` (semantic) or `HashingEmbedder` (offline, lexical) |
| Index | `indexing/vector_store.py` | Persistent FAISS `IndexFlatIP` over L2-normalised vectors (= cosine similarity) |
| Retrieval | `retrieval/retriever.py` | Query embedding + top-k search + latency; no LLM or web code |
| RAG | `rag/prompts.py`, `chain.py`, `citations.py` | Prompt building, LLM call, citation verification, fallback |
| Persistence | `database/` | SQLAlchemy models (`documents`, `query_logs`) and repository |
| Ingestion | `ingestion.py` | Orchestrates validate -> extract -> chunk -> embed -> index -> record |
| Monitoring | `monitoring/` | In-memory metrics; JSON logging |
| Evaluation | `evaluation/` | Recall@k, MRR, generation metrics, runner |
| UI | `frontend/` | Streamlit pages that call the REST API only |

## Data flow

**Ingestion** (`POST /documents/upload`): validate upload -> extract pages -> fingerprint the cleaned text
(SHA-256, first 16 hex chars = `document_id`) -> reject duplicates -> chunk -> embed all chunks in one batch -> write
to the vector store -> insert the SQL record. If the SQL insert fails, the vector entries are rolled back.

**Question answering** (`POST /chat`): embed the question -> top-k search -> if nothing is retrieved, return the
fallback without calling the LLM -> build a prompt with labelled passages `S1..Sk` -> LLM call -> if the model replies
`INSUFFICIENT_CONTEXT`, return the fallback -> resolve `[S#]` markers to real chunks, drop labels that do not exist ->
return answer, verified citations, all retrieved passages and timings.

## Design decisions and trade-offs

- **Document ID from text, not bytes.** Re-saving a PDF changes its bytes (timestamps, IDs) but not its text; hashing
  text catches those duplicates. Trade-off: two PDFs with identical text but different layout count as duplicates.
- **FAISS flat index persisted as `vectors.npy` + `meta.json`.** Simple, exact, easy to inspect, atomic writes via
  temp file + replace. Deletion rebuilds the in-memory index (O(n)); fine for thousands of chunks, not millions.
  The vector store also records the embedder name and refuses to open an index built with a different model.
- **Chunks never cross pages**, so every citation has an unambiguous page number, at the cost of splitting passages
  that span a page break.
- **Citations are verified, not trusted.** The model is told to cite `[S#]`; the server maps labels to retrieved chunks
  and strips unknown ones. This guarantees cited sources were retrieved; it does not prove the claim is supported by
  the cited text (see faithfulness estimate in [EVALUATION.md](EVALUATION.md)).
- **LLM lazily constructed.** Search, upload and the UI work without an API key; `/chat` returns 503 `not_configured`.
- **No module-level singletons.** `create_app(settings, embedder, llm)` builds everything, so tests inject a hashing
  embedder and a fake LLM.
- **Sync routes.** Blocking work (PDF parsing, embedding, FAISS) runs in FastAPI's threadpool; the store uses a lock.
- **SQLite by default.** Any SQLAlchemy URL can be supplied; PostgreSQL is untested.
- **Single tenant.** Rows carry an `owner` column and the store can filter by owner, but the API does not authenticate
  users, so everything belongs to `default`.

## Failure scenarios

| Scenario | Behaviour |
|---|---|
| Corrupt / encrypted / scanned PDF | 4xx with specific `error.code`; ingestion failure counted; nothing indexed |
| Duplicate upload | 409 `duplicate` |
| Empty index or nothing retrieved | `/search` returns `[]`; `/chat` returns the fallback, no LLM call |
| Missing API key | `/chat` 503 `not_configured`; search-only still works |
| LLM error or timeout | 502 `llm_unavailable` (SDK timeout 30 s, one retry); details never leaked |
| Model invents a citation | Label removed, reported in `invalid_citations_removed` |
| Embedder changed after indexing | Backend refuses to start with an explanatory error |
| SQL write fails after indexing | Vector entries rolled back |
| Process crash during save | Atomic replace leaves the previous files intact |
