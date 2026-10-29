# API reference

Base URL: `http://127.0.0.1:8000`. Interactive docs: `/docs` (Swagger) and `/redoc`. All errors use:

```json
{"error": {"code": "validation_error", "message": "Invalid request (query): String should have at least 1 character"}}
```

If `ADMIN_TOKEN` is set, `POST /documents/upload`, `DELETE /documents/{id}` and `GET /metrics` require the header
`X-API-Token: <token>` (401 `unauthorized` otherwise). With no token configured they are open.

| Method | Path | Purpose | Success |
|---|---|---|---|
| GET | `/health` | Status, index size, embedder, auth/LLM flags | 200 |
| POST | `/documents/upload` | Multipart `files` (one or more PDFs) | 201 |
| GET | `/documents` | List indexed documents | 200 |
| DELETE | `/documents/{document_id}` | Remove vectors and database record | 200 |
| POST | `/search` | Retrieve passages, no LLM | 200 |
| POST | `/chat` | RAG answer with citations | 200 |
| GET | `/metrics` | Operational metrics | 200 |
| GET | `/evaluation` | Last saved evaluation results | 200 / 404 |

## Examples (captured from a real run; passage text shortened; `/chat` used a scripted test LLM)

`GET /health`
```json
{"status": "ok", "indexed_chunks": 0, "embedder": "hashing-384", "auth_enabled": false, "llm_configured": true}
```

`POST /documents/upload` (201)
```json
{"results": [{"filename": "northwind_policies.pdf", "status": "indexed", "document_id": "00c8bc7908138bc2",
              "pages": 6, "chunk_count": 6, "ingestion_ms": 12.8, "error_code": null, "error": null}]}
```
If every file fails, the HTTP status reflects the first error and the body carries both `error` and `results`.

`POST /search` with `{"query": "How many vacation days?", "top_k": 2}`
```json
{"results": [{"label": "S1", "score": 0.3638, "document_id": "00c8bc7908138bc2", "filename": "northwind_policies.pdf",
              "page": 1, "chunk_id": "00c8bc7908138bc2:p1:c0", "text": "..."}], "retrieval_ms": 0.24}
```
`top_k` (1-50) is optional; `query` is 1-2000 characters.

`POST /chat` with `{"query": "How many vacation days?", "top_k": 2, "include_passages": true}`
```json
{"answer": "Full-time employees receive 22 paid vacation days per year [S1].",
 "insufficient_context": false,
 "citations": [{"label": "S1", "document_id": "00c8bc7908138bc2", "filename": "northwind_policies.pdf",
                "page": 1, "chunk_id": "00c8bc7908138bc2:p1:c0", "score": 0.3638}],
 "invalid_citations_removed": [], "retrieved": [ "...same shape as /search results..." ],
 "retrieval_ms": 0.45, "llm_ms": 0.02, "total_ms": 0.63, "prompt_tokens": 10, "completion_tokens": 5}
```
When the documents cannot answer, `insufficient_context` is `true`, `answer` is a fixed fallback message and
`citations` is empty.

`GET /metrics` returns `counters` (e.g. `requests_total`, `requests_failed`, `documents_ingested`,
`ingestion_failures`, `endpoint:/chat`, `prompt_tokens`), `latency_ms` (`count/mean/p50/p95/max` for `response_ms`,
`retrieval_ms`, `llm_ms`, `ingestion_ms`), `index` (`documents`, `chunks`) and `queries_logged`.

## Error codes

| HTTP | `error.code` | Cause |
|---|---|---|
| 401 | `unauthorized` | Token required and missing/wrong |
| 404 | `not_found` / `no_evaluation` | Unknown document / no evaluation saved yet |
| 409 | `duplicate` | Same document text already indexed |
| 413 | `file_too_large` | Exceeds `MAX_UPLOAD_MB` |
| 415 | `unsupported_type` | Not a `.pdf` |
| 422 | `validation_error` | Bad request body |
| 422 | `empty_file`, `bad_signature`, `corrupted`, `encrypted`, `no_pages`, `no_text` | Unusable PDF (`no_text` = likely scanned; OCR is not implemented) |
| 500 | `index_mismatch`, `internal_error` | Server-side problem (details are not exposed) |
| 502 | `llm_unavailable` | LLM provider failed or timed out |
| 503 | `not_configured` | `OPENAI_API_KEY` missing (`/chat` only) |
