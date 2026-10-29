# Security

## Implemented

| Control | Where |
|---|---|
| Secrets only from environment / `.env`; `.env` git-ignored; `.env.example` has placeholders | `config.py`, `.gitignore` |
| Settings shown in the UI exclude credentials | `Settings.public_dict()` |
| Upload validation: extension, size cap (reads at most limit+1 bytes), `%PDF-` signature, encrypted/corrupt rejection; path components stripped from filenames | `validation.py`, `pdf_reader.py`, `routes.py` |
| Request validation (Pydantic): query length, `top_k` bounds | `schemas.py` |
| Generic error messages; unhandled exceptions return `internal_error`, only the exception type is logged | `main.py` |
| Optional token auth (`ADMIN_TOKEN`, constant-time compare) on upload, delete, metrics | `routes.py` |
| LLM request timeout (30 s) and a single retry | `chain.py` |
| No raw questions or document text in logs; query log stores SHA-256 + length | `repository.py`, `logging_config.py` |
| Prompt-injection mitigation: passages delimited, delimiter tokens neutralised, system prompt declares them untrusted and forbids revealing configuration | `prompts.py` |
| Citation verification: only labels that map to retrieved chunks survive | `citations.py` |
| Container runs as non-root user | `Dockerfile` |

## Not implemented (recommended for a real enterprise deployment)

- User authentication, per-user/per-team document access control and audit logs. The vector store supports an `owner`
  filter, but the API does not assign owners, so all documents are shared.
- TLS termination, rate limiting, request-size limits at a gateway, CORS policy review.
- Malware scanning of uploads; sandboxed PDF parsing.
- Secret manager integration, key rotation, encryption of the index and database at rest.
- Persistent, tamper-evident audit logging; PII detection/redaction.
- Output moderation and stronger injection defences (instruction hierarchy models, content filters).

## Prompt injection: honest limits

Mitigations lower the risk that text inside a PDF redirects the model, but they cannot guarantee it. The tests check
that a passage cannot close the context block and that the system prompt contains the untrusted-data rules; they do
not prove a live model always obeys them. Treat answers as assistance, not authority, and review citations.

## Data handling

Document chunks are sent to OpenAI when `/chat` is used (question + retrieved passages). Search-only mode and local
embedding keep text on your machine. Do not index confidential documents unless that is acceptable under your
provider agreement. `ADMIN_TOKEN` unset means the protected endpoints are open (a warning is logged at startup).

## Publishing checklist

No `.env`, real keys, databases (`data/*`), vector indexes or private PDFs in Git. `sample_documents/` holds only the
fictional sample. Replace the licence placeholder.
