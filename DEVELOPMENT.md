# Development guide

## Setup (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Without PyTorch/Hugging Face access, set `EMBEDDING_BACKEND=hashing` in `.env` (lexical only). Tests do not need it.

## Common commands

| Task | Command | `make` target |
|---|---|---|
| Backend with reload | `python -m uvicorn backend.api.main:create_app --factory --reload --port 8000` | `run-backend` |
| Frontend | `python -m streamlit run frontend/app.py` | `run-frontend` |
| Tests | `python -m pytest` | `test` |
| Coverage | `python -m pytest --cov=backend --cov-report=term-missing` | |
| Lint / format | `ruff check .` / `ruff format .` | `lint` / `format` |
| Regenerate sample PDF | `python scripts/generate_sample_pdf.py` | `sample-pdf` |
| Offline evaluation | `python -m backend.evaluation.run_evaluation --embedder hashing` | `eval-offline` |
| Init database | `python scripts/initialize_database.py` | |

`make` is optional on Windows; the PowerShell commands above are equivalent.

## Conventions

- The application is built by `create_app(settings, embedder, llm)`; add dependencies to `AppState`
  (`backend/api/state.py`), not module globals.
- Keep ingestion, retrieval, prompting and generation separate; the retriever knows nothing about the LLM or HTTP.
- Tests: mock external services; use `HashingEmbedder`, `FakeLLM` (`tests/conftest.py`), `tmp_path` for storage.
  `tests/integration/` drives the real API (TestClient) and Streamlit pages (`streamlit.testing.v1.AppTest`).
- Never log document text, prompts or secrets. Log `error_type`, IDs and counts.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `sentence-transformers is not installed` | `pip install -r requirements.txt` |
| Model download fails / offline | `EMBEDDING_BACKEND=hashing` (or `--embedder hashing` for evaluation) |
| Backend refuses to start with "index was built with embedder ..." | Delete `data/vector_store` and re-upload, or restore the previous `EMBEDDING_*` |
| `/chat` returns 503 | Set `OPENAI_API_KEY` in `.env`, restart the backend |
| Streamlit says backend unreachable | Start uvicorn; check the sidebar API base URL |
| Upload returns 422 `no_text` | The PDF is image-only; OCR is not implemented |
| Port already in use | Pick another `--port` and update the sidebar URL / `API_BASE_URL` |
