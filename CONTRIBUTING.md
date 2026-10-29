# Contributing

1. Fork and create a branch: `git checkout -b feature/short-description`.
2. Set up the environment as described in the README (`pip install -r requirements-dev.txt`).
3. Make focused changes with tests. External services (OpenAI, Hugging Face) must be mocked; tests must pass without
   an API key or network access. Use `HashingEmbedder` and `FakeLLM` from `tests/conftest.py`.
4. Before opening a pull request run:
   ```powershell
   ruff check .
   ruff format .
   python -m pytest
   ```
5. Update the README or `docs/` if behaviour, configuration or endpoints change.
6. Never commit secrets, `.env`, private documents, databases or generated indexes.

Conventions: type hints on public functions, one responsibility per module (ingestion, retrieval, prompting and
generation stay separate), Pydantic schemas for API I/O, errors returned in the `{"error": {"code", "message"}}` format.
Report bugs with the issue template and remove any secrets or confidential text from logs.
