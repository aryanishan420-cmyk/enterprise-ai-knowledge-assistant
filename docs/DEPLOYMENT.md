# Deployment

> Nothing in this document has been executed against a real cloud account, and the Docker image has not been built
> in the environment where this project was developed. Treat it as a tested-by-reading starting point.

## Docker Compose

```powershell
Copy-Item .env.example .env        # add OPENAI_API_KEY for /chat; consider setting ADMIN_TOKEN
docker compose up --build
```

| Service | Port | Command |
|---|---|---|
| `backend` | 8000 | Dockerfile `CMD`: `uvicorn backend.api.main:create_app --factory --host 0.0.0.0 --port 8000` |
| `frontend` | 8501 | `streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501` with `API_BASE_URL=http://backend:8000` |

Configuration comes from `.env` (optional file; needs Docker Compose >= 2.24, otherwise remove `required: false`) and
the `environment:` overrides in `docker-compose.yml`. Secrets are never baked into the image (`.dockerignore` excludes
`.env`). The frontend starts after the backend health check passes.

**Persistence:** the named volume `app_data` is mounted at `/app/data` and holds the FAISS index, SQLite file and the
Hugging Face cache (`HF_HOME`). `docker compose down` keeps it; `docker compose down -v` deletes it.

Things to check when you first build: the PyTorch CPU wheel index in the Dockerfile
(`--extra-index-url https://download.pytorch.org/whl/cpu`), image size (PyTorch is large), and first-request latency
while the embedding model downloads.

## Cloud notes

Any container host that can run two services (or one service plus Streamlit Community Cloud / a second service) and
mount a **persistent disk** works. Requirements:

1. **Persistent storage at `/app/data`.** Free tiers of many platforms use an ephemeral filesystem: the index and
   database would be lost on every restart or redeploy. Either attach a persistent disk, or move metadata to a managed
   PostgreSQL via `DATABASE_URL` (untested) and accept that the FAISS index must still live on a disk, or be rebuilt.
2. **Secrets** (`OPENAI_API_KEY`, `ADMIN_TOKEN`) set through the platform's secret/environment-variable settings, never
   in the repository.
3. **Always set `ADMIN_TOKEN`** on anything reachable from the internet, and put TLS in front. There is no user login.
4. Set the frontend's `API_BASE_URL` to the backend's internal or public URL, and give the backend enough memory for
   the embedding model.
5. Health check path: `GET /health`.

Free-tier limits (CPU, RAM, sleep-on-idle, disk) change often; check the provider's current documentation before
choosing a plan. This project has not been deployed anywhere.

## CI

`.github/workflows/tests.yml` installs `requirements-dev.txt` (with the CPU PyTorch index), runs `ruff check .`,
`python -m pytest --cov=backend` and the offline evaluation on Python 3.11 and 3.12 without any API key. It has not been
executed on GitHub yet.
