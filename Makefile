# Convenience targets. On Windows without make, run the equivalent commands from docs/DEVELOPMENT.md.
PY ?= python

.PHONY: install install-dev run-backend run-frontend test lint format sample-pdf eval eval-offline eval-generate docker-up docker-down clean

install:
	$(PY) -m pip install -r requirements.txt

install-dev:
	$(PY) -m pip install -r requirements-dev.txt

run-backend:
	$(PY) -m uvicorn backend.api.main:create_app --factory --reload --port 8000

run-frontend:
	$(PY) -m streamlit run frontend/app.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

sample-pdf:
	$(PY) scripts/generate_sample_pdf.py

eval-offline:
	$(PY) -m backend.evaluation.run_evaluation --embedder hashing

eval:
	$(PY) -m backend.evaluation.run_evaluation

eval-generate:
	$(PY) -m backend.evaluation.run_evaluation --generate --judge

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
