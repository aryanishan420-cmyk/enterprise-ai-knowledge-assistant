"""HTTP-level tests through FastAPI's TestClient. No network or paid API (LLM is a fake)."""

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.config import Settings
from backend.embeddings.embedding_service import HashingEmbedder
from tests.conftest import FakeLLM, make_pdf

POLICY = [
    "Vacation policy\nEmployees accrue twenty vacation days per year.",
    "Security policy\nPasswords must be rotated every ninety days.",
]


def _settings(tmp_path, **kw):
    return Settings(
        _env_file=None,
        embedding_backend="hashing",
        max_upload_mb=1,
        openai_api_key="",
        vector_store_path=str(tmp_path / "vs"),
        database_url=f"sqlite:///{tmp_path}/app.db",
        **kw,
    )


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(_settings(tmp_path), HashingEmbedder(), FakeLLM("Twenty days [S1]"))) as c:
        yield c


def _upload(client, name="policy.pdf", data=None, headers=None):
    data = data if data is not None else make_pdf(POLICY)
    return client.post("/documents/upload", files=[("files", (name, data, "application/pdf"))], headers=headers or {})


def test_health(client):
    j = client.get("/health").json()
    assert j["status"] == "ok" and j["indexed_chunks"] == 0


def test_upload_list_search_chat_delete_roundtrip(client):
    r = _upload(client)
    assert r.status_code == 201 and r.json()["results"][0]["pages"] == 2
    doc_id = r.json()["results"][0]["document_id"]
    assert [d["document_id"] for d in client.get("/documents").json()] == [doc_id]

    s = client.post("/search", json={"query": "vacation days", "top_k": 2}).json()
    assert s["results"][0]["page"] == 1 and s["results"][0]["filename"] == "policy.pdf"

    c = client.post("/chat", json={"query": "How many vacation days?"}).json()
    assert c["citations"][0]["page"] == 1 and not c["insufficient_context"] and c["retrieved"]

    assert client.delete(f"/documents/{doc_id}").status_code == 200
    assert client.get("/documents").json() == []
    assert client.post("/search", json={"query": "vacation days"}).json()["results"] == []
    assert client.get("/health").json()["indexed_chunks"] == 0
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_deletion_removes_db_record_too(client):
    doc_id = _upload(client).json()["results"][0]["document_id"]
    state = client.app.state.eaka
    assert len(state.repo.list_documents()) == 1
    client.delete(f"/documents/{doc_id}")
    assert state.repo.list_documents() == []


def test_duplicate_upload_conflict(client):
    _upload(client)
    r = _upload(client)
    assert r.status_code == 409 and r.json()["error"]["code"] == "duplicate"


@pytest.mark.parametrize(
    "name,data,status,code",
    [
        ("notes.txt", b"hello", 415, "unsupported_type"),
        ("empty.pdf", b"", 422, "empty_file"),
        ("fake.pdf", b"not a pdf at all", 422, "bad_signature"),
        ("bad.pdf", b"%PDF-1.4 garbage", 422, "corrupted"),
        ("big.pdf", b"%PDF-" + b"0" * (1024 * 1024 + 10), 413, "file_too_large"),
    ],
)
def test_invalid_uploads(client, name, data, status, code):
    r = _upload(client, name, data)
    assert r.status_code == status and r.json()["error"]["code"] == code


def test_request_validation_errors_are_consistent(client):
    for path, body in [
        ("/search", {"query": ""}),
        ("/chat", {"query": "x", "top_k": 0}),
        ("/search", {}),
        ("/chat", {"query": "x" * 2001}),
    ]:
        r = client.post(path, json=body)
        assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_irrelevant_query_on_empty_index_gives_fallback(client):
    c = client.post("/chat", json={"query": "anything"}).json()
    assert c["insufficient_context"] and c["citations"] == []


def test_chat_without_api_key_returns_503(tmp_path):
    with TestClient(create_app(_settings(tmp_path), HashingEmbedder())) as c:
        _upload(c)
        r = c.post("/chat", json={"query": "vacation"})
        assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["error"]["message"]
        assert c.post("/search", json={"query": "vacation"}).status_code == 200  # search-only still works


def test_llm_failure_returns_502(tmp_path):
    from backend.rag.chain import LLMError

    app = create_app(_settings(tmp_path), HashingEmbedder(), FakeLLM(error=LLMError("timeout")))
    with TestClient(app) as c:
        _upload(c)
        r = c.post("/chat", json={"query": "vacation days"})
        assert r.status_code == 502 and r.json()["error"]["code"] == "llm_unavailable"


def test_admin_token_protects_destructive_and_metrics_endpoints(tmp_path):
    app = create_app(_settings(tmp_path, admin_token="s3cret"), HashingEmbedder(), FakeLLM())
    with TestClient(app) as c:
        assert _upload(c).status_code == 401
        assert _upload(c, headers={"X-API-Token": "wrong"}).status_code == 401
        ok = _upload(c, headers={"X-API-Token": "s3cret"})
        assert ok.status_code == 201
        doc = ok.json()["results"][0]["document_id"]
        assert c.delete(f"/documents/{doc}").status_code == 401
        assert c.get("/metrics").status_code == 401
        assert c.get("/metrics", headers={"X-API-Token": "s3cret"}).status_code == 200
        assert c.delete(f"/documents/{doc}", headers={"X-API-Token": "s3cret"}).status_code == 200


def test_metrics_reflect_real_activity(client):
    _upload(client)
    client.post("/search", json={"query": "vacation"})
    client.post("/chat", json={"query": "vacation"})
    m = client.get("/metrics").json()
    assert m["counters"]["requests_total"] >= 3 and m["counters"]["documents_ingested"] == 1
    assert m["index"]["documents"] == 1 and m["queries_logged"] == 2
    assert "retrieval_ms" in m["latency_ms"] and "llm_ms" in m["latency_ms"]


def test_failed_ingestion_counted(client):
    _upload(client, "x.pdf", b"%PDF-1.4 garbage")
    assert client.get("/metrics").json()["counters"]["ingestion_failures"] == 1


def test_query_log_stores_no_raw_question(client):
    client.post("/search", json={"query": "TOPSECRET-QUESTION-TEXT"})
    from sqlalchemy import text

    with client.app.state.eaka.repo.engine.connect() as conn:
        rows = conn.execute(text("select * from query_logs")).fetchall()
    assert rows and "TOPSECRET" not in str(rows)


def test_evaluation_endpoint_404_then_returns_saved_results(client):
    assert client.get("/evaluation").status_code == 404
    client.app.state.eaka.evaluation_path.write_text('{"recall_at_k": 1.0}')
    assert client.get("/evaluation").json() == {"recall_at_k": 1.0}


def test_persistence_across_app_restart(tmp_path):
    s = _settings(tmp_path)
    with TestClient(create_app(s, HashingEmbedder(), FakeLLM())) as c:
        _upload(c)
    with TestClient(create_app(s, HashingEmbedder(), FakeLLM())) as c2:
        assert len(c2.get("/documents").json()) == 1
        assert c2.post("/search", json={"query": "passwords rotated"}).json()["results"][0]["page"] == 2
