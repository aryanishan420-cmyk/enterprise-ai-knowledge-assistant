"""Streamlit pages exercised headlessly (streamlit.testing AppTest) against a real local API server."""

import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

from backend.api.main import create_app
from backend.config import Settings
from backend.embeddings.embedding_service import HashingEmbedder
from tests.conftest import FakeLLM

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
PDF = ROOT / "sample_documents" / "northwind_policies.pdf"
sys.path.insert(0, str(FRONTEND))  # pages import `common` like `streamlit run` does


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def serve(tmp_path):
    servers = []

    def start(llm=None, sub="a"):
        d = tmp_path / sub
        s = Settings(
            _env_file=None,
            embedding_backend="hashing",
            openai_api_key="",
            vector_store_path=str(d / "vs"),
            database_url=f"sqlite:///{d}/app.db",
        )
        port = _free_port()
        server = uvicorn.Server(
            uvicorn.Config(create_app(s, HashingEmbedder(), llm), host="127.0.0.1", port=port, log_level="error")
        )
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        servers.append(server)
        import httpx

        r = httpx.post(
            f"http://127.0.0.1:{port}/documents/upload",
            files=[("files", (PDF.name, PDF.read_bytes(), "application/pdf"))],
        )
        assert r.status_code == 201
        return f"http://127.0.0.1:{port}", d

    yield start
    for s in servers:
        s.should_exit = True


def _page(name: str, base: str) -> AppTest:
    at = AppTest.from_file(str(FRONTEND / name), default_timeout=30)
    at.session_state["api_base"] = base
    return at.run()


def _text(at: AppTest) -> str:
    els = [*at.markdown, *at.error, *at.warning, *at.caption, *at.info]
    return " ".join(str(e.value) for e in els)


def test_chat_shows_answer_and_page_citation(serve):
    base, _ = serve(FakeLLM("Employees get 22 vacation days [S1]."))
    at = _page("app.py", base)
    at.chat_input[0].set_value("How many vacation days?").run()
    assert not at.exception
    t = _text(at)
    assert "22 vacation days" in t and "northwind_policies.pdf, p. 1" in t


def test_missing_api_key_message_and_search_only_still_works(serve):
    base, _ = serve(llm=None)
    at = _page("app.py", base)
    at.chat_input[0].set_value("vacation days").run()
    assert "OPENAI_API_KEY" in _text(at)
    at.sidebar.radio[0].set_value("Search only")
    at.chat_input[0].set_value("vacation days").run()
    assert not at.exception and "passage(s)" in _text(at)


def test_backend_unavailable_message():
    at = _page("app.py", f"http://127.0.0.1:{_free_port()}")
    assert "Cannot reach the backend" in _text(at)


def test_documents_monitoring_and_evaluation_pages(serve):
    base, d = serve(FakeLLM())
    docs = _page("pages/1_Documents.py", base)
    assert not docs.exception and any(b.label == "Delete" for b in docs.button)
    assert any(PDF.name in str(m.value) for m in docs.markdown)
    mon = _page("pages/2_Monitoring.py", base)
    assert not mon.exception and any(m.label == "Indexed chunks" and m.value == "6" for m in mon.metric)
    assert "No evaluation results" in _text(_page("pages/3_Evaluation.py", base))
    (d / "evaluation_results.json").write_text((ROOT / "tests" / "fixtures_eval.json").read_text())
    ev = _page("pages/3_Evaluation.py", base)
    assert not ev.exception and any(m.label == "MRR" for m in ev.metric)
    assert "NOT measure semantic" in _text(ev)


def test_delete_button_removes_document(serve):
    import httpx

    base, _ = serve(FakeLLM())
    at = _page("pages/1_Documents.py", base)
    next(b for b in at.button if b.label == "Delete").click().run()
    assert httpx.get(f"{base}/documents").json() == []
