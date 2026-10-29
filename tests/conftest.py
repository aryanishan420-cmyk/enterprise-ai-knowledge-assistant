import io

import pytest
from reportlab.pdfgen import canvas

from backend.config import Settings
from backend.embeddings.embedding_service import HashingEmbedder
from backend.indexing.vector_store import VectorStore


def make_pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 14
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture
def pdf_factory():
    return make_pdf


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def store(tmp_path, embedder):
    return VectorStore(tmp_path / "vs", embedder.dim, embedder.name)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        embedding_backend="hashing",
        openai_api_key="",
        vector_store_path=str(tmp_path / "vs"),
        database_url=f"sqlite:///{tmp_path}/app.db",
    )


class FakeLLM:
    """Scripted LLM for tests: returns `reply`, records the messages it received."""

    def __init__(self, reply="Answer [S1]", prompt_tokens=10, completion_tokens=5, error=None):
        self.reply, self.error, self.calls = reply, error, []
        self.pt, self.ct = prompt_tokens, completion_tokens

    def complete(self, messages):
        from backend.rag.chain import LLMResult

        self.calls.append(messages)
        if self.error:
            raise self.error
        return LLMResult(self.reply, self.pt, self.ct)


@pytest.fixture
def fake_llm():
    return FakeLLM()
