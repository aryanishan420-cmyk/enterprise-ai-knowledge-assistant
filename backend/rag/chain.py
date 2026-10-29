"""RAG orchestration: retrieve -> prompt -> generate -> verify citations."""

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

from backend.config import Settings
from backend.rag.citations import Citation, resolve_citations, strip_invalid_markers
from backend.rag.prompts import FALLBACK_MESSAGE, INSUFFICIENT, build_messages
from backend.retrieval.retriever import Retriever

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The LLM provider failed or timed out."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClient(Protocol):
    def complete(self, messages: list[dict]) -> LLMResult: ...


class OpenAIClient:
    def __init__(self, api_key: str, model: str, timeout: float = 30.0):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=1)
        self.model = model

    def complete(self, messages: list[dict]) -> LLMResult:
        try:
            r = self._client.chat.completions.create(model=self.model, messages=messages, temperature=0)
        except Exception as exc:  # provider SDK raises many types; log type only, never content
            log.error("llm_request_failed", extra={"fields": {"error_type": type(exc).__name__}})
            raise LLMError("The language model request failed or timed out.") from exc
        u = getattr(r, "usage", None)
        return LLMResult(
            r.choices[0].message.content or "", getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None)
        )


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider != "openai":
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
    return OpenAIClient(settings.require_llm_key(), settings.llm_model)


@dataclass
class RAGResponse:
    answer: str
    insufficient_context: bool
    citations: list[Citation]
    retrieved: list[dict]
    invalid_citations: list[str] = field(default_factory=list)
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class RAGChain:
    def __init__(self, retriever: Retriever, llm: LLMClient):
        self.retriever, self.llm = retriever, llm

    def answer(self, question: str, top_k: int | None = None, owner: str | None = None) -> RAGResponse:
        res = self.retriever.search(question, top_k, owner)
        retrieved = [{"label": f"S{i}", "score": round(h.score, 4), **h.chunk} for i, h in enumerate(res.hits, 1)]
        if not res.hits:  # nothing relevant: skip the LLM entirely
            return RAGResponse(FALLBACK_MESSAGE, True, [], [], retrieval_ms=res.latency_ms)
        t0 = time.perf_counter()
        out = self.llm.complete(build_messages(question, res.hits))
        llm_ms = (time.perf_counter() - t0) * 1000
        base = dict(
            retrieved=retrieved,
            retrieval_ms=res.latency_ms,
            llm_ms=llm_ms,
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
        )
        if INSUFFICIENT in out.text:
            return RAGResponse(FALLBACK_MESSAGE, True, [], **base)
        citations, invalid = resolve_citations(out.text, res.hits)
        return RAGResponse(
            strip_invalid_markers(out.text, invalid), False, citations, invalid_citations=invalid, **base
        )
