"""Pydantic request/response models."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class SearchRequest(BaseModel):
    query: Query
    top_k: int | None = Field(None, ge=1, le=50)


class ChatRequest(SearchRequest):
    include_passages: bool = True


class Passage(BaseModel):
    label: str
    score: float
    document_id: str
    filename: str
    page: int
    chunk_id: str
    text: str


class SearchResponse(BaseModel):
    results: list[Passage]
    retrieval_ms: float


class CitationOut(BaseModel):
    label: str
    document_id: str
    filename: str
    page: int
    chunk_id: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    insufficient_context: bool
    citations: list[CitationOut]
    invalid_citations_removed: list[str]
    retrieved: list[Passage]
    retrieval_ms: float
    llm_ms: float
    total_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class DocumentOut(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunk_count: int


class UploadItem(BaseModel):
    filename: str
    status: str  # "indexed" | "error"
    document_id: str | None = None
    pages: int | None = None
    chunk_count: int | None = None
    ingestion_ms: float | None = None
    error_code: str | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    results: list[UploadItem]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
