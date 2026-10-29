"""Map [S#] markers in an answer back to the chunks that were actually retrieved."""

import re
from dataclasses import dataclass

from backend.indexing.vector_store import SearchHit

_MARKER = re.compile(r"\[S(\d+)\]")


@dataclass(frozen=True)
class Citation:
    label: str
    document_id: str
    filename: str
    page: int
    chunk_id: str
    score: float

    def format(self) -> str:
        return f"{self.filename}, p. {self.page}"


def to_citation(label_index: int, hit: SearchHit) -> Citation:
    c = hit.chunk
    return Citation(f"S{label_index}", c["document_id"], c["filename"], c["page"], c["chunk_id"], round(hit.score, 4))


def resolve_citations(answer: str, hits: list[SearchHit]) -> tuple[list[Citation], list[str]]:
    """Return (valid citations in order of first use, labels the model invented)."""
    valid, invalid, seen = [], [], set()
    for m in _MARKER.finditer(answer):
        n = int(m.group(1))
        label = f"S{n}"
        if label in seen:
            continue
        seen.add(label)
        if 1 <= n <= len(hits):
            valid.append(to_citation(n, hits[n - 1]))
        else:
            invalid.append(label)
    return valid, invalid


def strip_invalid_markers(answer: str, invalid: list[str]) -> str:
    for label in invalid:
        answer = answer.replace(f"[{label}]", "")
    return re.sub(r"[ \t]{2,}", " ", answer).strip()
