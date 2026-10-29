"""Page-aware overlapping character chunking."""

from dataclasses import asdict, dataclass

from backend.document_loading.pdf_reader import PageText
from backend.processing.text_cleaner import clean_text


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    page: int
    index: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # prefer breaking at whitespace in the last 20% of the window
            cut = text.rfind(" ", start + int(size * 0.8), end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_pages(
    document_id: str, filename: str, pages: list[PageText], size: int = 800, overlap: int = 120
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for i, piece in enumerate(split_text(clean_text(page.text), size, overlap)):
            chunks.append(
                Chunk(f"{document_id}:p{page.page_number}:c{i}", document_id, filename, page.page_number, i, piece)
            )
    return chunks
