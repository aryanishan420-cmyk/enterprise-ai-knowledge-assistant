"""Page-by-page PDF text extraction using pypdf."""

import io
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from backend.document_loading.validation import PDFValidationError


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based
    text: str


def extract_pages(data: bytes) -> list[PageText]:
    """Extract raw text per page. Raises PDFValidationError for unreadable PDFs."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise PDFValidationError("Encrypted PDFs are not supported.", "encrypted")
        pages = [PageText(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    except PDFValidationError:
        raise
    except (PyPdfError, ValueError, KeyError, OSError, RecursionError) as exc:
        raise PDFValidationError("The PDF is corrupted or unreadable.", "corrupted") from exc
    if not pages:
        raise PDFValidationError("The PDF has no pages.", "no_pages")
    if not any(p.text.strip() for p in pages):
        raise PDFValidationError(
            "No extractable text found. This may be a scanned PDF; OCR is not implemented.", "no_text"
        )
    return pages
