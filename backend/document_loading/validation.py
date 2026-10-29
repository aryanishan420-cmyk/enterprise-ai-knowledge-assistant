"""Upload validation: type, size, magic bytes."""

from dataclasses import dataclass


class PDFValidationError(ValueError):
    def __init__(self, message: str, code: str = "invalid_pdf"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ValidatedUpload:
    filename: str
    size_bytes: int


def validate_upload(filename: str, data: bytes, max_bytes: int) -> ValidatedUpload:
    name = (filename or "").replace("\\", "/").split("/")[-1].strip()
    if not name.lower().endswith(".pdf"):
        raise PDFValidationError("Only .pdf files are accepted.", "unsupported_type")
    if not data:
        raise PDFValidationError("The file is empty.", "empty_file")
    if len(data) > max_bytes:
        raise PDFValidationError(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit.", "file_too_large")
    if data.lstrip()[:5] != b"%PDF-":
        raise PDFValidationError("File content is not a valid PDF.", "bad_signature")
    return ValidatedUpload(filename=name, size_bytes=len(data))
