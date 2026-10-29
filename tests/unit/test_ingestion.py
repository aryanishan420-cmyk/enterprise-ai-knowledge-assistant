import io

import pytest
from pypdf import PdfWriter

from backend.config import ConfigError, Settings
from backend.document_loading.pdf_reader import PageText, extract_pages
from backend.document_loading.validation import PDFValidationError, validate_upload
from backend.processing.chunking import chunk_pages, split_text
from backend.processing.text_cleaner import clean_text

MB = 1024 * 1024


def test_validate_rejects_bad_inputs(pdf_factory):
    good = pdf_factory(["hello"])
    assert validate_upload("a.pdf", good, MB).filename == "a.pdf"
    for name, data, code in [
        ("a.txt", good, "unsupported_type"),
        ("a.pdf", b"", "empty_file"),
        ("a.pdf", b"x" * 10, "bad_signature"),
        ("a.pdf", good, "file_too_large"),
    ]:
        with pytest.raises(PDFValidationError) as e:
            validate_upload(name, data, 10 if code == "file_too_large" else MB)
        assert e.value.code == code


def test_validate_strips_path_components(pdf_factory):
    assert validate_upload("../../etc/x.pdf", pdf_factory(["a"]), MB).filename == "x.pdf"


def test_extract_pages_keeps_page_numbers(pdf_factory):
    pages = extract_pages(pdf_factory(["alpha on page one", "beta on page two"]))
    assert [p.page_number for p in pages] == [1, 2]
    assert "beta" in pages[1].text


def test_corrupted_pdf_rejected():
    with pytest.raises(PDFValidationError) as e:
        extract_pages(b"%PDF-1.4 this is not really a pdf")
    assert e.value.code == "corrupted"


def test_encrypted_pdf_rejected():
    w = PdfWriter()
    w.add_blank_page(100, 100)
    w.encrypt("secret")
    buf = io.BytesIO()
    w.write(buf)
    with pytest.raises(PDFValidationError) as e:
        extract_pages(buf.getvalue())
    assert e.value.code == "encrypted"


def test_blank_pdf_reports_ocr_unavailable():
    w = PdfWriter()
    w.add_blank_page(100, 100)
    buf = io.BytesIO()
    w.write(buf)
    with pytest.raises(PDFValidationError) as e:
        extract_pages(buf.getvalue())
    assert e.value.code == "no_text" and "OCR is not implemented" in str(e.value)


def test_clean_text():
    assert clean_text("a   b\t\tc\n\n\n\nd  \n") == "a b c\n\nd"
    assert clean_text("inter-\nnational") == "international"


def test_split_text_overlap_and_bounds():
    text = " ".join(f"w{i}" for i in range(400))
    parts = split_text(text, 200, 40)
    assert len(parts) > 1 and all(len(p) <= 200 for p in parts)
    assert parts[0][-10:].split()[-1] in parts[1]  # overlap carries context


def test_split_text_invalid_params():
    with pytest.raises(ValueError):
        split_text("abc", 10, 10)


def test_chunk_metadata_preserved():
    pages = [PageText(1, "x " * 300), PageText(2, "short page")]
    chunks = chunk_pages("doc1", "f.pdf", pages, 200, 20)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.document_id == "doc1" and c.filename == "f.pdf" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_config_overlap_validation():
    with pytest.raises(ValueError):
        Settings(_env_file=None, chunk_size=200, chunk_overlap=200)


def test_missing_api_key_message():
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        Settings(_env_file=None, openai_api_key="").require_llm_key()


def test_public_settings_hide_secrets():
    d = Settings(_env_file=None, openai_api_key="sk-secret", admin_token="t").public_dict()
    assert "sk-secret" not in str(d) and d["llm_key_configured"] is True
