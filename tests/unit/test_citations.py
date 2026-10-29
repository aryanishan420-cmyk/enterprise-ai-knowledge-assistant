from backend.indexing.vector_store import SearchHit
from backend.rag.citations import resolve_citations, strip_invalid_markers


def _hit(n, page=1):
    return SearchHit(
        {"document_id": f"d{n}", "filename": f"f{n}.pdf", "page": page, "chunk_id": f"d{n}:p{page}:c0", "text": "t"},
        0.5,
    )


def test_valid_markers_map_to_chunks_in_first_use_order():
    cits, bad = resolve_citations("A [S2] then B [S1] and again [S2]", [_hit(1, 3), _hit(2, 7)])
    assert [c.label for c in cits] == ["S2", "S1"] and bad == []
    assert cits[0].format() == "f2.pdf, p. 7" and cits[1].page == 3


def test_invented_labels_are_flagged_and_stripped():
    cits, bad = resolve_citations("Claim [S9] and [S1]", [_hit(1)])
    assert [c.label for c in cits] == ["S1"] and bad == ["S9"]
    assert "[S9]" not in strip_invalid_markers("Claim [S9] and [S1]", bad)


def test_zero_marker_is_invalid():
    assert resolve_citations("x [S0]", [_hit(1)])[1] == ["S0"]


def test_no_markers():
    assert resolve_citations("plain", [_hit(1)]) == ([], [])
