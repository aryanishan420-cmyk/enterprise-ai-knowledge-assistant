import numpy as np
import pytest

from backend.indexing.vector_store import DimensionMismatchError, DuplicateDocumentError, VectorStore
from backend.processing.chunking import Chunk


def _chunks(doc, texts, name="f.pdf"):
    return [Chunk(f"{doc}:p1:c{i}", doc, name, 1, i, t) for i, t in enumerate(texts)]


def _add(store, embedder, doc, texts, owner="default"):
    store.add_document(doc, f"{doc}.pdf", _chunks(doc, texts), embedder.embed(texts), owner)


def test_empty_index_search_returns_nothing(store, embedder):
    assert store.search(embedder.embed(["anything"])[0]) == []


def test_search_ranks_relevant_first(store, embedder):
    _add(store, embedder, "d1", ["refund policy allows returns within 30 days", "the cafeteria opens at nine"])
    hits = store.search(embedder.embed(["refund returns policy"])[0], top_k=2)
    assert "refund" in hits[0].chunk["text"] and hits[0].score > hits[1].score


def test_persistence_across_restart(store, embedder, tmp_path):
    _add(store, embedder, "d1", ["persistent chunk about vacation days"])
    reopened = VectorStore(tmp_path / "vs", embedder.dim, embedder.name)
    assert reopened.chunk_count == 1 and reopened.list_documents()[0]["filename"] == "d1.pdf"
    assert reopened.search(embedder.embed(["vacation days"])[0])[0].chunk["document_id"] == "d1"


def test_duplicate_rejected(store, embedder):
    _add(store, embedder, "d1", ["a"])
    with pytest.raises(DuplicateDocumentError):
        _add(store, embedder, "d1", ["a"])


def test_delete_removes_chunks_and_persists(store, embedder, tmp_path):
    _add(store, embedder, "d1", ["alpha beta"])
    _add(store, embedder, "d2", ["gamma delta"])
    assert store.delete_document("d1") is True and store.delete_document("d1") is False
    assert [h.chunk["document_id"] for h in store.search(embedder.embed(["alpha"])[0], 5)] == ["d2"]
    assert VectorStore(tmp_path / "vs", embedder.dim).chunk_count == 1


def test_threshold_filters_low_scores(store, embedder):
    _add(store, embedder, "d1", ["completely unrelated words here"])
    assert store.search(embedder.embed(["zebra quantum"])[0], threshold=0.5) == []


def test_owner_isolation(store, embedder):
    _add(store, embedder, "d1", ["secret salary bands"], owner="alice")
    q = embedder.embed(["salary bands"])[0]
    assert store.search(q, owner="bob") == []
    assert len(store.search(q, owner="alice")) == 1
    assert store.list_documents(owner="bob") == []


def test_dimension_mismatch_detected(store, embedder, tmp_path):
    _add(store, embedder, "d1", ["x"])
    with pytest.raises(DimensionMismatchError):
        VectorStore(tmp_path / "vs", 128)


def test_length_mismatch(store, embedder):
    with pytest.raises(ValueError):
        store.add_document("d", "d.pdf", _chunks("d", ["a", "b"]), np.zeros((1, embedder.dim)))


def test_embedder_name_mismatch_detected_even_with_same_dimension(store, embedder, tmp_path):
    _add(store, embedder, "d1", ["x"])
    with pytest.raises(DimensionMismatchError, match="different models"):
        VectorStore(tmp_path / "vs", embedder.dim, "some-other-384-dim-model")
