import json
from pathlib import Path

import pytest

from backend.document_loading.pdf_reader import extract_pages
from backend.evaluation.evaluator import (
    evaluate_generation,
    evaluate_retrieval,
    judge_faithfulness,
    keyword_match,
    load_dataset,
)
from backend.evaluation.run_evaluation import main as run_eval
from backend.indexing.vector_store import SearchHit
from backend.rag.chain import LLMError, RAGChain
from backend.retrieval.retriever import RetrievalResult
from scripts.generate_sample_pdf import PAGES, build_pdf
from tests.conftest import FakeLLM

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "evaluation_data" / "sample_questions.json"


class ScriptedRetriever:
    """Returns pre-set (filename, page) hits per query to test metric arithmetic exactly."""

    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query, top_k=None, owner=None):
        hits = [SearchHit({"filename": f, "page": p, "text": "t"}, 1.0) for f, p in self.ranking[query]]
        return RetrievalResult(hits[:top_k], 1.0)


def test_recall_and_mrr_arithmetic():
    qs = [
        {"id": "a", "question": "a", "source": {"filename": "f.pdf", "pages": [1]}},
        {"id": "b", "question": "b", "source": {"filename": "f.pdf", "pages": [2]}},
        {"id": "c", "question": "c", "source": {"filename": "f.pdf", "pages": [9]}},
    ]
    r = ScriptedRetriever(
        {
            "a": [("f.pdf", 1), ("f.pdf", 2)],  # rank 1
            "b": [("f.pdf", 1), ("f.pdf", 3), ("f.pdf", 2)],  # rank 3
            "c": [("f.pdf", 1), ("f.pdf", 2)],
        }
    )  # miss
    m = evaluate_retrieval(r, qs, ks=(1, 3))
    assert m["recall_at_k"] == {"1": round(1 / 3, 4), "3": round(2 / 3, 4)}
    assert m["mrr"] == round((1 + 1 / 3 + 0) / 3, 4)


def test_wrong_filename_is_not_relevant():
    qs = [{"id": "a", "question": "a", "source": {"filename": "f.pdf", "pages": [1]}}]
    m = evaluate_retrieval(ScriptedRetriever({"a": [("other.pdf", 1)]}), qs, ks=(1,))
    assert m["recall_at_k"]["1"] == 0.0 and m["mrr"] == 0.0


def test_keyword_match_whole_tokens():
    assert keyword_match("You get 22 days.", ["22"])
    assert not keyword_match("You get 122 days.", ["22"])
    assert not keyword_match("about 2.2 days", ["22"])
    assert keyword_match("Hours are 10:00 to 15:00.", ["10:00", "15:00"])


def test_sample_pdf_is_valid_with_one_section_per_page(tmp_path):
    pages = extract_pages(build_pdf(tmp_path / "s.pdf").read_bytes())
    assert len(pages) == len(PAGES) == 6 and "Vacation" in pages[0].text


def test_dataset_is_consistent_with_committed_sample_pdf():
    """Every expected keyword must literally appear on the page the dataset cites."""
    ds = load_dataset(DATASET)
    pages = extract_pages((ROOT / ds["document"]).read_bytes())
    for q in ds["questions"]:
        text = " ".join(pages[p - 1].text for p in q["source"]["pages"])
        assert keyword_match(text, q["expected_keywords"]), q["id"]


def test_load_dataset_rejects_malformed(tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"questions": [{"id": "x", "question": "q", "source": {}}]}))
    with pytest.raises(ValueError):
        load_dataset(f)


@pytest.fixture
def sample_chain_parts(store, embedder, tmp_path, settings):
    from backend.api.state import build_state

    state = build_state(settings, embedder)
    pdf = ROOT / "sample_documents" / "northwind_policies.pdf"
    state.ingestion.ingest(pdf.name, pdf.read_bytes())
    return state


def test_generation_metrics_with_scripted_llm(sample_chain_parts):
    ds = load_dataset(DATASET)
    ds["questions"] = ds["questions"][:1]  # q01: 22 days, page 1
    ds["unanswerable"] = ds["unanswerable"][:1]
    llm = FakeLLM("Full-time staff get 22 vacation days [S1].")
    # S1 is the top hit for the vacation question (page 1) with the lexical embedder
    g = evaluate_generation(RAGChain(sample_chain_parts.retriever, llm), ds)
    assert g["keyword_match_rate"] == 1.0 and g["citation_correct_rate"] == 1.0
    assert g["correct_fallback_rate"] == 0.0  # scripted LLM never refuses -> counted as wrong fallback
    assert g["failure_rate"] == 0.0


def test_generation_counts_failures_and_fallbacks(sample_chain_parts):
    ds = load_dataset(DATASET)
    ds["questions"] = ds["questions"][:2]
    g = evaluate_generation(RAGChain(sample_chain_parts.retriever, FakeLLM(error=LLMError("x"))), ds)
    assert g["failure_rate"] == 1.0
    g2 = evaluate_generation(RAGChain(sample_chain_parts.retriever, FakeLLM("INSUFFICIENT_CONTEXT")), ds)
    assert g2["false_refusal_rate"] == 1.0 and g2["correct_fallback_rate"] == 1.0


def test_judge_parses_score_and_handles_garbage():
    r = [{"label": "S1", "text": "t"}]
    assert judge_faithfulness(FakeLLM('{"score": 4}'), "q", "a", r) == 4
    assert judge_faithfulness(FakeLLM("no idea"), "q", "a", r) is None


def test_run_evaluation_offline_writes_results(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    from backend.config import get_settings

    get_settings.cache_clear()
    out = tmp_path / "res.json"
    assert run_eval(["--embedder", "hashing", "--output", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["mode"] == "offline-retrieval" and res["embedder"]["semantic"] is False
    assert res["retrieval"]["recall_at_k"]["5"] == 1.0 and res["generation"] is None
    assert any("NOT semantic" in c for c in res["caveats"])
    get_settings.cache_clear()


def test_run_evaluation_generate_without_key_exits_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from backend.config import get_settings

    get_settings.cache_clear()
    with pytest.raises(SystemExit) as e:
        run_eval(["--embedder", "hashing", "--generate", "--output", str(tmp_path / "x.json")])
    assert "OPENAI_API_KEY" in str(e.value)
    get_settings.cache_clear()
