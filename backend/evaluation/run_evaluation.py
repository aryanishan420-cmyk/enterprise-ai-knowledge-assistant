"""Run the evaluation:  python -m backend.evaluation.run_evaluation [options]

Default (offline): ingests the sample PDF into a TEMPORARY index and reports retrieval metrics.
--generate : also run end-to-end RAG answers (needs OPENAI_API_KEY and network).
--judge    : additionally ask the LLM for a faithfulness estimate (model-based, not ground truth).
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from backend.api.state import build_state
from backend.config import ConfigError, get_settings
from backend.embeddings.embedding_service import get_embedder
from backend.evaluation.evaluator import evaluate_generation, evaluate_retrieval, load_dataset
from backend.rag.chain import RAGChain, build_llm

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=str(ROOT / "evaluation_data" / "sample_questions.json"))
    p.add_argument("--embedder", choices=["sentence-transformers", "hashing"], help="override EMBEDDING_BACKEND")
    p.add_argument("--ks", default="1,3,5", help="comma-separated k values")
    p.add_argument("--generate", action="store_true")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--output", help="results JSON path (default: <data dir>/evaluation_results.json)")
    a = p.parse_args(argv)

    settings = get_settings()
    if a.embedder:
        settings = settings.model_copy(update={"embedding_backend": a.embedder})
    dataset = load_dataset(a.dataset)
    pdf = Path(a.dataset).resolve().parent.parent / dataset["document"]
    if not pdf.exists():
        sys.exit(f"Sample document not found: {pdf}. Run: python scripts/generate_sample_pdf.py")
    output = Path(a.output) if a.output else Path(settings.vector_store_path).parent / "evaluation_results.json"

    try:
        embedder = get_embedder(settings)
    except Exception as exc:  # e.g. model cannot be downloaded
        sys.exit(
            f"Could not load embedder '{settings.embedding_backend}': {exc}\n"
            "Tip: use --embedder hashing for an offline (lexical, non-semantic) run."
        )
    llm = None
    if a.generate:
        try:
            llm = build_llm(settings)
        except ConfigError as exc:
            sys.exit(f"--generate needs an LLM: {exc}")

    with tempfile.TemporaryDirectory() as tmp:  # never touch the user's real index
        iso = settings.model_copy(update={"vector_store_path": f"{tmp}/vs", "database_url": f"sqlite:///{tmp}/eval.db"})
        state = build_state(iso, embedder, llm)
        ing = state.ingestion.ingest(pdf.name, pdf.read_bytes())
        ks = tuple(int(k) for k in a.ks.split(","))
        retrieval = evaluate_retrieval(state.retriever, dataset["questions"], ks)
        generation = None
        if llm:
            generation = evaluate_generation(RAGChain(state.retriever, llm), dataset, llm if a.judge else None)

    semantic = not embedder.name.startswith("hashing")
    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "end-to-end" if generation else "offline-retrieval",
        "embedder": {"name": embedder.name, "semantic": semantic},
        "llm_model": settings.llm_model if generation else None,
        "config": {"chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap, "ks": list(ks)},
        "dataset": {
            "path": a.dataset,
            "questions": len(dataset["questions"]),
            "unanswerable": len(dataset.get("unanswerable", [])),
            "indexed_chunks": ing["chunk_count"],
            "indexed_pages": ing["pages"],
        },
        "retrieval": retrieval,
        "generation": generation,
        "caveats": [
            "Tiny fictional dataset; results show the pipeline works, not production quality.",
            "Recall@k with k >= number of pages is trivially 1.0 for page-level relevance.",
            "keyword_match is a deterministic proxy; faithfulness_estimate (if present) is LLM-judged "
            "and is an estimate, not ground truth.",
        ]
        + ([] if semantic else ["Embedder is lexical hashing (NOT semantic): numbers reflect word overlap only."]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Mode: {results['mode']} | embedder: {embedder.name} (semantic={semantic})")
    print(f"Questions: {retrieval['n_questions']} | chunks indexed: {ing['chunk_count']}")
    for k, v in retrieval["recall_at_k"].items():
        print(f"  Recall@{k}: {v:.3f}")
    print(f"  MRR: {retrieval['mrr']:.3f} | retrieval latency mean/p95 ms: {retrieval['latency_ms']}")
    if generation:
        g = generation
        print(
            f"  keyword_match: {g['keyword_match_rate']:.3f} | citation_correct: {g['citation_correct_rate']:.3f} "
            f"| failure_rate: {g['failure_rate']:.3f} | correct_fallback: {g['correct_fallback_rate']}"
        )
    print(f"Results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
