"""Evaluation metrics.

* Retrieval (offline, deterministic): Recall@k and Mean Reciprocal Rank. A hit is relevant when its
  filename and page match the dataset's supporting source.
* Generation (needs an LLM): keyword match (deterministic proxy), citation correctness, fallback
  behaviour on unanswerable questions, latency and failure rate.
* Optional LLM-judged faithfulness: a model-based ESTIMATE, not ground truth.
"""

import json
import re
import time
from pathlib import Path

import numpy as np

from backend.rag.chain import LLMClient, LLMError, RAGChain
from backend.retrieval.retriever import Retriever


def load_dataset(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for q in data["questions"]:
        if not q.get("question") or not q.get("source", {}).get("pages"):
            raise ValueError(f"Malformed dataset entry: {q.get('id')}")
    return data


def is_relevant(chunk: dict, source: dict) -> bool:
    return chunk["filename"] == source["filename"] and chunk["page"] in source["pages"]


def keyword_match(answer: str, keywords: list[str]) -> bool:
    """All keywords appear as whole tokens (case-insensitive)."""
    return all(re.search(rf"(?<![\w.]){re.escape(k)}(?![\w])", answer, re.I) for k in keywords)


def _latency(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "p95": None}
    a = np.asarray(values)
    return {"mean": round(float(a.mean()), 2), "p95": round(float(np.percentile(a, 95)), 2)}


def evaluate_retrieval(retriever: Retriever, questions: list[dict], ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    ks = tuple(sorted(set(ks)))
    ranks, lats, per_q = [], [], []
    for q in questions:
        res = retriever.search(q["question"], top_k=max(ks))
        rank = next((i for i, h in enumerate(res.hits, 1) if is_relevant(h.chunk, q["source"])), None)
        ranks.append(rank)
        lats.append(res.latency_ms)
        per_q.append({"id": q["id"], "rank": rank, "retrieval_ms": round(res.latency_ms, 2)})
    n = len(questions)
    return {
        "n_questions": n,
        "recall_at_k": {str(k): round(sum(1 for r in ranks if r and r <= k) / n, 4) for k in ks},
        "mrr": round(sum(1 / r for r in ranks if r) / n, 4),
        "latency_ms": _latency(lats),
        "per_question": per_q,
    }


JUDGE_PROMPT = (
    "You grade whether an ANSWER is fully supported by the CONTEXT passages. Reply with only "
    'JSON: {"score": <integer 1-5>} where 5 = every claim supported, 1 = unsupported.'
)


def judge_faithfulness(llm: LLMClient, question: str, answer: str, retrieved: list[dict]) -> int | None:
    """LLM-as-judge estimate (1-5). Returns None if the judge output cannot be parsed."""
    ctx = "\n".join(f"[{r['label']}] {r['text']}" for r in retrieved)
    out = llm.complete(
        [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {question}\nANSWER: {answer}"},
        ]
    )
    m = re.search(r'"score"\s*:\s*([1-5])', out.text)
    return int(m.group(1)) if m else None


def evaluate_generation(chain: RAGChain, dataset: dict, judge_llm: LLMClient | None = None) -> dict:
    rows, failures, totals = [], 0, []
    for q in dataset["questions"]:
        t0 = time.perf_counter()
        try:
            r = chain.answer(q["question"])
        except LLMError:
            failures += 1
            rows.append({"id": q["id"], "failed": True})
            continue
        ms = (time.perf_counter() - t0) * 1000
        totals.append(ms)
        src = q["source"]
        row = {
            "id": q["id"],
            "failed": False,
            "insufficient_context": r.insufficient_context,
            "keyword_match": (not r.insufficient_context) and keyword_match(r.answer, q["expected_keywords"]),
            "citation_correct": any(c.filename == src["filename"] and c.page in src["pages"] for c in r.citations),
            "invalid_citations": len(r.invalid_citations),
            "total_ms": round(ms, 2),
            "llm_ms": round(r.llm_ms, 2),
            "faithfulness_estimate": None,
        }
        if judge_llm and not r.insufficient_context:
            row["faithfulness_estimate"] = judge_faithfulness(judge_llm, q["question"], r.answer, r.retrieved)
        rows.append(row)
    ok = [r for r in rows if not r["failed"]]
    n = len(dataset["questions"])
    unans, unans_fail = [], 0
    for u in dataset.get("unanswerable", []):
        try:
            unans.append(chain.answer(u["question"]).insufficient_context)
        except LLMError:
            unans_fail += 1
    judged = [r["faithfulness_estimate"] for r in ok if r["faithfulness_estimate"] is not None]
    return {
        "n_questions": n,
        "failure_rate": round((failures + unans_fail) / (n + len(dataset.get("unanswerable", []))), 4),
        "keyword_match_rate": round(sum(r["keyword_match"] for r in ok) / n, 4),
        "citation_correct_rate": round(sum(r["citation_correct"] for r in ok) / n, 4),
        "false_refusal_rate": round(sum(r["insufficient_context"] for r in ok) / n, 4),
        "correct_fallback_rate": round(sum(unans) / len(unans), 4) if unans else None,
        "faithfulness_estimate_mean": round(float(np.mean(judged)), 2) if judged else None,
        "latency_ms": _latency(totals),
        "per_question": rows,
    }
