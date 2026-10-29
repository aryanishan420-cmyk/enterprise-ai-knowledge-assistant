# Evaluation

## Dataset

`evaluation_data/sample_questions.json`: 15 answerable questions and 3 unanswerable ones about the **fictional**
`sample_documents/northwind_policies.pdf` (6 pages). Each answerable item has an expected answer, `expected_keywords`
and the supporting `filename` + `pages`. A test (`test_dataset_is_consistent_with_committed_sample_pdf`) verifies that
every expected keyword literally appears on the cited page.

## Metrics

**Offline retrieval (deterministic, no LLM):** a retrieved chunk is relevant if its filename and page match the
supporting source.
- Recall@k: fraction of questions with a relevant chunk in the top k.
- MRR: mean of 1/rank of the first relevant chunk (0 if none).
- Retrieval latency (mean, p95).

**End-to-end (`--generate`, needs `OPENAI_API_KEY` and network):**
- `keyword_match_rate`: all expected keywords appear as whole tokens in the answer. A deterministic **proxy** for
  correctness: it can pass a wrong answer that mentions the number and fail a right answer phrased differently.
- `citation_correct_rate`: at least one verified citation points to the supporting page.
- `false_refusal_rate`: answerable questions where the system declined.
- `correct_fallback_rate`: unanswerable questions where the system declined.
- `failure_rate` (LLM errors) and end-to-end latency.
- `--judge`: an LLM scores faithfulness of each answer to its retrieved context from 1 to 5. This is a **model-based
  estimate**, shares biases with the model family, and is not ground truth.

## Running

```powershell
python -m backend.evaluation.run_evaluation --embedder hashing
python -m backend.evaluation.run_evaluation                       # uses EMBEDDING_BACKEND from .env
python -m backend.evaluation.run_evaluation --generate --judge    # end-to-end
```

Options: `--dataset`, `--ks 1,3,5`, `--output`. The run builds a temporary index, so your real data is untouched.
Results (JSON) are saved to `data/evaluation_results.json`, served by `GET /evaluation`, and shown in the UI.

## Measured results

Run in the development environment on 2026-10-29, offline mode, **hashing embedder (lexical, not semantic)**,
chunk size 800 / overlap 120 (6 chunks indexed):

| Metric | Value |
|---|---|
| Recall@1 | 0.933 |
| Recall@3 | 1.000 |
| Recall@5 | 1.000 |
| MRR | 0.967 |

Not yet measured: sentence-transformers embeddings, any end-to-end metric, any LLM-judged score.

## Limitations

- Tiny fictional dataset; the index has 6 chunks, so Recall@5 is nearly guaranteed. These numbers show the harness
  works. They do not describe real-world retrieval quality.
- Page-level relevance only; no chunk-level labels.
- Detecting real quality drift needs a larger labelled set and repeated runs; the in-app metrics are operational, not
  quality measurements.
