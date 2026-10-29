"""Evaluation results from GET /evaluation (produced by python -m backend.evaluation.run_evaluation)."""

import pandas as pd
import streamlit as st
from common import call, show_error, sidebar

st.set_page_config(page_title="Evaluation", page_icon="🧪", layout="wide")
sidebar()
st.title("Evaluation")

res = call("GET", "/evaluation")
if res.code == "no_evaluation":
    st.info("No evaluation results yet.")
    st.markdown("Run the offline retrieval evaluation from the project root, then refresh:")
    st.code("python -m backend.evaluation.run_evaluation", language="powershell")
    st.markdown(
        "Add `--generate` for end-to-end answers (needs `OPENAI_API_KEY`) and `--embedder hashing` "
        "to run without downloading a model."
    )
    st.stop()
if not res.ok:
    show_error(res)
    st.stop()

r = res.data
emb = r["embedder"]
st.caption(f"Run at {r['generated_at']} · mode: **{r['mode']}** · embedder: `{emb['name']}`")
if not emb["semantic"]:
    st.warning(
        "This run used the lexical hashing embedder, which matches shared words only. "
        "These numbers do NOT measure semantic retrieval quality."
    )

ret = r["retrieval"]
st.subheader("Retrieval (deterministic)")
cols = st.columns(len(ret["recall_at_k"]) + 2)
for col, (k, v) in zip(cols, ret["recall_at_k"].items(), strict=False):
    col.metric(f"Recall@{k}", f"{v:.3f}")
cols[-2].metric("MRR", f"{ret['mrr']:.3f}")
cols[-1].metric("Mean retrieval ms", ret["latency_ms"]["mean"])

gen = r.get("generation")
st.subheader("Answer generation")
if gen:
    cols = st.columns(4)
    cols[0].metric("Keyword match", f"{gen['keyword_match_rate']:.3f}")
    cols[1].metric("Citation correct", f"{gen['citation_correct_rate']:.3f}")
    cols[2].metric("Failure rate", f"{gen['failure_rate']:.3f}")
    cols[3].metric("Correct fallback", gen["correct_fallback_rate"])
    if gen.get("faithfulness_estimate_mean") is not None:
        st.write(f"LLM-judged faithfulness (estimate, 1-5): **{gen['faithfulness_estimate_mean']}**")
    st.dataframe(pd.DataFrame(gen["per_question"]), width="stretch")
else:
    st.info("Generation was not evaluated in this run (offline mode). Use `--generate` with a configured API key.")

with st.expander("Per-question retrieval ranks"):
    st.dataframe(pd.DataFrame(ret["per_question"]), width="stretch")
d = r["dataset"]
st.caption(
    f"Dataset: {d['questions']} answerable + {d['unanswerable']} unanswerable questions · "
    f"{d['indexed_chunks']} chunks · chunk size {r['config']['chunk_size']}"
)
st.subheader("Limitations")
for cav in r["caveats"]:
    st.markdown(f"- {cav}")
