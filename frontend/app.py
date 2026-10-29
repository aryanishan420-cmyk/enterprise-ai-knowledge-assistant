"""Enterprise AI Knowledge Assistant: chat page. Run: streamlit run frontend/app.py"""

import streamlit as st
from common import call, show_error, sidebar

st.set_page_config(page_title="Enterprise AI Knowledge Assistant", page_icon="📄", layout="wide")
sidebar()

st.title("Enterprise AI Knowledge Assistant")
st.caption(
    "Ask questions about your uploaded PDFs. Answers are generated only from retrieved passages "
    "and cite the source document and page."
)

health = call("GET", "/health", timeout=5)
with st.sidebar:
    st.header("Status")
    if health.ok:
        h = health.data
        st.success("Backend connected")
        st.write(f"Indexed chunks: **{h['indexed_chunks']}**")
        st.write(f"Embedder: `{h['embedder']}`")
        st.write("LLM key: " + ("configured" if h["llm_configured"] else "**not configured** (search only)"))
    else:
        st.error("Backend unavailable")
    st.header("Retrieval settings")
    mode = st.radio("Mode", ["Ask (RAG answer)", "Search only"])
    top_k = st.slider("Top-k passages", 1, 20, 5)
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

if not health.ok:
    show_error(health)
    st.stop()
if health.data["indexed_chunks"] == 0:
    st.info("No documents are indexed yet. Upload a PDF on the **Documents** page first.")

st.session_state.setdefault("messages", [])


def render_passages(passages: list[dict]) -> None:
    with st.expander(f"Retrieved passages ({len(passages)})"):
        for p in passages:
            st.markdown(f"**[{p['label']}] {p['filename']}, page {p['page']}** · similarity {p['score']:.3f}")
            st.text(p["text"])


def render(msg: dict) -> None:
    if msg["role"] == "user":
        st.markdown(msg["content"])
        return
    if msg.get("kind") == "search":
        st.markdown(f"Found **{len(msg['passages'])}** passage(s) in {msg['retrieval_ms']} ms.")
        render_passages(msg["passages"])
        return
    st.markdown(msg["content"])
    if msg.get("insufficient"):
        st.warning("The documents do not contain enough information to answer this question.")
    if msg.get("citations"):
        st.markdown(
            "**Sources:** " + "; ".join(f"[{c['label']}] {c['filename']}, p. {c['page']}" for c in msg["citations"])
        )
    if msg.get("removed"):
        st.caption(f"Removed unverifiable citation(s) invented by the model: {', '.join(msg['removed'])}")
    if msg.get("passages"):
        render_passages(msg["passages"])
    st.caption(f"retrieval {msg['retrieval_ms']} ms · LLM {msg['llm_ms']} ms · total {msg['total_ms']} ms")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        render(m)

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            if mode == "Search only":
                res = call("POST", "/search", json={"query": question, "top_k": top_k})
            else:
                res = call("POST", "/chat", json={"query": question, "top_k": top_k})
        if not res.ok:
            show_error(res)
        elif mode == "Search only":
            msg = {
                "role": "assistant",
                "kind": "search",
                "passages": res.data["results"],
                "retrieval_ms": res.data["retrieval_ms"],
            }
            st.session_state.messages.append(msg)
            render(msg)
        else:
            d = res.data
            msg = {
                "role": "assistant",
                "kind": "chat",
                "content": d["answer"],
                "insufficient": d["insufficient_context"],
                "citations": d["citations"],
                "removed": d["invalid_citations_removed"],
                "passages": d["retrieved"],
                "retrieval_ms": d["retrieval_ms"],
                "llm_ms": d["llm_ms"],
                "total_ms": d["total_ms"],
            }
            st.session_state.messages.append(msg)
            render(msg)
