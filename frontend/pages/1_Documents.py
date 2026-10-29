"""Upload, list and delete documents."""

import streamlit as st
from common import call, show_error, sidebar

st.set_page_config(page_title="Documents", page_icon="📚", layout="wide")
sidebar()
st.title("Documents")

st.subheader("Upload PDFs")
files = st.file_uploader("Choose one or more PDF files", type=["pdf"], accept_multiple_files=True)
if st.button("Index selected files", disabled=not files, type="primary"):
    payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
    with st.spinner("Extracting, embedding and indexing..."):
        res = call("POST", "/documents/upload", files=payload)
    results = (res.data or {}).get("results") if isinstance(res.data, dict) else None
    if results:
        for item in results:
            if item["status"] == "indexed":
                st.success(
                    f"{item['filename']}: indexed {item['pages']} page(s), "
                    f"{item['chunk_count']} chunk(s) in {item['ingestion_ms']} ms"
                )
            else:
                st.error(f"{item['filename']}: {item['error']} (code: {item['error_code']})")
    if not res.ok and not results:
        show_error(res)

st.subheader("Indexed documents")
if st.button("Refresh"):
    st.rerun()
docs = call("GET", "/documents")
if not docs.ok:
    show_error(docs)
elif not docs.data:
    st.info("No documents indexed yet.")
else:
    header = st.columns([4, 1, 1, 2, 1])
    for col, label in zip(header, ["File", "Pages", "Chunks", "Document ID", ""], strict=True):
        col.markdown(f"**{label}**")
    for d in docs.data:
        c = st.columns([4, 1, 1, 2, 1])
        c[0].write(d["filename"])
        c[1].write(d["pages"])
        c[2].write(d["chunk_count"])
        c[3].code(d["document_id"])
        if c[4].button("Delete", key=f"del_{d['document_id']}"):
            res = call("DELETE", f"/documents/{d['document_id']}")
            if res.ok:
                st.rerun()
            else:
                show_error(res)
