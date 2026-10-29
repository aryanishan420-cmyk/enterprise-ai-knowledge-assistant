"""Operational metrics from GET /metrics (in-process; reset when the backend restarts)."""

import pandas as pd
import streamlit as st
from common import call, show_error, sidebar

st.set_page_config(page_title="Monitoring", page_icon="📈", layout="wide")
sidebar()
st.title("Monitoring")
st.caption(
    "Metrics are collected in memory by the backend and reset on restart. "
    "No document text or question text is recorded."
)
st.button("Refresh")

res = call("GET", "/metrics")
if not res.ok:
    show_error(res)
    st.stop()

m = res.data
c = m.get("counters", {})
cols = st.columns(4)
cols[0].metric("Requests", c.get("requests_total", 0))
cols[1].metric("Failed requests", c.get("requests_failed", 0))
cols[2].metric("Documents ingested", c.get("documents_ingested", 0))
cols[3].metric("Ingestion failures", c.get("ingestion_failures", 0))
cols = st.columns(4)
cols[0].metric("Indexed documents", m.get("index", {}).get("documents", 0))
cols[1].metric("Indexed chunks", m.get("index", {}).get("chunks", 0))
cols[2].metric("Queries logged", m.get("queries_logged", 0))
cols[3].metric("Insufficient-context answers", c.get("insufficient_context", 0))

lat = m.get("latency_ms", {})
st.subheader("Latency (ms)")
if lat:
    st.dataframe(pd.DataFrame(lat).T[["count", "mean", "p50", "p95", "max"]], width="stretch")
else:
    st.info("No latency data yet. Make some requests first.")

if c.get("prompt_tokens") or c.get("completion_tokens"):
    st.subheader("LLM token usage (as reported by the provider)")
    st.write(f"Prompt: {c.get('prompt_tokens', 0)} · Completion: {c.get('completion_tokens', 0)}")

endpoints = {k.removeprefix("endpoint:"): v for k, v in c.items() if k.startswith("endpoint:")}
if endpoints:
    st.subheader("Requests by endpoint")
    st.bar_chart(pd.Series(endpoints, name="requests"))
