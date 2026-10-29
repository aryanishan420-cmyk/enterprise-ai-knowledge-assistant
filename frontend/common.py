"""Shared helpers for the Streamlit pages: backend connection settings and a safe API client."""

import os
from dataclasses import dataclass
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


@dataclass
class ApiResult:
    ok: bool
    status: int | None = None
    data: Any = None
    code: str | None = None
    message: str | None = None


def sidebar() -> None:
    """Render connection settings (shared by all pages through session state)."""
    st.session_state.setdefault("api_base", DEFAULT_API_BASE)
    st.session_state.setdefault("api_token", os.getenv("ADMIN_TOKEN", ""))
    with st.sidebar:
        st.header("Backend connection")
        st.text_input("API base URL", key="api_base", help="FastAPI server, e.g. http://127.0.0.1:8000")
        st.text_input(
            "X-API-Token (optional)",
            key="api_token",
            type="password",
            help="Only needed when the backend has ADMIN_TOKEN set.",
        )


def call(method: str, path: str, *, timeout: float = 120.0, **kwargs: Any) -> ApiResult:
    base = st.session_state.get("api_base", DEFAULT_API_BASE).rstrip("/")
    headers = {"X-API-Token": st.session_state["api_token"]} if st.session_state.get("api_token") else {}
    try:
        r = httpx.request(method, f"{base}{path}", headers=headers, timeout=timeout, **kwargs)
    except httpx.ConnectError:
        return ApiResult(
            False,
            None,
            code="unreachable",
            message=f"Cannot reach the backend at {base}. Start it with: "
            "uvicorn backend.api.main:create_app --factory --port 8000",
        )
    except httpx.TimeoutException:
        return ApiResult(False, None, code="timeout", message="The backend did not respond in time.")
    except httpx.HTTPError as exc:
        return ApiResult(False, None, code="http_error", message=f"Request failed: {type(exc).__name__}")
    try:
        body = r.json()
    except ValueError:
        body = None
    if r.is_success:
        return ApiResult(True, r.status_code, body)
    err = (body or {}).get("error", {}) if isinstance(body, dict) else {}
    return ApiResult(
        False,
        r.status_code,
        body,
        err.get("code", "error"),
        err.get("message", f"Backend returned HTTP {r.status_code}."),
    )


def show_error(res: ApiResult) -> None:
    """Consistent, actionable error display."""
    if res.code == "not_configured":
        st.error(
            f"{res.message}\n\nCreate a `.env` file (see `.env.example`), set `OPENAI_API_KEY`, "
            "and restart the backend. You can still use **Search only** mode without a key."
        )
    elif res.code == "unauthorized":
        st.error("The backend requires an API token. Enter it as X-API-Token in the sidebar.")
    elif res.code == "unreachable":
        st.error(res.message)
    else:
        st.error(f"{res.message} (code: {res.code}, HTTP {res.status})")
