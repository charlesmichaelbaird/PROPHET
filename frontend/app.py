"""PROPHET web dashboard UI (Streamlit)."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pandas as pd
import requests
import streamlit as st

# Ensure repository root is importable when running
# `streamlit run frontend/app.py` from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp_server.server import run_article_count_query, run_ask_the_prophet, run_pipeline
from mcp_server.rag import discover_ollama_models, get_indexing_status, ingest_new_articles
from frontend.btc_data import fetch_btc_history, fetch_spot_btc_price

st.set_page_config(
    page_title="PROPHET | Zero-Cost News Analyzer",
    page_icon="🔮",
    layout="wide",
)

st.markdown(
    """
<style>
:root {
  --bg: #070b13;
  --panel: rgba(17, 26, 45, 0.82);
  --panel-border: rgba(111, 160, 255, 0.23);
  --text: #d9e7ff;
  --muted: #8ea5cb;
}
html, body, [data-testid="stAppViewContainer"], .stApp {
  background: radial-gradient(circle at 10% 10%, #162442 0%, #0a0f1d 35%, #060910 100%);
  color: var(--text);
}
.block-container { padding-top: 1.3rem; max-width: 1500px; }
.hero {
  border: 1px solid var(--panel-border); border-radius: 20px; padding: 1.2rem 1.4rem;
  background: linear-gradient(125deg, rgba(8, 13, 24, 0.95), rgba(20, 31, 56, 0.88)); margin-bottom: 1rem;
}
.brand { letter-spacing: 0.42rem; font-weight: 800; font-size: 1.95rem; margin: 0; }
.subtitle { margin-top: 0.1rem; color: var(--muted); font-size: 0.97rem; }
.meta-chip {
  border: 1px solid rgba(132, 180, 255, 0.3); border-radius: 999px; padding: 0.4rem 0.7rem;
  display: inline-block; margin-right: 0.55rem; margin-top: 0.45rem; color: #d2e6ff;
  background: rgba(40, 57, 90, 0.4); font-size: 0.82rem;
}
.panel {
  border: 1px solid var(--panel-border); border-radius: 16px; padding: 0.95rem 1rem;
  background: var(--panel); margin-bottom: 0.8rem;
}
.panel-title { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.14rem; color: #b6cdff; margin-bottom: 0.55rem; }
.small { color: var(--muted); font-size: 0.87rem; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
  background: #0a1324; border: 1px solid rgba(132, 181, 255, 0.3); color: #d8ebff;
}
.stButton > button {
  border-radius: 10px; border: 1px solid rgba(138, 183, 255, 0.4); background: linear-gradient(130deg, #163769, #1f5f8d);
  color: #eaf5ff; font-weight: 700;
}
[data-testid="stToggle"] [role="switch"][aria-checked="true"] {
  background-color: #1f9d55 !important;
  border-color: #1f9d55 !important;
}
[data-testid="stHorizontalBlock"] [data-baseweb="tab-list"] {
  gap: 0.5rem;
}
[data-baseweb="tab"] {
  background: rgba(16, 26, 45, 0.78);
  border: 1px solid rgba(111, 160, 255, 0.28);
  border-radius: 12px;
  color: #c8ddff;
  padding: 0.45rem 0.8rem;
}
[aria-selected="true"][data-baseweb="tab"] {
  background: linear-gradient(130deg, #14305a, #205985);
}
.source-banner {
  border: 1px solid rgba(111, 160, 255, 0.26); border-radius: 16px; padding: 0.85rem 1rem;
  background: rgba(12, 20, 36, 0.86); margin: 0 auto 1rem auto; max-width: 980px;
}
.source-card {
  border: 1px solid rgba(136, 182, 255, 0.38); border-radius: 14px; padding: 0.85rem;
  background: linear-gradient(130deg, rgba(20, 40, 72, 0.95), rgba(19, 66, 112, 0.78));
  text-align: center; min-height: 124px;
}
.source-card-label { font-weight: 800; font-size: 1.06rem; letter-spacing: 0.03rem; }
.source-card-url { color: #acc6ee; font-size: 0.8rem; margin-top: 0.4rem; }
</style>
""",
    unsafe_allow_html=True,
)

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "btc_runtime_points" not in st.session_state:
    st.session_state.btc_runtime_points = []
if "ask_prophet_answer" not in st.session_state:
    st.session_state.ask_prophet_answer = ""
if "ask_prophet_error" not in st.session_state:
    st.session_state.ask_prophet_error = ""
if "ask_prophet_citations" not in st.session_state:
    st.session_state.ask_prophet_citations = []
if "ask_prophet_engine" not in st.session_state:
    st.session_state.ask_prophet_engine = ""
if "ask_prophet_indexing_triggered" not in st.session_state:
    st.session_state.ask_prophet_indexing_triggered = False
if "ask_prophet_index_verification" not in st.session_state:
    st.session_state.ask_prophet_index_verification = {}
if "ask_prophet_embedding_mode" not in st.session_state:
    st.session_state.ask_prophet_embedding_mode = ""
if "index_data_feedback" not in st.session_state:
    st.session_state.index_data_feedback = {}
if "ollama_process" not in st.session_state:
    st.session_state.ollama_process = None
if "ollama_managed_by_ui" not in st.session_state:
    st.session_state.ollama_managed_by_ui = False
if "ollama_toggle_state" not in st.session_state:
    st.session_state.ollama_toggle_state = False
if "ollama_last_error" not in st.session_state:
    st.session_state.ollama_last_error = ""
if "ollama_runtime_status" not in st.session_state:
    st.session_state.ollama_runtime_status = "idle"
if "ollama_runtime_note" not in st.session_state:
    st.session_state.ollama_runtime_note = ""
if "ollama_autostart_attempted" not in st.session_state:
    st.session_state.ollama_autostart_attempted = False
if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = os.getenv("PROPHET_OLLAMA_HOST", "http://localhost:11434")
if "selected_embedding_model" not in st.session_state:
    st.session_state.selected_embedding_model = ""
if "selected_answer_model" not in st.session_state:
    st.session_state.selected_answer_model = ""
if "ollama_model_discovery" not in st.session_state:
    st.session_state.ollama_model_discovery = {}
if "ap_query_result" not in st.session_state:
    st.session_state.ap_query_result = {}
if "ap_scrape_feedback" not in st.session_state:
    st.session_state.ap_scrape_feedback = ""


AP_NEWS_URL = "https://apnews.com/"
AP_SOURCE_DIRNAME = "apnews-com"
AP_SCRAPE_FALLBACK_MAX_ARTICLES = 200


def _is_ollama_api_alive(host: str) -> bool:
    try:
        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=1.5)
        return response.status_code == 200
    except Exception:
        return False


def _start_ollama_server(host: str) -> tuple[bool, str]:
    if _is_ollama_api_alive(host):
        st.session_state.ollama_managed_by_ui = False
        st.session_state.ollama_last_error = ""
        return True, "Ollama server already running (external process detected)."

    try:
        process = subprocess.Popen(  # noqa: S603
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False, "Could not find 'ollama' in PATH. Install Ollama and reopen the app."
    except Exception as exc:
        return False, f"Failed to start Ollama server: {exc}"

    for _ in range(20):
        if _is_ollama_api_alive(host):
            st.session_state.ollama_process = process
            st.session_state.ollama_managed_by_ui = True
            st.session_state.ollama_last_error = ""
            return True, "Ollama server started by UI control."
        time.sleep(0.25)

    return False, "Ollama process launched, but API did not become reachable at configured host."


def _stop_ollama_server() -> tuple[bool, str]:
    process = st.session_state.ollama_process
    managed = st.session_state.ollama_managed_by_ui
    if not managed or process is None:
        return False, "Ollama appears to be externally managed. Stop it from your terminal if needed."

    if process.poll() is not None:
        st.session_state.ollama_process = None
        st.session_state.ollama_managed_by_ui = False
        return True, "Ollama process was already stopped."

    try:
        process.terminate()
        process.wait(timeout=4)
    except Exception:
        try:
            os.kill(process.pid, signal.SIGKILL)
        except Exception as exc:
            return False, f"Could not stop Ollama process: {exc}"

    st.session_state.ollama_process = None
    st.session_state.ollama_managed_by_ui = False
    return True, "Ollama server stopped."


def _ensure_ollama_runtime_started(host: str) -> None:
    if st.session_state.ollama_autostart_attempted:
        return

    st.session_state.ollama_autostart_attempted = True
    st.session_state.ollama_runtime_status = "starting"
    st.session_state.ollama_runtime_note = "Starting Ollama local runtime..."
    started, message = _start_ollama_server(host)
    if started or _is_ollama_api_alive(host):
        st.session_state.ollama_runtime_status = "online"
        st.session_state.ollama_runtime_note = "Ollama online."
        st.session_state.ollama_last_error = ""
        st.session_state.ollama_toggle_state = True
    else:
        st.session_state.ollama_runtime_status = "failed"
        st.session_state.ollama_runtime_note = "Ollama failed to start."
        st.session_state.ollama_last_error = message
        st.session_state.ollama_toggle_state = False


@st.fragment(run_every=1)
def render_meta_chips() -> None:
    now_utc = datetime.now(timezone.utc)
    st.markdown(
        f'<span class="meta-chip">DATE • {now_utc:%Y-%m-%d}</span>'
        f'<span class="meta-chip">UTC • {now_utc:%H:%M:%S}</span>'
        '<span class="meta-chip">MODE • ZERO-COST + LOCAL LLM READY</span>',
        unsafe_allow_html=True,
    )


def _count_locally_scraped_ap_articles() -> int:
    processed_root = REPO_ROOT / "data" / "processed" / AP_SOURCE_DIRNAME
    if not processed_root.exists():
        return 0
    return sum(1 for _ in processed_root.glob("*/*/metadata.json"))


st.markdown('<section class="hero">', unsafe_allow_html=True)
_ensure_ollama_runtime_started(st.session_state.ollama_host)
hero_left, hero_middle, hero_right = st.columns([2.5, 1.8, 1.2], gap="medium")
with hero_left:
    st.markdown('<p class="brand">PROPHET</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Predictive Reasoning of Probabilistic Hypotheses and Event Tracking</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="small">Zero-cost / no-LLM homepage article analysis and word frequency dashboard.</p>',
        unsafe_allow_html=True,
    )
    render_meta_chips()

with hero_middle:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Local Runtime Models</div>', unsafe_allow_html=True)
    model_discovery = discover_ollama_models(base_url=st.session_state.ollama_host, timeout_seconds=2.5)
    st.session_state.ollama_model_discovery = model_discovery
    available_models = model_discovery.get("models", [])
    embed_candidates = model_discovery.get("embedding_candidates", []) or available_models
    answer_candidates = model_discovery.get("answer_candidates", []) or available_models

    if available_models:
        if st.session_state.selected_embedding_model not in embed_candidates:
            st.session_state.selected_embedding_model = embed_candidates[0]
        if st.session_state.selected_answer_model not in answer_candidates:
            st.session_state.selected_answer_model = answer_candidates[0]
    else:
        st.session_state.selected_embedding_model = ""
        st.session_state.selected_answer_model = ""

    if model_discovery.get("available"):
        st.markdown(
            f'<div class="small">Ollama models discovered: <strong>{len(available_models)}</strong></div>',
            unsafe_allow_html=True,
        )
        st.selectbox(
            "Embedding Model",
            options=embed_candidates,
            index=embed_candidates.index(st.session_state.selected_embedding_model),
            key="selected_embedding_model",
        )
        st.selectbox(
            "Answer Model",
            options=answer_candidates,
            index=answer_candidates.index(st.session_state.selected_answer_model),
            key="selected_answer_model",
        )
    else:
        st.markdown('<div class="small">Ollama unavailable: model dropdowns are disabled.</div>', unsafe_allow_html=True)
        if model_discovery.get("error"):
            st.markdown(f'<div class="small">{model_discovery.get("error")}</div>', unsafe_allow_html=True)
        st.selectbox("Embedding Model", options=["No models available"], index=0, disabled=True)
        st.selectbox("Answer Model", options=["No models available"], index=0, disabled=True)

    index_clicked = st.button(
        "Index Data",
        use_container_width=True,
        disabled=not model_discovery.get("available"),
    )
    if not model_discovery.get("available"):
        st.markdown('<div class="small">Indexing requires local Ollama runtime.</div>', unsafe_allow_html=True)
    if index_clicked:
        with st.spinner("Indexing saved local corpus from /data ..."):
            st.session_state.index_data_feedback = ingest_new_articles(
                embedding_model=st.session_state.selected_embedding_model,
                answer_model=st.session_state.selected_answer_model,
            )

    index_feedback = st.session_state.index_data_feedback
    if index_feedback:
        if not index_feedback.get("error"):
            inspected = index_feedback.get("total_discovered", index_feedback.get("processed_articles_total", 0))
            st.markdown(
                (
                    '<div class="small">Index complete · '
                    f"Inspected: <strong>{inspected}</strong> · "
                    f"Newly indexed: <strong>{index_feedback.get('new_articles_indexed', 0)}</strong> · "
                    f"New chunks: <strong>{index_feedback.get('new_chunks_indexed', 0)}</strong></div>"
                ),
                unsafe_allow_html=True,
            )
        else:
            st.warning(index_feedback.get("error", "Indexing failed."))
    st.markdown('</div>', unsafe_allow_html=True)

with hero_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    configured_host = st.session_state.ollama_host
    alive_now = _is_ollama_api_alive(configured_host)
    if alive_now:
        st.session_state.ollama_runtime_status = "online"
        st.session_state.ollama_runtime_note = "Ollama online."
    elif st.session_state.ollama_runtime_status not in {"starting", "failed"}:
        st.session_state.ollama_runtime_status = "failed"
        st.session_state.ollama_runtime_note = "Ollama failed to start."
    source = "UI-managed" if st.session_state.ollama_managed_by_ui else "External/unknown"
    st.markdown(f'<div class="small">Host: <strong>{configured_host}</strong></div>', unsafe_allow_html=True)
    status = st.session_state.ollama_runtime_status
    if status == "starting":
        st.markdown('<div class="small">Status: <strong>Starting Ollama…</strong></div>', unsafe_allow_html=True)
    elif status == "online":
        st.markdown(f'<div class="small">Status: <strong>Ollama online</strong> ({source})</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="small">Status: <strong>Ollama failed to start</strong></div>', unsafe_allow_html=True)
    if st.session_state.ollama_last_error:
        st.markdown(f'<div class="small">{st.session_state.ollama_last_error}</div>', unsafe_allow_html=True)
    if st.button("Retry Ollama Start", use_container_width=True):
        st.session_state.ollama_autostart_attempted = False
        _ensure_ollama_runtime_started(configured_host)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown("</section>", unsafe_allow_html=True)

st.markdown('<section class="source-banner">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Source Ingestion · AP News</div>', unsafe_allow_html=True)
banner_left, banner_middle, banner_right = st.columns([1.2, 1.6, 1.2], gap="large")
with banner_middle:
    st.markdown(
        (
            '<div class="source-card">'
            '<div style="font-size:1.35rem;">📰</div>'
            '<div class="source-card-label">AP News</div>'
            '<div class="source-card-url">https://apnews.com/</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    query_clicked = st.button("Query Site Article Count", use_container_width=True)
    scrape_clicked = st.button("Data Scrape", use_container_width=True)

    if query_clicked:
        with st.spinner("Querying AP News homepage links..."):
            st.session_state.ap_query_result = run_article_count_query(
                homepage_url=AP_NEWS_URL,
                max_links=220,
            )

    if scrape_clicked:
        latest_query = st.session_state.ap_query_result if st.session_state.ap_query_result.get("ok") == "true" else {}
        requested_scrape_count = int(latest_query.get("links_found", 0)) or AP_SCRAPE_FALLBACK_MAX_ARTICLES
        with st.spinner("Running AP News data scrape..."):
            st.session_state.analysis_result = run_pipeline(
                AP_NEWS_URL,
                requested_scrape_count,
                keyword="",
                keyword_filter_enabled=False,
            )
        result_ok = st.session_state.analysis_result.get("ok") == "true"
        if result_ok:
            scraped = st.session_state.analysis_result.get("articles_scraped", 0)
            attempted = st.session_state.analysis_result.get("articles_attempted", 0)
            st.session_state.ap_scrape_feedback = (
                "AP News scrape complete. "
                f"Requested: {requested_scrape_count} · Attempted: {attempted} · Scraped: {scraped}."
            )
        else:
            st.session_state.ap_scrape_feedback = st.session_state.analysis_result.get("error", "AP News scrape failed.")

    query_result = st.session_state.ap_query_result
    if query_result:
        if query_result.get("ok") == "true":
            st.markdown(
                (
                    '<div class="small">Discovery complete · Candidate AP article links: '
                    f"<strong>{query_result.get('links_found', 0)}</strong></div>"
                ),
                unsafe_allow_html=True,
            )
            preview = query_result.get("preview", [])
            if preview:
                st.markdown('<div class="small">Preview:</div>', unsafe_allow_html=True)
                for item in preview[:5]:
                    st.markdown(f"- [{item.get('title', item.get('url', ''))}]({item.get('url', '')})")
        else:
            st.warning(query_result.get("error", "AP News count query failed."))

    if st.session_state.ap_scrape_feedback:
        st.markdown(f'<div class="small">{st.session_state.ap_scrape_feedback}</div>', unsafe_allow_html=True)
    elif query_result and query_result.get("ok") == "true":
        st.markdown(
            (
                '<div class="small">Data Scrape will target the latest discovered count: '
                f"<strong>{query_result.get('links_found', 0)}</strong> candidate links.</div>"
            ),
            unsafe_allow_html=True,
        )

    scraped_local_count = _count_locally_scraped_ap_articles()
    discovered_count = query_result.get("links_found", 0) if query_result else 0
    st.markdown(
        (
            '<div class="small">AP corpus status · '
            f"Discovered (latest query): <strong>{discovered_count}</strong> · "
            f"Scraped locally: <strong>{scraped_local_count}</strong></div>"
        ),
        unsafe_allow_html=True,
    )
st.markdown("</section>", unsafe_allow_html=True)

view = st.radio(
    "Dashboard View",
    ["PROPHET Dashboard", "BTC/USD Monitor"],
    horizontal=True,
    label_visibility="collapsed",
)

def render_prophet_dashboard() -> None:
    with st.container():
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Ask The Prophet</div>', unsafe_allow_html=True)
        st.caption("Query your scraped corpus with local LLM (Ollama) when available, with extractive fallback when unavailable.")

        ask_question = st.text_input("Ask a corpus-grounded question", value="", key="ask_prophet_question")
        ask_clicked = st.button("Ask The Prophet", use_container_width=True)

        result = st.session_state.analysis_result
        index_status = get_indexing_status()
        model_discovery = st.session_state.ollama_model_discovery or {}
        st.markdown(
            (
                '<div class="small">Local runtime: Ollama + SQLite index · '
                f"Processed articles: <strong>{index_status.get('processed_articles_total', 0)}</strong> · "
                f"Indexed articles: <strong>{index_status.get('indexed_articles_total', 0)}</strong> · "
                f"Up to date: <strong>{'Yes' if index_status.get('is_index_up_to_date') else 'No'}</strong> · "
                f"Ollama available: <strong>{'Yes' if model_discovery.get('available') else 'No'}</strong></div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                '<div class="small">Selected Embedding Model: '
                f"<strong>{st.session_state.selected_embedding_model or 'None'}</strong> · "
                'Selected Answer Model: '
                f"<strong>{st.session_state.selected_answer_model or 'None'}</strong></div>"
            ),
            unsafe_allow_html=True,
        )
        if not model_discovery.get("available") and model_discovery.get("error"):
            st.warning(model_discovery.get("error"))
        if ask_clicked:
            if not ask_question.strip():
                st.session_state.ask_prophet_error = "Please enter a question first."
                st.session_state.ask_prophet_answer = ""
                st.session_state.ask_prophet_citations = []
                st.session_state.ask_prophet_engine = ""
                st.session_state.ask_prophet_indexing_triggered = False
                st.session_state.ask_prophet_index_verification = {}
                st.session_state.ask_prophet_embedding_mode = ""
            else:
                ask_result = run_ask_the_prophet(
                    question=ask_question,
                    article_corpus=(result or {}).get("article_corpus", []),
                    embedding_model=st.session_state.selected_embedding_model,
                    answer_model=st.session_state.selected_answer_model,
                )
                st.session_state.ask_prophet_error = ask_result.get("error", "")
                st.session_state.ask_prophet_answer = ask_result.get("answer", "")
                st.session_state.ask_prophet_citations = ask_result.get("citations", [])
                st.session_state.ask_prophet_engine = ask_result.get("engine", "")
                st.session_state.ask_prophet_indexing_triggered = bool(ask_result.get("indexing_triggered"))
                st.session_state.ask_prophet_index_verification = ask_result.get("index_verification", {})
                st.session_state.ask_prophet_embedding_mode = ask_result.get("embedding_mode", "")

        if st.session_state.ask_prophet_error:
            st.warning(st.session_state.ask_prophet_error)

        if st.session_state.ask_prophet_answer:
            st.markdown("**Answer**")
            engine = st.session_state.ask_prophet_engine
            if engine == "fallback":
                st.markdown('<div class="small">Engine: Extractive fallback (no local model runtime detected)</div>', unsafe_allow_html=True)
            elif engine in {"ollama", "ollama-rag"}:
                st.markdown('<div class="small">Engine: Local Ollama + persistent local retrieval index</div>', unsafe_allow_html=True)
            st.markdown(st.session_state.ask_prophet_answer)
            embedding_mode = st.session_state.ask_prophet_embedding_mode
            if embedding_mode:
                st.markdown(
                    f'<div class="small">Embedding API mode: <strong>{embedding_mode}</strong></div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                (
                    '<div class="small">Answer model used: '
                    f"<strong>{st.session_state.selected_answer_model or 'N/A'}</strong></div>"
                ),
                unsafe_allow_html=True,
            )
            if st.session_state.ask_prophet_indexing_triggered:
                st.markdown(
                    '<div class="small">Pre-answer check: Missing corpus items were indexed before retrieval.</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="small">Answer will appear here after you ask a question.</div>', unsafe_allow_html=True)

        if st.session_state.ask_prophet_citations:
            st.markdown("**Supporting scraped sources**")
            for citation in st.session_state.ask_prophet_citations:
                st.markdown(f"- [{citation['title']}]({citation['url']})")

        st.markdown(
            '<div class="small">Grounding note: responses are constrained to scraped article excerpts and may decline when evidence is insufficient.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


def _build_btc_chart_data(history_rows: list[dict], runtime_points: list[dict]) -> pd.DataFrame:
    history_df = pd.DataFrame(
        {
            "timestamp": [row["timestamp"] for row in history_rows],
            "BTC/USD Price": [row["close"] for row in history_rows],
            "10-day MA": [row["ma_10"] for row in history_rows],
            "30-day MA": [row["ma_30"] for row in history_rows],
            "100-day MA": [row["ma_100"] for row in history_rows],
        }
    )

    if runtime_points:
        runtime_df = pd.DataFrame(
            {
                "timestamp": [point["timestamp"] for point in runtime_points],
                "BTC/USD Price": [point["price"] for point in runtime_points],
                "10-day MA": [history_rows[-1]["ma_10"]] * len(runtime_points),
                "30-day MA": [history_rows[-1]["ma_30"]] * len(runtime_points),
                "100-day MA": [history_rows[-1]["ma_100"]] * len(runtime_points),
            }
        )
        chart_df = pd.concat([history_df, runtime_df], ignore_index=True)
    else:
        chart_df = history_df

    chart_df = chart_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return chart_df.set_index("timestamp")


@st.fragment(run_every=2)
def render_btc_live_view() -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">BTC/USD Monitor</div>', unsafe_allow_html=True)

    try:
        history = fetch_btc_history(limit=420)
    except Exception as exc:
        st.error(f"Historical BTC data unavailable right now: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if len(history) < 100:
        st.error("Not enough historical BTC data to compute moving averages.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    try:
        spot_price, updated_at = fetch_spot_btc_price()
    except Exception as exc:
        st.warning(f"Live BTC quote unavailable right now: {exc}")
        spot_price = history[-1]["close"]
        updated_at = datetime.now(timezone.utc)

    runtime_points = st.session_state.btc_runtime_points
    runtime_points.append({"timestamp": updated_at, "price": spot_price})
    st.session_state.btc_runtime_points = runtime_points[-720:]

    headline_left, headline_mid, headline_right = st.columns([1.1, 1, 1.4])
    headline_left.metric("Current BTC/USD", f"${spot_price:,.2f}")
    headline_mid.metric("10-day MA", f"${history[-1]['ma_10']:,.2f}")
    headline_right.markdown(f'<div class="small">Last update (UTC): <strong>{updated_at:%Y-%m-%d %H:%M:%S}</strong></div>', unsafe_allow_html=True)

    chart_df = _build_btc_chart_data(history, st.session_state.btc_runtime_points)
    st.line_chart(
        chart_df,
        use_container_width=True,
        height=470,
        color=["#5bc0ff", "#8ef8ce", "#f8d66b", "#ff8ca8"],
    )
    st.markdown("</div>", unsafe_allow_html=True)


if view == "PROPHET Dashboard":
    render_prophet_dashboard()
else:
    render_btc_live_view()
