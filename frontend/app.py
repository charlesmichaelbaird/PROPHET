"""PROPHET web dashboard UI (Streamlit)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import subprocess
import sys
import threading
import time

import pandas as pd
import requests
import streamlit as st

# Ensure repository root is importable when running
# `streamlit run frontend/app.py` from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp_server.server import (
    run_article_count_query_by_date,
    run_ask_the_prophet,
    run_pipeline_by_date,
    run_stop_pipeline_by_date,
)
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
  padding-top: 0.22rem; padding-bottom: 0.22rem; line-height: 1.05; min-height: 1.9rem;
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
if "ask_prophet_diagnostics" not in st.session_state:
    st.session_state.ask_prophet_diagnostics = {}
if "ask_prophet_status" not in st.session_state:
    st.session_state.ask_prophet_status = "idle"
if "ask_prophet_started_at" not in st.session_state:
    st.session_state.ask_prophet_started_at = 0.0
if "ask_prophet_elapsed_seconds" not in st.session_state:
    st.session_state.ask_prophet_elapsed_seconds = 0
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
if "bbc_query_result" not in st.session_state:
    st.session_state.bbc_query_result = {}
if "bbc_scrape_feedback" not in st.session_state:
    st.session_state.bbc_scrape_feedback = ""
if "aj_query_result" not in st.session_state:
    st.session_state.aj_query_result = {}
if "aj_scrape_feedback" not in st.session_state:
    st.session_state.aj_scrape_feedback = ""
if "pp_query_result" not in st.session_state:
    st.session_state.pp_query_result = {}
if "propublica_scrape_feedback" not in st.session_state:
    st.session_state.propublica_scrape_feedback = ""
if "ap_selected_date" not in st.session_state:
    st.session_state.ap_selected_date = datetime.now(timezone.utc).strftime("%m/%d/%Y")
if "bbc_selected_date" not in st.session_state:
    st.session_state.bbc_selected_date = datetime.now(timezone.utc).strftime("%m/%d/%Y")
if "aj_selected_date" not in st.session_state:
    st.session_state.aj_selected_date = datetime.now(timezone.utc).strftime("%m/%d/%Y")
if "pp_selected_date" not in st.session_state:
    st.session_state.pp_selected_date = datetime.now(timezone.utc).strftime("%m/%d/%Y")
if "ap_scrape_active" not in st.session_state:
    st.session_state.ap_scrape_active = False
if "bbc_scrape_active" not in st.session_state:
    st.session_state.bbc_scrape_active = False
if "aj_scrape_active" not in st.session_state:
    st.session_state.aj_scrape_active = False
if "propublica_scrape_active" not in st.session_state:
    st.session_state.propublica_scrape_active = False
if "ap_scrape_thread" not in st.session_state:
    st.session_state.ap_scrape_thread = None
if "bbc_scrape_thread" not in st.session_state:
    st.session_state.bbc_scrape_thread = None
if "aj_scrape_thread" not in st.session_state:
    st.session_state.aj_scrape_thread = None
if "propublica_scrape_thread" not in st.session_state:
    st.session_state.propublica_scrape_thread = None
if "ap_scrape_queue" not in st.session_state:
    st.session_state.ap_scrape_queue = Queue()
if "bbc_scrape_queue" not in st.session_state:
    st.session_state.bbc_scrape_queue = Queue()
if "aj_scrape_queue" not in st.session_state:
    st.session_state.aj_scrape_queue = Queue()
if "pp_scrape_queue" not in st.session_state:
    st.session_state.pp_scrape_queue = Queue()
if "ap_stop_requested" not in st.session_state:
    st.session_state.ap_stop_requested = False
if "bbc_stop_requested" not in st.session_state:
    st.session_state.bbc_stop_requested = False
if "aj_stop_requested" not in st.session_state:
    st.session_state.aj_stop_requested = False
if "propublica_stop_requested" not in st.session_state:
    st.session_state.propublica_stop_requested = False
if "ap_scrape_started_at" not in st.session_state:
    st.session_state.ap_scrape_started_at = 0.0
if "bbc_scrape_started_at" not in st.session_state:
    st.session_state.bbc_scrape_started_at = 0.0
if "aj_scrape_started_at" not in st.session_state:
    st.session_state.aj_scrape_started_at = 0.0
if "propublica_scrape_started_at" not in st.session_state:
    st.session_state.propublica_scrape_started_at = 0.0
if "ap_scrape_status" not in st.session_state:
    st.session_state.ap_scrape_status = "idle"
if "bbc_scrape_status" not in st.session_state:
    st.session_state.bbc_scrape_status = "idle"
if "aj_scrape_status" not in st.session_state:
    st.session_state.aj_scrape_status = "idle"
if "propublica_scrape_status" not in st.session_state:
    st.session_state.propublica_scrape_status = "idle"
if "ap_scrape_progress" not in st.session_state:
    st.session_state.ap_scrape_progress = {}
if "bbc_scrape_progress" not in st.session_state:
    st.session_state.bbc_scrape_progress = {}
if "aj_scrape_progress" not in st.session_state:
    st.session_state.aj_scrape_progress = {}
if "propublica_scrape_progress" not in st.session_state:
    st.session_state.propublica_scrape_progress = {}
if "ap_last_elapsed_seconds" not in st.session_state:
    st.session_state.ap_last_elapsed_seconds = 0
if "bbc_last_elapsed_seconds" not in st.session_state:
    st.session_state.bbc_last_elapsed_seconds = 0
if "aj_last_elapsed_seconds" not in st.session_state:
    st.session_state.aj_last_elapsed_seconds = 0
if "propublica_last_elapsed_seconds" not in st.session_state:
    st.session_state.propublica_last_elapsed_seconds = 0
if "ap_index_feedback" not in st.session_state:
    st.session_state.ap_index_feedback = {}
if "bbc_index_feedback" not in st.session_state:
    st.session_state.bbc_index_feedback = {}
if "aj_index_feedback" not in st.session_state:
    st.session_state.aj_index_feedback = {}
if "pp_index_feedback" not in st.session_state:
    st.session_state.pp_index_feedback = {}
if "last_live_refresh_at" not in st.session_state:
    st.session_state.last_live_refresh_at = 0.0


AP_SOURCE_DIRNAME = "apnews-com"
AP_SCRAPE_FALLBACK_MAX_ARTICLES = 200
BBC_SOURCE_DIRNAME = "www-bbc-com"
BBC_SCRAPE_FALLBACK_MAX_ARTICLES = 200
AP_INDEX_PARTITION = "ap-news"
BBC_INDEX_PARTITION = "bbc"
AJ_SOURCE_DIRNAME = "www-aljazeera-com"
AJ_SCRAPE_FALLBACK_MAX_ARTICLES = 200
AJ_INDEX_PARTITION = "aljazeera-com"
PP_SOURCE_DIRNAME = "propublica"
PP_SCRAPE_FALLBACK_MAX_ARTICLES = 200
PP_INDEX_PARTITION = "propublica"


def _start_date_scrape_worker(source_name: str, date_str: str, max_articles: int, queue_key: str) -> None:
    queue_obj = st.session_state[queue_key]

    def _runner() -> None:
        def _progress(event: dict) -> None:
            queue_obj.put({"kind": "progress", "source": source_name, "event": event})

        result = run_pipeline_by_date(
            source_name=source_name,
            date_str=date_str,
            max_articles=max_articles,
            progress_callback=_progress,
        )
        queue_obj.put({"kind": "result", "source": source_name, "payload": result})

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    st.session_state[f"{source_name}_scrape_thread"] = thread
    st.session_state[f"{source_name}_scrape_active"] = True
    st.session_state[f"{source_name}_stop_requested"] = False
    st.session_state[f"{source_name}_scrape_started_at"] = time.time()
    st.session_state[f"{source_name}_scrape_status"] = "querying/discovering"
    st.session_state[f"{source_name}_scrape_progress"] = {
        "articles_attempted": 0,
        "articles_scraped": 0,
        "articles_failed": 0,
        "links_found": 0,
        "progress_pct": 0,
    }


def _poll_date_scrape_result(source_name: str, queue_key: str) -> dict | None:
    queue_obj = st.session_state[queue_key]
    result_payload = None
    while True:
        try:
            message = queue_obj.get_nowait()
        except Empty:
            break
        if isinstance(message, dict) and message.get("kind") == "progress":
            event = message.get("event", {})
            progress = st.session_state.get(f"{source_name}_scrape_progress", {}).copy()
            progress.update(
                {
                    "articles_attempted": int(event.get("articles_attempted", progress.get("articles_attempted", 0))),
                    "articles_scraped": int(event.get("articles_scraped", progress.get("articles_scraped", 0))),
                    "articles_failed": int(event.get("articles_failed", progress.get("articles_failed", 0))),
                    "links_found": int(event.get("links_found", progress.get("links_found", 0))),
                }
            )
            links_found = max(int(progress.get("links_found", 0)), 0)
            attempted = max(int(progress.get("articles_attempted", 0)), 0)
            max_articles = max(int(event.get("max_articles", 0)), 0)
            known_total = links_found if links_found > 0 else max_articles
            if known_total > 0:
                progress["progress_pct"] = min(int((attempted / known_total) * 100), 100)
            else:
                progress["progress_pct"] = 0
            st.session_state[f"{source_name}_scrape_progress"] = progress

            event_type = str(event.get("event", "")).strip().lower()
            if event_type == "discover_complete":
                st.session_state[f"{source_name}_scrape_status"] = "scraping"
            elif event_type == "stopped":
                st.session_state[f"{source_name}_scrape_status"] = "stopped"
            elif event_type in {"article_started", "article_done", "article_failed"}:
                st.session_state[f"{source_name}_scrape_status"] = "scraping"
        elif isinstance(message, dict) and message.get("kind") == "result":
            result_payload = message.get("payload", {})
    return result_payload


def _format_elapsed(seconds: int) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _ask_status_label(status: str) -> str:
    labels = {
        "idle": "idle",
        "running": "running",
        "completed": "completed",
        "failed": "failed",
    }
    return labels.get(status, status or "idle")


def _render_scrape_loading_bar(source_name: str) -> None:
    started_at = float(st.session_state.get(f"{source_name}_scrape_started_at", 0.0) or 0.0)
    elapsed_seconds = int(max(time.time() - started_at, 0.0)) if started_at else 0
    st.session_state[f"{source_name}_last_elapsed_seconds"] = elapsed_seconds
    progress = st.session_state.get(f"{source_name}_scrape_progress", {})
    pct = int(progress.get("progress_pct", 0))
    attempted = int(progress.get("articles_attempted", 0))
    scraped = int(progress.get("articles_scraped", 0))
    failed = int(progress.get("articles_failed", 0))
    links_found = int(progress.get("links_found", 0))
    st.progress(min(max(pct, 0), 100))
    total_text = f"/ {links_found}" if links_found > 0 else ""
    st.markdown(
        (
            f'<div class="small">Status: <strong>{st.session_state.get(f"{source_name}_scrape_status", "scraping")}</strong>'
            f' · Progress: <strong>{attempted}{total_text}</strong>'
            f' · Scraped: <strong>{scraped}</strong>'
            f' · Failed: <strong>{failed}</strong>'
            f' · Elapsed: <strong>{_format_elapsed(elapsed_seconds)}</strong></div>'
        ),
        unsafe_allow_html=True,
    )


def _run_source_indexing(source_partition: str, source_label: str) -> dict:
    progress_slot = st.empty()
    progress_bar = progress_slot.progress(0)
    progress_label = st.empty()
    progress_state = {"total": 0, "indexed": 0}

    def _index_progress(event: dict) -> None:
        event_name = str(event.get("event", "")).strip().lower()
        if event_name == "scan":
            progress_state["total"] = int(event.get("eligible_for_indexing", 0) or 0)
            progress_state["indexed"] = 0
        elif event_name == "article_done":
            progress_state["total"] = int(event.get("total_to_index", progress_state.get("total", 0)) or 0)
            progress_state["indexed"] = int(event.get("indexed_so_far", progress_state.get("indexed", 0)) or 0)
        elif event_name == "step":
            progress_state["total"] = int(event.get("total_to_index", progress_state.get("total", 0)) or 0)
            article_position = int(event.get("article_position", 0) or 0)
            progress_state["indexed"] = max(progress_state["indexed"], max(article_position - 1, 0))

        total = max(int(progress_state["total"]), 0)
        indexed = max(int(progress_state["indexed"]), 0)
        remaining = max(total - indexed, 0)
        pct = int((indexed / total) * 100) if total > 0 else 100
        progress_bar.progress(min(max(pct, 0), 100))
        progress_label.markdown(
            (
                f'<div class="small">{source_label} indexing: '
                f'<strong>{indexed}</strong> / <strong>{total}</strong> indexed'
                f' · Remaining: <strong>{remaining}</strong></div>'
            ),
            unsafe_allow_html=True,
        )

    with st.spinner(f"Indexing {source_label} local corpus..."):
        feedback = ingest_new_articles(
            embedding_model=st.session_state.selected_embedding_model,
            answer_model=st.session_state.selected_answer_model,
            source_partition=source_partition,
            progress_callback=_index_progress,
        )

    total_indexed = int(feedback.get("new_articles_indexed", 0))
    total_eligible = int(feedback.get("eligible_for_indexing", total_indexed))
    remaining = max(total_eligible - total_indexed, 0)
    progress_bar.progress(100 if total_eligible == 0 else min(int((total_indexed / total_eligible) * 100), 100))
    progress_label.markdown(
        (
            f'<div class="small">{source_label} indexing: '
            f'<strong>{total_indexed}</strong> / <strong>{total_eligible}</strong> indexed'
            f' · Remaining: <strong>{remaining}</strong></div>'
        ),
        unsafe_allow_html=True,
    )
    return feedback


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


def _count_locally_scraped_bbc_articles() -> int:
    processed_root = REPO_ROOT / "data" / "processed" / BBC_SOURCE_DIRNAME
    if not processed_root.exists():
        return 0
    return sum(1 for _ in processed_root.glob("*/*/metadata.json"))


def _count_locally_scraped_aj_articles() -> int:
    processed_root = REPO_ROOT / "data" / "processed" / AJ_SOURCE_DIRNAME
    if not processed_root.exists():
        return 0
    return sum(1 for _ in processed_root.glob("*/*/metadata.json"))


def _count_locally_scraped_pp_articles() -> int:
    processed_root = REPO_ROOT / "data" / "processed" / PP_SOURCE_DIRNAME
    if not processed_root.exists():
        return 0
    return sum(1 for _ in processed_root.glob("*/*/metadata.json"))


def _indexed_count_for_source(index_status_payload: dict, source_partitions: list[str]) -> int:
    source_keys = [key.strip().lower() for key in source_partitions if key.strip()]
    source_counts = index_status_payload.get("indexed_counts_by_source", {}) or {}
    manifest_root = Path(str(index_status_payload.get("active_model_index_root", "")).strip())
    manifest_count = 0
    if manifest_root.exists():
        for source_key in source_keys:
            manifest_path = manifest_root / source_key / "vector_manifest.json"
            if not manifest_path.exists():
                continue
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            articles = payload.get("articles", payload.get("indexed_articles", {}))
            if isinstance(articles, dict):
                manifest_count += len(articles)
    if manifest_count > 0:
        return manifest_count
    return sum(int(source_counts.get(source_key, 0) or 0) for source_key in source_keys)


def _format_bbc_user_error(error_message: str) -> str:
    normalized = (error_message or "").lower()
    if any(token in normalized for token in ("blocked", "forbidden", "http 401", "http 403", "http 429")):
        return (
            "BBC blocked this automated request. "
            "Try again later; if blocking persists, an alternate discovery path/source may be needed."
        )
    if "no candidate article links" in normalized or "no article links" in normalized:
        return (
            "BBC discovery succeeded but no article links were detected from the current discovery pages. "
            "Try again later or use an alternate BBC discovery path."
        )
    return error_message or "BBC request failed."


def _format_aj_user_error(error_message: str) -> str:
    normalized = (error_message or "").lower()
    if "query failed" in normalized or "could not be parsed or fetched" in normalized:
        return (
            "Al Jazeera archive query failed (fetch/parser issue). "
            "This is different from zero results; check diagnostics and retry later."
        )
    if "returned no article links" in normalized:
        return "Al Jazeera discovery ran but found 0 English candidate links for the selected date."
    return error_message or "Al Jazeera request failed."


st.markdown('<section class="hero">', unsafe_allow_html=True)
_ensure_ollama_runtime_started(st.session_state.ollama_host)
model_discovery = discover_ollama_models(base_url=st.session_state.ollama_host, timeout_seconds=2.5)
st.session_state.ollama_model_discovery = model_discovery
hero_left, hero_middle, hero_right = st.columns([2.5, 1.8, 1.2], gap="medium")
with hero_left:
    st.markdown('<p class="brand">PROPHET</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Predictive Reasoning of Probable Hypotheses and Event Tracking</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="small">Zero-cost / no-LLM homepage article analysis and word frequency dashboard.</p>',
        unsafe_allow_html=True,
    )
    render_meta_chips()
    index_status = get_indexing_status(embedding_model=st.session_state.selected_embedding_model)
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
    st.markdown(
        (
            '<div class="small">Active model index root: '
            f"<strong>{index_status.get('active_model_index_root', 'N/A')}</strong></div>"
        ),
        unsafe_allow_html=True,
    )
    source_counts = index_status.get("indexed_counts_by_source", {}) or {}
    if source_counts:
        st.markdown(
            '<div class="small">Indexed by source (active model): '
            + " · ".join(f"{source}: {count}" for source, count in sorted(source_counts.items()))
            + "</div>",
            unsafe_allow_html=True,
        )

with hero_middle:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Local Runtime Models</div>', unsafe_allow_html=True)
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

    st.markdown('</div>', unsafe_allow_html=True)

with hero_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    configured_host = st.session_state.ollama_host
    alive_now = _is_ollama_api_alive(configured_host)
    toggle_label = "Server Online" if alive_now else "Server Offline"
    toggle_value = st.toggle(
        toggle_label,
        value=alive_now,
        help="Toggle local Ollama runtime on/off.",
    )
    if toggle_value != alive_now:
        if toggle_value:
            started, message = _start_ollama_server(configured_host)
            alive_now = started or _is_ollama_api_alive(configured_host)
            if alive_now:
                st.session_state.ollama_runtime_status = "online"
                st.session_state.ollama_runtime_note = "Ollama online."
                st.session_state.ollama_last_error = ""
            else:
                st.session_state.ollama_runtime_status = "failed"
                st.session_state.ollama_runtime_note = "Ollama failed to start."
                st.session_state.ollama_last_error = message
        else:
            stopped, message = _stop_ollama_server()
            alive_now = _is_ollama_api_alive(configured_host)
            if stopped and not alive_now:
                st.session_state.ollama_runtime_status = "failed"
                st.session_state.ollama_runtime_note = "Ollama failed to start."
                st.session_state.ollama_last_error = ""
            elif not stopped:
                st.session_state.ollama_last_error = message
                st.warning(message)

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
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown("</section>", unsafe_allow_html=True)

st.markdown('<section class="source-banner">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Source Ingestion · AP News + BBC + Al Jazeera English + ProPublica</div>', unsafe_allow_html=True)
banner_left, banner_middle, banner_right = st.columns([0.5, 3.2, 0.5], gap="large")
with banner_middle:
    ap_col, bbc_col, aj_col, pp_col = st.columns(4, gap="large")

with ap_col:
    st.markdown(
        (
            '<div class="source-card">'
            '<div style="font-size:1.35rem;">📰</div>'
            '<div class="source-card-label">AP News</div>'
            '<div class="source-card-url">Date-based archive discovery</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    ap_query_date_col, ap_query_button_col = st.columns([1.25, 1], gap="small")
    with ap_query_date_col:
        st.text_input(
            "Date (MM/DD/YYYY)",
            key="ap_selected_date",
            label_visibility="collapsed",
            placeholder="MM/DD/YYYY",
        )
    with ap_query_button_col:
        query_clicked = st.button("Article Count", use_container_width=True, key="ap_query_btn")

    ap_scrape_active = bool(st.session_state.ap_scrape_active)
    ap_action_col_left, ap_action_col_right = st.columns(2, gap="small")
    with ap_action_col_left:
        scrape_clicked = st.button(
            "Stop Scraping" if ap_scrape_active else "Scrape Data",
            use_container_width=True,
            key="ap_data_scrape_btn",
        )
    with ap_action_col_right:
        ap_index_clicked = st.button(
            "Index Data",
            use_container_width=True,
            key="ap_index_data_btn",
            disabled=not model_discovery.get("available"),
        )

    if query_clicked:
        st.session_state.ap_scrape_status = "querying/discovering"
        with st.spinner("Querying AP News archive metadata for selected date..."):
            st.session_state.ap_query_result = run_article_count_query_by_date(
                source_name="ap",
                date_str=st.session_state.ap_selected_date,
                max_links=250,
            )

    if scrape_clicked and not ap_scrape_active:
        latest_query = st.session_state.ap_query_result if st.session_state.ap_query_result.get("ok") == "true" else {}
        requested_scrape_count = int(latest_query.get("links_found", 0)) or AP_SCRAPE_FALLBACK_MAX_ARTICLES
        st.session_state.ap_scrape_feedback = "AP News scrape active..."
        _start_date_scrape_worker(
            source_name="ap",
            date_str=st.session_state.ap_selected_date,
            max_articles=requested_scrape_count,
            queue_key="ap_scrape_queue",
        )
    elif scrape_clicked and ap_scrape_active:
        run_stop_pipeline_by_date(source_name="ap")
        st.session_state.ap_stop_requested = True
        st.session_state.ap_scrape_status = "stopped"
        st.session_state.ap_scrape_feedback = "Stopping AP News scrape..."

    if ap_index_clicked:
        st.session_state.ap_index_feedback = _run_source_indexing(
            source_partition=AP_INDEX_PARTITION,
            source_label="AP News",
        )

    ap_result = _poll_date_scrape_result("ap", "ap_scrape_queue")
    if ap_result is not None:
        st.session_state.analysis_result = ap_result
        st.session_state.ap_scrape_active = False
        st.session_state.ap_scrape_thread = None
        if st.session_state.ap_scrape_started_at:
            st.session_state.ap_last_elapsed_seconds = int(time.time() - st.session_state.ap_scrape_started_at)
        st.session_state.ap_scrape_started_at = 0.0
        ap_ok = ap_result.get("ok") == "true"
        if ap_ok:
            scraped = ap_result.get("articles_scraped", 0)
            attempted = ap_result.get("articles_attempted", 0)
            if ap_result.get("scrape_stopped_by_user"):
                st.session_state.ap_scrape_status = "stopped"
                st.session_state.ap_scrape_feedback = (
                    "Scrape stopped by user. "
                    f"Attempted: {attempted} · Scraped: {scraped} · Elapsed: {_format_elapsed(st.session_state.ap_last_elapsed_seconds)}."
                )
            else:
                st.session_state.ap_scrape_status = "completed"
                st.session_state.ap_scrape_feedback = (
                    "AP News scrape complete. "
                    f"Attempted: {attempted} · Scraped: {scraped} · Elapsed: {_format_elapsed(st.session_state.ap_last_elapsed_seconds)}."
                )
        else:
            st.session_state.ap_scrape_status = "idle"
            st.session_state.ap_scrape_feedback = ap_result.get("error", "AP News scrape failed.")

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

    st.markdown(
        f'<div class="small">Scrape status: <strong>{st.session_state.ap_scrape_status}</strong></div>',
        unsafe_allow_html=True,
    )
    if st.session_state.ap_scrape_active:
        _render_scrape_loading_bar("ap")
    if st.session_state.ap_scrape_feedback:
        st.markdown(f'<div class="small">{st.session_state.ap_scrape_feedback}</div>', unsafe_allow_html=True)
    elif query_result and query_result.get("ok") == "true":
        st.markdown(
            (
                '<div class="small">Scrape Data will target the latest discovered count: '
                f"<strong>{query_result.get('links_found', 0)}</strong> candidate links.</div>"
            ),
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="small">Selected date: <strong>{st.session_state.ap_selected_date}</strong></div>',
        unsafe_allow_html=True,
    )

    scraped_local_count = _count_locally_scraped_ap_articles()
    discovered_count = query_result.get("links_found", 0) if query_result else 0
    ap_indexed_count = _indexed_count_for_source(index_status, [AP_INDEX_PARTITION, AP_SOURCE_DIRNAME])
    st.markdown(
        (
            '<div class="small">AP corpus status · '
            f"Discovered (latest query): <strong>{discovered_count}</strong> · "
            f"Scraped locally: <strong>{scraped_local_count}</strong> · "
            f"Indexed: <strong>{ap_indexed_count}</strong></div>"
        ),
        unsafe_allow_html=True,
    )
    ap_index_feedback = st.session_state.ap_index_feedback
    if ap_index_feedback:
        ap_eligible = int(ap_index_feedback.get("eligible_for_indexing", ap_index_feedback.get("new_articles_indexed", 0)))
        ap_indexed = int(ap_index_feedback.get("new_articles_indexed", 0))
        st.markdown(
            (
                '<div class="small">AP index update · '
                f"Indexed: <strong>{ap_indexed}</strong> / <strong>{ap_eligible}</strong> · "
                f"Remaining: <strong>{max(ap_eligible - ap_indexed, 0)}</strong></div>"
            ),
            unsafe_allow_html=True,
        )

with bbc_col:
    st.markdown(
        (
            '<div class="source-card">'
            '<div style="font-size:1.35rem;">🌐</div>'
            '<div class="source-card-label">BBC</div>'
            '<div class="source-card-url">Date-based archive discovery</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    bbc_query_date_col, bbc_query_button_col = st.columns([1.25, 1], gap="small")
    with bbc_query_date_col:
        st.text_input(
            "Date (MM/DD/YYYY)",
            key="bbc_selected_date",
            label_visibility="collapsed",
            placeholder="MM/DD/YYYY",
        )
    with bbc_query_button_col:
        bbc_query_clicked = st.button(
            "Article Count",
            key="bbc_query_site_article_count",
            use_container_width=True,
        )

    bbc_scrape_active = bool(st.session_state.bbc_scrape_active)
    bbc_action_col_left, bbc_action_col_right = st.columns(2, gap="small")
    with bbc_action_col_left:
        bbc_scrape_clicked = st.button(
            "Stop Scraping" if bbc_scrape_active else "Scrape Data",
            key="bbc_data_scrape",
            use_container_width=True,
        )
    with bbc_action_col_right:
        bbc_index_clicked = st.button(
            "Index Data",
            key="bbc_index_data_btn",
            use_container_width=True,
            disabled=not model_discovery.get("available"),
        )

    if bbc_query_clicked:
        st.session_state.bbc_scrape_status = "querying/discovering"
        with st.spinner("Querying BBC archive metadata for selected date..."):
            st.session_state.bbc_query_result = run_article_count_query_by_date(
                source_name="bbc",
                date_str=st.session_state.bbc_selected_date,
                max_links=250,
            )

    if bbc_scrape_clicked and not bbc_scrape_active:
        latest_bbc_query = (
            st.session_state.bbc_query_result if st.session_state.bbc_query_result.get("ok") == "true" else {}
        )
        requested_bbc_scrape_count = (
            int(latest_bbc_query.get("links_found", 0)) or BBC_SCRAPE_FALLBACK_MAX_ARTICLES
        )
        st.session_state.bbc_scrape_feedback = "BBC scrape active..."
        _start_date_scrape_worker(
            source_name="bbc",
            date_str=st.session_state.bbc_selected_date,
            max_articles=requested_bbc_scrape_count,
            queue_key="bbc_scrape_queue",
        )
    elif bbc_scrape_clicked and bbc_scrape_active:
        run_stop_pipeline_by_date(source_name="bbc")
        st.session_state.bbc_stop_requested = True
        st.session_state.bbc_scrape_status = "stopped"
        st.session_state.bbc_scrape_feedback = "Stopping BBC scrape..."

    if bbc_index_clicked:
        st.session_state.bbc_index_feedback = _run_source_indexing(
            source_partition=BBC_INDEX_PARTITION,
            source_label="BBC",
        )

    bbc_result = _poll_date_scrape_result("bbc", "bbc_scrape_queue")
    if bbc_result is not None:
        st.session_state.analysis_result = bbc_result
        st.session_state.bbc_scrape_active = False
        st.session_state.bbc_scrape_thread = None
        if st.session_state.bbc_scrape_started_at:
            st.session_state.bbc_last_elapsed_seconds = int(time.time() - st.session_state.bbc_scrape_started_at)
        st.session_state.bbc_scrape_started_at = 0.0
        bbc_ok = bbc_result.get("ok") == "true"
        if bbc_ok:
            bbc_scraped = bbc_result.get("articles_scraped", 0)
            bbc_attempted = bbc_result.get("articles_attempted", 0)
            if bbc_result.get("scrape_stopped_by_user"):
                st.session_state.bbc_scrape_status = "stopped"
                st.session_state.bbc_scrape_feedback = (
                    "Scrape stopped by user. "
                    f"Attempted: {bbc_attempted} · Scraped: {bbc_scraped} · Elapsed: {_format_elapsed(st.session_state.bbc_last_elapsed_seconds)}."
                )
            else:
                st.session_state.bbc_scrape_status = "completed"
                st.session_state.bbc_scrape_feedback = (
                    "BBC scrape complete. "
                    f"Attempted: {bbc_attempted} · Scraped: {bbc_scraped} · Elapsed: {_format_elapsed(st.session_state.bbc_last_elapsed_seconds)}."
                )
        else:
            st.session_state.bbc_scrape_status = "idle"
            st.session_state.bbc_scrape_feedback = _format_bbc_user_error(
                bbc_result.get("error", "BBC scrape failed.")
            )

    bbc_query_result = st.session_state.bbc_query_result
    if bbc_query_result:
        if bbc_query_result.get("ok") == "true":
            st.markdown(
                (
                    '<div class="small">Discovery complete · Candidate BBC article links: '
                    f"<strong>{bbc_query_result.get('links_found', 0)}</strong></div>"
                ),
                unsafe_allow_html=True,
            )
            bbc_preview = bbc_query_result.get("preview", [])
            if bbc_preview:
                st.markdown('<div class="small">Preview:</div>', unsafe_allow_html=True)
                for item in bbc_preview[:5]:
                    st.markdown(f"- [{item.get('title', item.get('url', ''))}]({item.get('url', '')})")
        else:
            st.warning(_format_bbc_user_error(bbc_query_result.get("error", "BBC count query failed.")))

    st.markdown(
        f'<div class="small">Scrape status: <strong>{st.session_state.bbc_scrape_status}</strong></div>',
        unsafe_allow_html=True,
    )
    if st.session_state.bbc_scrape_active:
        _render_scrape_loading_bar("bbc")
    if st.session_state.bbc_scrape_feedback:
        st.markdown(f'<div class="small">{st.session_state.bbc_scrape_feedback}</div>', unsafe_allow_html=True)
    elif bbc_query_result and bbc_query_result.get("ok") == "true":
        st.markdown(
            (
                '<div class="small">Scrape Data will target the latest discovered count: '
                f"<strong>{bbc_query_result.get('links_found', 0)}</strong> candidate links.</div>"
            ),
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="small">Selected date: <strong>{st.session_state.bbc_selected_date}</strong></div>',
        unsafe_allow_html=True,
    )

    bbc_scraped_local_count = _count_locally_scraped_bbc_articles()
    bbc_discovered_count = bbc_query_result.get("links_found", 0) if bbc_query_result else 0
    bbc_indexed_count = _indexed_count_for_source(index_status, [BBC_INDEX_PARTITION, BBC_SOURCE_DIRNAME, "bbc-com"])
    st.markdown(
        (
            '<div class="small">BBC corpus status · '
            f"Discovered (latest query): <strong>{bbc_discovered_count}</strong> · "
            f"Scraped locally: <strong>{bbc_scraped_local_count}</strong> · "
            f"Indexed: <strong>{bbc_indexed_count}</strong></div>"
        ),
        unsafe_allow_html=True,
    )
    bbc_index_feedback = st.session_state.bbc_index_feedback
    if bbc_index_feedback:
        bbc_eligible = int(bbc_index_feedback.get("eligible_for_indexing", bbc_index_feedback.get("new_articles_indexed", 0)))
        bbc_indexed = int(bbc_index_feedback.get("new_articles_indexed", 0))
        st.markdown(
            (
                '<div class="small">BBC index update · '
                f"Indexed: <strong>{bbc_indexed}</strong> / <strong>{bbc_eligible}</strong> · "
                f"Remaining: <strong>{max(bbc_eligible - bbc_indexed, 0)}</strong></div>"
            ),
            unsafe_allow_html=True,
        )

with aj_col:
    st.markdown(
        (
            '<div class="source-card">'
            '<div style="font-size:1.35rem;">🛰️</div>'
            '<div class="source-card-label">Al Jazeera English</div>'
            '<div class="source-card-url">Date-based archive discovery</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    aj_query_date_col, aj_query_button_col = st.columns([1.25, 1], gap="small")
    with aj_query_date_col:
        st.text_input(
            "Date (MM/DD/YYYY)",
            key="aj_selected_date",
            label_visibility="collapsed",
            placeholder="MM/DD/YYYY",
        )
    with aj_query_button_col:
        aj_query_clicked = st.button(
            "Article Count",
            key="aj_query_site_article_count",
            use_container_width=True,
        )

    aj_scrape_active = bool(st.session_state.aj_scrape_active)
    aj_action_col_left, aj_action_col_right = st.columns(2, gap="small")
    with aj_action_col_left:
        aj_scrape_clicked = st.button(
            "Stop Scraping" if aj_scrape_active else "Scrape Data",
            key="aj_data_scrape",
            use_container_width=True,
        )
    with aj_action_col_right:
        aj_index_clicked = st.button(
            "Index Data",
            key="aj_index_data_btn",
            use_container_width=True,
            disabled=not model_discovery.get("available"),
        )

    if aj_query_clicked:
        st.session_state.aj_scrape_status = "querying/discovering"
        with st.spinner("Querying Al Jazeera archive metadata for selected date..."):
            st.session_state.aj_query_result = run_article_count_query_by_date(
                source_name="aljazeera",
                date_str=st.session_state.aj_selected_date,
                max_links=250,
            )

    if aj_scrape_clicked and not aj_scrape_active:
        latest_aj_query = st.session_state.aj_query_result if st.session_state.aj_query_result.get("ok") == "true" else {}
        requested_aj_scrape_count = int(latest_aj_query.get("links_found", 0)) or AJ_SCRAPE_FALLBACK_MAX_ARTICLES
        st.session_state.aj_scrape_feedback = "Al Jazeera scrape active..."
        _start_date_scrape_worker(
            source_name="aljazeera",
            date_str=st.session_state.aj_selected_date,
            max_articles=requested_aj_scrape_count,
            queue_key="aj_scrape_queue",
        )
    elif aj_scrape_clicked and aj_scrape_active:
        run_stop_pipeline_by_date(source_name="aljazeera")
        st.session_state.aj_stop_requested = True
        st.session_state.aj_scrape_status = "stopped"
        st.session_state.aj_scrape_feedback = "Stopping Al Jazeera scrape..."

    if aj_index_clicked:
        st.session_state.aj_index_feedback = _run_source_indexing(
            source_partition=AJ_INDEX_PARTITION,
            source_label="Al Jazeera English",
        )

    aj_result = _poll_date_scrape_result("aljazeera", "aj_scrape_queue")
    if aj_result is not None:
        st.session_state.analysis_result = aj_result
        st.session_state.aj_scrape_active = False
        st.session_state.aj_scrape_thread = None
        if st.session_state.aj_scrape_started_at:
            st.session_state.aj_last_elapsed_seconds = int(time.time() - st.session_state.aj_scrape_started_at)
        st.session_state.aj_scrape_started_at = 0.0
        aj_ok = aj_result.get("ok") == "true"
        if aj_ok:
            aj_scraped = aj_result.get("articles_scraped", 0)
            aj_attempted = aj_result.get("articles_attempted", 0)
            if aj_result.get("scrape_stopped_by_user"):
                st.session_state.aj_scrape_status = "stopped"
                st.session_state.aj_scrape_feedback = (
                    "Scrape stopped by user. "
                    f"Attempted: {aj_attempted} · Scraped: {aj_scraped} · Elapsed: {_format_elapsed(st.session_state.aj_last_elapsed_seconds)}."
                )
            else:
                st.session_state.aj_scrape_status = "completed"
                st.session_state.aj_scrape_feedback = (
                    "Al Jazeera scrape complete. "
                    f"Attempted: {aj_attempted} · Scraped: {aj_scraped} · Elapsed: {_format_elapsed(st.session_state.aj_last_elapsed_seconds)}."
                )
        else:
            st.session_state.aj_scrape_status = "idle"
            st.session_state.aj_scrape_feedback = aj_result.get("error", "Al Jazeera scrape failed.")

    aj_query_result = st.session_state.aj_query_result
    if aj_query_result:
        if aj_query_result.get("ok") == "true":
            aj_links_found = int(aj_query_result.get("links_found", 0))
            if aj_links_found > 0:
                st.markdown(
                    (
                        '<div class="small">Discovery complete · Candidate Al Jazeera links: '
                        f"<strong>{aj_links_found}</strong></div>"
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.info(aj_query_result.get("status_message", "No Al Jazeera results for selected date."))
            aj_preview = aj_query_result.get("preview", [])
            if aj_preview:
                st.markdown('<div class="small">Preview:</div>', unsafe_allow_html=True)
                for item in aj_preview[:5]:
                    st.markdown(f"- [{item.get('title', item.get('url', ''))}]({item.get('url', '')})")
        else:
            st.warning(_format_aj_user_error(aj_query_result.get("error", "Al Jazeera count query failed.")))

    st.markdown(
        f'<div class="small">Scrape status: <strong>{st.session_state.aj_scrape_status}</strong></div>',
        unsafe_allow_html=True,
    )
    if st.session_state.aj_scrape_active:
        _render_scrape_loading_bar("aljazeera")
    if st.session_state.aj_scrape_feedback:
        st.markdown(f'<div class="small">{st.session_state.aj_scrape_feedback}</div>', unsafe_allow_html=True)
    elif aj_query_result and aj_query_result.get("ok") == "true":
        st.markdown(
            (
                '<div class="small">Scrape Data will target the latest discovered count: '
                f"<strong>{aj_query_result.get('links_found', 0)}</strong> candidate links.</div>"
            ),
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="small">Selected date: <strong>{st.session_state.aj_selected_date}</strong></div>',
        unsafe_allow_html=True,
    )

    aj_scraped_local_count = _count_locally_scraped_aj_articles()
    aj_discovered_count = aj_query_result.get("links_found", 0) if aj_query_result else 0
    aj_indexed_count = _indexed_count_for_source(index_status, [AJ_INDEX_PARTITION, AJ_SOURCE_DIRNAME])
    st.markdown(
        (
            '<div class="small">Al Jazeera corpus status · '
            f"Discovered (latest query): <strong>{aj_discovered_count}</strong> · "
            f"Scraped locally: <strong>{aj_scraped_local_count}</strong> · "
            f"Indexed: <strong>{aj_indexed_count}</strong></div>"
        ),
        unsafe_allow_html=True,
    )
    aj_index_feedback = st.session_state.aj_index_feedback
    if aj_index_feedback:
        aj_eligible = int(aj_index_feedback.get("eligible_for_indexing", aj_index_feedback.get("new_articles_indexed", 0)))
        aj_indexed = int(aj_index_feedback.get("new_articles_indexed", 0))
        st.markdown(
            (
                '<div class="small">Al Jazeera index update · '
                f"Indexed: <strong>{aj_indexed}</strong> / <strong>{aj_eligible}</strong> · "
                f"Remaining: <strong>{max(aj_eligible - aj_indexed, 0)}</strong></div>"
            ),
            unsafe_allow_html=True,
        )

with pp_col:
    st.markdown(
        (
            '<div class="source-card">'
            '<div style="font-size:1.35rem;">🔎</div>'
            '<div class="source-card-label">ProPublica</div>'
            '<div class="source-card-url">Date-based sitemap discovery (English only)</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    pp_query_date_col, pp_query_button_col = st.columns([1.25, 1], gap="small")
    with pp_query_date_col:
        st.text_input(
            "Date (MM/DD/YYYY)",
            key="pp_selected_date",
            label_visibility="collapsed",
            placeholder="MM/DD/YYYY",
        )
    with pp_query_button_col:
        pp_query_clicked = st.button(
            "Article Count",
            key="pp_query_site_article_count",
            use_container_width=True,
        )

    pp_scrape_active = bool(st.session_state.propublica_scrape_active)
    pp_action_col_left, pp_action_col_right = st.columns(2, gap="small")
    with pp_action_col_left:
        pp_scrape_clicked = st.button(
            "Stop Scraping" if pp_scrape_active else "Scrape Data",
            key="pp_data_scrape",
            use_container_width=True,
        )
    with pp_action_col_right:
        pp_index_clicked = st.button(
            "Index Data",
            key="pp_index_data_btn",
            use_container_width=True,
            disabled=not model_discovery.get("available"),
        )

    if pp_query_clicked:
        st.session_state.propublica_scrape_status = "querying/discovering"
        with st.spinner("Querying ProPublica archive metadata for selected date..."):
            st.session_state.pp_query_result = run_article_count_query_by_date(
                source_name="propublica",
                date_str=st.session_state.pp_selected_date,
                max_links=250,
            )

    if pp_scrape_clicked and not pp_scrape_active:
        latest_pp_query = st.session_state.pp_query_result if st.session_state.pp_query_result.get("ok") == "true" else {}
        requested_pp_scrape_count = int(latest_pp_query.get("links_found", 0)) or PP_SCRAPE_FALLBACK_MAX_ARTICLES
        st.session_state.propublica_scrape_feedback = "ProPublica scrape active..."
        _start_date_scrape_worker(
            source_name="propublica",
            date_str=st.session_state.pp_selected_date,
            max_articles=requested_pp_scrape_count,
            queue_key="pp_scrape_queue",
        )
    elif pp_scrape_clicked and pp_scrape_active:
        run_stop_pipeline_by_date(source_name="propublica")
        st.session_state.propublica_stop_requested = True
        st.session_state.propublica_scrape_status = "stopped"
        st.session_state.propublica_scrape_feedback = "Stopping ProPublica scrape..."

    if pp_index_clicked:
        st.session_state.pp_index_feedback = _run_source_indexing(
            source_partition=PP_INDEX_PARTITION,
            source_label="ProPublica",
        )

    pp_result = _poll_date_scrape_result("propublica", "pp_scrape_queue")
    if pp_result is not None:
        st.session_state.analysis_result = pp_result
        st.session_state.propublica_scrape_active = False
        st.session_state.propublica_scrape_thread = None
        if st.session_state.propublica_scrape_started_at:
            st.session_state.propublica_last_elapsed_seconds = int(time.time() - st.session_state.propublica_scrape_started_at)
        st.session_state.propublica_scrape_started_at = 0.0
        pp_ok = pp_result.get("ok") == "true"
        if pp_ok:
            pp_scraped = pp_result.get("articles_scraped", 0)
            pp_attempted = pp_result.get("articles_attempted", 0)
            if pp_result.get("scrape_stopped_by_user"):
                st.session_state.propublica_scrape_status = "stopped"
                st.session_state.propublica_scrape_feedback = (
                    "Scrape stopped by user. "
                    f"Attempted: {pp_attempted} · Scraped: {pp_scraped} · Elapsed: {_format_elapsed(st.session_state.propublica_last_elapsed_seconds)}."
                )
            else:
                st.session_state.propublica_scrape_status = "completed"
                st.session_state.propublica_scrape_feedback = (
                    "ProPublica scrape complete. "
                    f"Attempted: {pp_attempted} · Scraped: {pp_scraped} · Elapsed: {_format_elapsed(st.session_state.propublica_last_elapsed_seconds)}."
                )
        else:
            st.session_state.propublica_scrape_status = "idle"
            st.session_state.propublica_scrape_feedback = pp_result.get("error", "ProPublica scrape failed.")

    pp_query_result = st.session_state.pp_query_result
    if pp_query_result:
        if pp_query_result.get("ok") == "true":
            links_found = int(pp_query_result.get("links_found", 0))
            if links_found > 0:
                st.markdown(
                    (
                        '<div class="small">Discovery complete · Candidate ProPublica links: '
                        f"<strong>{links_found}</strong></div>"
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.info(pp_query_result.get("status_message", "No ProPublica results for selected date."))
            pp_preview = pp_query_result.get("preview", [])
            if pp_preview:
                st.markdown('<div class="small">Preview:</div>', unsafe_allow_html=True)
                for item in pp_preview[:5]:
                    st.markdown(f"- [{item.get('title', item.get('url', ''))}]({item.get('url', '')})")
        else:
            st.warning(pp_query_result.get("error", "ProPublica count query failed."))

    st.markdown(
        f'<div class="small">Scrape status: <strong>{st.session_state.propublica_scrape_status}</strong></div>',
        unsafe_allow_html=True,
    )
    if st.session_state.propublica_scrape_active:
        _render_scrape_loading_bar("propublica")
    if st.session_state.propublica_scrape_feedback:
        st.markdown(f'<div class="small">{st.session_state.propublica_scrape_feedback}</div>', unsafe_allow_html=True)
    elif pp_query_result and pp_query_result.get("ok") == "true":
        st.markdown(
            (
                '<div class="small">Scrape Data will target the latest discovered count: '
                f"<strong>{pp_query_result.get('links_found', 0)}</strong> candidate links.</div>"
            ),
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="small">Selected date: <strong>{st.session_state.pp_selected_date}</strong></div>',
        unsafe_allow_html=True,
    )

    pp_scraped_local_count = _count_locally_scraped_pp_articles()
    pp_discovered_count = pp_query_result.get("links_found", 0) if pp_query_result else 0
    pp_indexed_count = _indexed_count_for_source(index_status, [PP_INDEX_PARTITION, PP_SOURCE_DIRNAME, "propublica-org"])
    st.markdown(
        (
            '<div class="small">ProPublica corpus status · '
            f"Discovered (latest query): <strong>{pp_discovered_count}</strong> · "
            f"Scraped locally: <strong>{pp_scraped_local_count}</strong> · "
            f"Indexed: <strong>{pp_indexed_count}</strong></div>"
        ),
        unsafe_allow_html=True,
    )
    pp_index_feedback = st.session_state.pp_index_feedback
    if pp_index_feedback:
        pp_eligible = int(pp_index_feedback.get("eligible_for_indexing", pp_index_feedback.get("new_articles_indexed", 0)))
        pp_indexed = int(pp_index_feedback.get("new_articles_indexed", 0))
        st.markdown(
            (
                '<div class="small">ProPublica index update · '
                f"Indexed: <strong>{pp_indexed}</strong> / <strong>{pp_eligible}</strong> · "
                f"Remaining: <strong>{max(pp_eligible - pp_indexed, 0)}</strong></div>"
            ),
            unsafe_allow_html=True,
        )
st.markdown("</section>", unsafe_allow_html=True)

if (
    st.session_state.ap_scrape_active
    or st.session_state.bbc_scrape_active
    or st.session_state.aj_scrape_active
    or st.session_state.propublica_scrape_active
):
    now_ts = time.time()
    if now_ts - float(st.session_state.last_live_refresh_at or 0.0) >= 1.0:
        st.session_state.last_live_refresh_at = now_ts
        st.rerun()

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
        model_discovery = st.session_state.ollama_model_discovery or {}
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
                st.session_state.ask_prophet_diagnostics = {}
                st.session_state.ask_prophet_status = "idle"
                st.session_state.ask_prophet_started_at = 0.0
                st.session_state.ask_prophet_elapsed_seconds = 0
            else:
                st.session_state.ask_prophet_status = "running"
                st.session_state.ask_prophet_started_at = time.time()
                st.session_state.ask_prophet_elapsed_seconds = 0
                with st.spinner("Running Ask The Prophet (embed → retrieve → generate)…"):
                    ask_result = run_ask_the_prophet(
                        question=ask_question,
                        article_corpus=(result or {}).get("article_corpus", []),
                        embedding_model=st.session_state.selected_embedding_model,
                        answer_model=st.session_state.selected_answer_model,
                    )
                st.session_state.ask_prophet_elapsed_seconds = int(
                    max(time.time() - float(st.session_state.ask_prophet_started_at or time.time()), 0.0)
                )
                st.session_state.ask_prophet_error = ask_result.get("error", "")
                st.session_state.ask_prophet_answer = ask_result.get("answer", "")
                st.session_state.ask_prophet_citations = ask_result.get("citations", [])
                st.session_state.ask_prophet_engine = ask_result.get("engine", "")
                st.session_state.ask_prophet_indexing_triggered = bool(ask_result.get("indexing_triggered"))
                st.session_state.ask_prophet_index_verification = ask_result.get("index_verification", {})
                st.session_state.ask_prophet_embedding_mode = ask_result.get("embedding_mode", "")
                st.session_state.ask_prophet_diagnostics = ask_result.get("diagnostics", {})
                st.session_state.ask_prophet_status = "failed" if st.session_state.ask_prophet_error else "completed"

        st.markdown(
            (
                '<div class="small">Ask status: '
                f"<strong>{_ask_status_label(st.session_state.ask_prophet_status)}</strong> · "
                f"Elapsed: <strong>{_format_elapsed(st.session_state.ask_prophet_elapsed_seconds)}</strong></div>"
            ),
            unsafe_allow_html=True,
        )

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
            diagnostics = st.session_state.ask_prophet_diagnostics or {}
            if diagnostics:
                st.markdown(
                    (
                        '<div class="small">Ask runtime: '
                        f"stage=<strong>{diagnostics.get('stage', 'unknown')}</strong> · "
                        f"retrieved=<strong>{int(diagnostics.get('retrieval_count', 0))}</strong> · "
                        f"top score=<strong>{float(diagnostics.get('top_score', 0.0)):.3f}</strong> · "
                        f"embed/retrieve/generate ms=<strong>"
                        f"{int(diagnostics.get('embed_ms', 0))}/"
                        f"{int(diagnostics.get('retrieval_ms', 0))}/"
                        f"{int(diagnostics.get('chat_ms', 0))}</strong> · "
                        f"total ms=<strong>{int(diagnostics.get('total_ms', 0))}</strong></div>"
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
        index_verification = st.session_state.ask_prophet_index_verification or {}
        if index_verification:
            st.markdown(
                (
                    '<div class="small">Index verification: '
                    f"processed=<strong>{int(index_verification.get('processed_articles_total', 0))}</strong> · "
                    f"indexed=<strong>{int(index_verification.get('indexed_articles_total', 0))}</strong> · "
                    f"missing=<strong>{int(index_verification.get('missing_articles_total', 0))}</strong></div>"
                ),
                unsafe_allow_html=True,
            )

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
