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

from mcp_server.server import run_ask_the_prophet, run_pipeline
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
.block-container { padding-top: 1.3rem; max-width: 1500px; padding-left: 0.55rem; }
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
.control-strip {
  padding: 0.1rem 0.1rem 0.25rem;
  margin-left: -0.6rem;
}
.control-strip .panel-title {
  font-size: 0.68rem;
  letter-spacing: 0.11rem;
  margin-bottom: 0.45rem;
}
.runtime-card {
  border: 1px solid rgba(110, 164, 255, 0.24);
  border-radius: 10px;
  padding: 0.5rem;
  margin-bottom: 0.55rem;
  background: rgba(13, 23, 41, 0.8);
}
.runtime-card-head {
  font-size: 0.62rem;
  letter-spacing: 0.08rem;
  color: #95b6ec;
  text-transform: uppercase;
  margin-bottom: 0.4rem;
}
.runtime-status {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border: 1px solid rgba(102, 160, 255, 0.28);
  border-radius: 999px;
  padding: 0.16rem 0.42rem;
  font-size: 0.63rem;
  margin-bottom: 0.4rem;
  background: rgba(7, 15, 31, 0.76);
}
.runtime-dot {
  width: 0.46rem;
  height: 0.46rem;
  border-radius: 999px;
  display: inline-block;
  margin-right: 0.28rem;
  box-shadow: 0 0 0.5rem currentColor;
}
.runtime-state-online {
  color: #70ffc2;
}
.runtime-state-offline {
  color: #ff7b9b;
}
.control-strip [data-testid="stButton"] button {
  border-radius: 8px;
  font-size: 0.72rem;
  letter-spacing: 0.05rem;
  border: 1px solid rgba(118, 174, 255, 0.35);
  width: 100%;
}
.control-strip [data-testid="stButton"] button[kind="secondary"] {
  background: rgba(11, 21, 38, 0.86);
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
if "ollama_process" not in st.session_state:
    st.session_state.ollama_process = None
if "ollama_managed_by_ui" not in st.session_state:
    st.session_state.ollama_managed_by_ui = False
if "ollama_toggle_state" not in st.session_state:
    st.session_state.ollama_toggle_state = False
if "ollama_last_error" not in st.session_state:
    st.session_state.ollama_last_error = ""
if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = os.getenv("PROPHET_OLLAMA_HOST", "http://localhost:11434")


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


@st.fragment(run_every=1)
def render_meta_chips() -> None:
    now_utc = datetime.now(timezone.utc)
    st.markdown(
        f'<span class="meta-chip">DATE • {now_utc:%Y-%m-%d}</span>'
        f'<span class="meta-chip">UTC • {now_utc:%H:%M:%S}</span>'
        '<span class="meta-chip">MODE • ZERO-COST + LOCAL LLM READY</span>',
        unsafe_allow_html=True,
    )

st.markdown('<section class="hero">', unsafe_allow_html=True)
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
st.markdown("</section>", unsafe_allow_html=True)

left_strip, main_content = st.columns([0.52, 5.48], gap="medium")
with left_strip:
    st.markdown('<div class="control-strip">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Local Runtime Control</div>', unsafe_allow_html=True)
    configured_host = st.session_state.ollama_host
    alive_now = _is_ollama_api_alive(configured_host)
    runtime_state = "ONLINE" if alive_now else "OFFLINE"
    runtime_class = "runtime-state-online" if alive_now else "runtime-state-offline"
    runtime_source = "UI" if st.session_state.ollama_managed_by_ui else "EXT"
    runtime_button_bg = "linear-gradient(135deg, #431724, #7a2337)" if not alive_now else "linear-gradient(135deg, #12362b, #1f785b)"
    st.markdown(
        f"""
        <style>
        .control-strip [data-testid="stButton"] button {{
          background: {runtime_button_bg};
          color: #e9f4ff;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="runtime-card">
          <div class="runtime-card-head">Ollama Runtime</div>
          <div class="runtime-status">
            <span><span class="runtime-dot {runtime_class}"></span>{runtime_state}</span>
            <span>{runtime_source}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    runtime_button_label = "running" if alive_now else "offline"
    runtime_button_clicked = st.button(
        runtime_button_label,
        key="ollama_runtime_button",
        use_container_width=True,
        help="Start/stop local Ollama runtime for Ask The Prophet.",
    )

    if runtime_button_clicked:
        if alive_now:
            stopped, message = _stop_ollama_server()
            if stopped:
                st.session_state.ollama_toggle_state = False
                st.success(message)
            else:
                st.session_state.ollama_toggle_state = _is_ollama_api_alive(configured_host)
                st.warning(message)
        else:
            started, message = _start_ollama_server(configured_host)
            st.session_state.ollama_last_error = "" if started else message
            st.session_state.ollama_toggle_state = started or _is_ollama_api_alive(configured_host)
            if started:
                st.success(message)
            else:
                st.error(message)

    alive_now = _is_ollama_api_alive(configured_host)
    source = "UI-managed" if st.session_state.ollama_managed_by_ui else "External/unknown"
    st.markdown(f'<div class="small">Host · <strong>{configured_host}</strong></div>', unsafe_allow_html=True)
    if alive_now:
        st.markdown(f'<div class="small">Status · <strong>Running</strong> ({source})</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="small">Status · <strong>Stopped</strong></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with main_content:
    view = st.radio(
        "Dashboard View",
        ["PROPHET Dashboard", "BTC/USD Monitor"],
        horizontal=True,
        label_visibility="collapsed",
    )

def render_prophet_dashboard() -> None:
    left, right = st.columns([1.05, 1.6], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Operator Console</div>', unsafe_allow_html=True)
        st.caption("Provide a homepage (e.g., https://apnews.com), scrape article links, and analyze top words.")

        homepage_url = st.text_input("Source Homepage URL", value="https://apnews.com")
        max_articles = st.number_input("Max articles to scrape", min_value=5, max_value=50, value=20, step=5)
        keyword_filter_enabled = st.checkbox("Enable keyword filtering", value=False)
        keyword = st.text_input("Keyword (article-level match)", value="")

        run_clicked = st.button("Run Zero-Cost Analysis", use_container_width=True)
        clear_clicked = st.button("Reset Results", use_container_width=True)

        if clear_clicked:
            st.session_state.analysis_result = None

        if run_clicked:
            with st.spinner("Fetching homepage and scraping articles..."):
                st.session_state.analysis_result = run_pipeline(
                    homepage_url.strip(),
                    int(max_articles),
                    keyword=keyword.strip(),
                    keyword_filter_enabled=keyword_filter_enabled,
                )

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Ask The Prophet</div>', unsafe_allow_html=True)
        st.caption("Query your scraped corpus with local LLM (Ollama) when available, with extractive fallback when unavailable.")

        ask_question = st.text_input("Ask a corpus-grounded question", value="", key="ask_prophet_question")
        ask_clicked = st.button("Ask The Prophet", use_container_width=True)

        result = st.session_state.analysis_result
        if ask_clicked:
            if not result or result.get("ok") == "false":
                st.session_state.ask_prophet_error = "Run a scrape analysis first so Prophet has evidence to search."
                st.session_state.ask_prophet_answer = ""
                st.session_state.ask_prophet_citations = []
                st.session_state.ask_prophet_engine = ""
            else:
                ask_result = run_ask_the_prophet(
                    question=ask_question,
                    article_corpus=result.get("article_corpus", []),
                )
                st.session_state.ask_prophet_error = ask_result.get("error", "")
                st.session_state.ask_prophet_answer = ask_result.get("answer", "")
                st.session_state.ask_prophet_citations = ask_result.get("citations", [])
                st.session_state.ask_prophet_engine = ask_result.get("engine", "")

        if st.session_state.ask_prophet_error:
            st.warning(st.session_state.ask_prophet_error)

        if st.session_state.ask_prophet_answer:
            st.markdown("**Answer**")
            engine = st.session_state.ask_prophet_engine
            if engine == "fallback":
                st.markdown('<div class="small">Engine: Extractive fallback (no local model runtime detected)</div>', unsafe_allow_html=True)
            elif engine == "ollama":
                st.markdown('<div class="small">Engine: Local Ollama model</div>', unsafe_allow_html=True)
            st.markdown(st.session_state.ask_prophet_answer)
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

    result = st.session_state.analysis_result
    bottom_left, bottom_center, bottom_right = st.columns([1, 1.5, 1], gap="large")
    with bottom_center:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Prophet Summary</div>', unsafe_allow_html=True)
        if not result:
            st.markdown(
                '<div class="small">Run analysis to generate a synthesized corpus summary.</div>',
                unsafe_allow_html=True,
            )
        elif result["ok"] == "false":
            st.markdown('<div class="small">Summary unavailable due to analysis error.</div>', unsafe_allow_html=True)
        else:
            st.write(result.get("summary", "No summary available for this run."))
            representative_line = result.get("representative_line", "")
            if representative_line:
                st.markdown("**Representative line**")
                st.markdown(f"> {representative_line}")
                source_url = result.get("representative_source_url", "")
                if source_url:
                    st.markdown(f'<div class="small"><a href="{source_url}" target="_blank">View Article</a></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="small">No clear source identified.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    bottom2_left, bottom2_center, bottom2_right = st.columns([1, 1.9, 1], gap="large")
    with bottom2_center:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Zero-Cost Analysis Results</div>', unsafe_allow_html=True)

        if not result:
            st.info("No analysis run yet.")
        elif result["ok"] == "false":
            st.error(result["error"])
        else:
            if result.get("keyword_filter_enabled"):
                active_keyword = result.get("keyword", "")
                st.success(f"Keyword filtering active: '{active_keyword}'")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Links found", result["links_found"])
            c2.metric("Articles attempted", result.get("articles_attempted", 0))
            c3.metric("Articles scraped", result["articles_scraped"])
            c4.metric("Articles failed", result.get("articles_failed", 0))

            if result.get("keyword_filter_enabled"):
                k1, k2, k3 = st.columns(3)
                k1.metric("Keyword", result.get("keyword", ""))
                k2.metric("Matching articles", result.get("matching_articles", 0))
                k3.metric("Candidate articles considered", result.get("candidate_articles_considered", 0))

            st.markdown("**Top 10 most common words**")
            st.caption("Common stopwords, month/day/date terms, and source-noise words (e.g., photo/file/ap/news/said) are excluded.")
            top_words = result.get("top_words", [])
            if top_words:
                rows = [
                    {
                        "word": row["word"],
                        "total occurrences": row["total_occurrences"],
                        "article count": row["article_count"],
                        "article coverage %": row["article_coverage_pct"],
                    }
                    for row in top_words
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.warning("No word data available. Try increasing max articles or a different homepage.")

            st.markdown("**Preview of scraped articles**")
            preview = result.get("scraped_preview", [])
            if preview:
                for entry in preview:
                    title = entry.get("title", entry["url"])
                    word_count = entry.get("word_count")
                    suffix = f" · {word_count} filtered words" if word_count else ""
                    st.markdown(f"- [{title}]({entry['url']}){suffix}")
            else:
                st.markdown('<div class="small">No article previews available.</div>', unsafe_allow_html=True)

            suggestions = result.get("keyword_suggestions", [])
            if result.get("keyword_filter_enabled") and suggestions:
                st.markdown("**Suggested alternative keywords**")
                st.caption("Low keyword hit count detected. Try one of these broader terms.")
                st.write(", ".join(suggestions))

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
