"""PROPHET web dashboard UI (Streamlit)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import streamlit as st
from plotly import graph_objects as go

# Ensure repository root is importable when running
# `streamlit run frontend/app.py` from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp_server.server import run_pipeline
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


@st.fragment(run_every=1)
def render_meta_chips() -> None:
    now_utc = datetime.now(timezone.utc)
    st.markdown(
        f'<span class="meta-chip">DATE • {now_utc:%Y-%m-%d}</span>'
        f'<span class="meta-chip">UTC • {now_utc:%H:%M:%S}</span>'
        '<span class="meta-chip">MODE • ZERO-COST NO-LLM</span>',
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
        st.markdown('<div class="panel-title">Zero-Cost Analysis Results</div>', unsafe_allow_html=True)

        result = st.session_state.analysis_result
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


def _build_btc_chart(history_rows: list[dict], runtime_points: list[dict]) -> go.Figure:
    price_x = [row["timestamp"] for row in history_rows] + [point["timestamp"] for point in runtime_points]
    price_y = [row["close"] for row in history_rows] + [point["price"] for point in runtime_points]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_x,
            y=price_y,
            mode="lines",
            name="BTC/USD Price",
            line={"color": "#5bc0ff", "width": 2.6},
        )
    )
    for field, name, color in (
        ("ma_10", "10-day MA", "#8ef8ce"),
        ("ma_30", "30-day MA", "#f8d66b"),
        ("ma_100", "100-day MA", "#ff8ca8"),
    ):
        fig.add_trace(
            go.Scatter(
                x=[row["timestamp"] for row in history_rows],
                y=[row[field] for row in history_rows],
                mode="lines",
                name=name,
                line={"width": 1.8, "color": color},
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8, 15, 28, 0.72)",
        margin={"l": 18, "r": 18, "t": 18, "b": 18},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0.01},
        xaxis={"title": "", "gridcolor": "rgba(140, 177, 255, 0.12)"},
        yaxis={"title": "Price (USD)", "gridcolor": "rgba(140, 177, 255, 0.12)", "tickprefix": "$"},
        hovermode="x unified",
    )
    return fig


@st.fragment(run_every=2)
def render_btc_live_view() -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">BTC/USD Monitor</div>', unsafe_allow_html=True)

    history = fetch_btc_history(limit=420)
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

    fig = _build_btc_chart(history, st.session_state.btc_runtime_points)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


if view == "PROPHET Dashboard":
    render_prophet_dashboard()
else:
    render_btc_live_view()
