"""PROPHET web dashboard UI (Streamlit)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import streamlit as st

# Ensure repository root is importable when running
# `streamlit run frontend/app.py` from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp_server.server import run_pipeline

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
</style>
""",
    unsafe_allow_html=True,
)

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

now_utc = datetime.now(timezone.utc)

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
st.markdown(
    f'<span class="meta-chip">DATE • {now_utc:%Y-%m-%d}</span>'
    f'<span class="meta-chip">UTC • {now_utc:%H:%M:%S}</span>'
    '<span class="meta-chip">MODE • ZERO-COST NO-LLM</span>',
    unsafe_allow_html=True,
)
st.markdown("</section>", unsafe_allow_html=True)

left, right = st.columns([1.05, 1.6], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Operator Console</div>', unsafe_allow_html=True)
    st.caption("Provide a homepage (e.g., https://apnews.com), scrape article links, and analyze top words.")

    homepage_url = st.text_input("Source Homepage URL", value="https://apnews.com")
    max_articles = st.number_input("Max articles to scrape", min_value=5, max_value=50, value=20, step=5)

    run_clicked = st.button("Run Zero-Cost Analysis", use_container_width=True)
    clear_clicked = st.button("Reset Results", use_container_width=True)

    if clear_clicked:
        st.session_state.analysis_result = None

    if run_clicked:
        with st.spinner("Fetching homepage and scraping articles..."):
            st.session_state.analysis_result = run_pipeline(homepage_url.strip(), int(max_articles))

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
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Links found", result["links_found"])
        c2.metric("Articles attempted", result.get("articles_attempted", 0))
        c3.metric("Articles scraped", result["articles_scraped"])
        c4.metric("Articles failed", result.get("articles_failed", 0))

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
    st.markdown('</div>', unsafe_allow_html=True)
