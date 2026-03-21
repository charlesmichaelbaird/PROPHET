"""PROPHET web dashboard UI (Streamlit)."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from mcp_server.server import run_pipeline

st.set_page_config(
    page_title="PROPHET | Prediction Intelligence Dashboard",
    page_icon="🔮",
    layout="wide",
)

st.markdown(
    """
<style>
:root {
  --bg: #070b13;
  --bg-soft: #0c1220;
  --panel: rgba(17, 26, 45, 0.82);
  --panel-border: rgba(111, 160, 255, 0.23);
  --text: #d9e7ff;
  --muted: #8ea5cb;
  --accent: #4dd3ff;
  --accent-2: #7d8bff;
  --good: #5dd39e;
  --warn: #f5c26b;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  background: radial-gradient(circle at 10% 10%, #162442 0%, #0a0f1d 35%, #060910 100%);
  color: var(--text);
}

.block-container {
  padding-top: 1.3rem;
  max-width: 1500px;
}

.hero {
  position: relative;
  border: 1px solid var(--panel-border);
  border-radius: 20px;
  overflow: hidden;
  background:
    radial-gradient(circle at 80% 15%, rgba(77, 211, 255, 0.21), transparent 30%),
    radial-gradient(circle at 22% 80%, rgba(125, 139, 255, 0.18), transparent 32%),
    linear-gradient(125deg, rgba(8, 13, 24, 0.95), rgba(20, 31, 56, 0.88));
  padding: 1.2rem 1.4rem 1.3rem 1.4rem;
  margin-bottom: 1rem;
}

.hero::after {
  content: "";
  position: absolute;
  right: -120px;
  top: -95px;
  width: 420px;
  height: 420px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(130, 201, 255, 0.35), rgba(130, 201, 255, 0.04) 55%, transparent 72%);
  pointer-events: none;
}

.brand {
  letter-spacing: 0.42rem;
  font-weight: 800;
  font-size: 1.95rem;
  margin: 0;
}

.subtitle {
  margin-top: 0.1rem;
  color: var(--muted);
  font-size: 0.97rem;
}

.meta-chip {
  border: 1px solid rgba(132, 180, 255, 0.3);
  border-radius: 999px;
  padding: 0.4rem 0.7rem;
  display: inline-block;
  margin-right: 0.55rem;
  margin-top: 0.45rem;
  color: #d2e6ff;
  background: rgba(40, 57, 90, 0.4);
  font-size: 0.82rem;
}

.visual {
  border: 1px solid rgba(153, 196, 255, 0.2);
  border-radius: 12px;
  padding: 0.8rem;
  min-height: 118px;
  background:
    linear-gradient(180deg, rgba(145, 202, 255, 0.08), rgba(75, 98, 172, 0.04)),
    repeating-linear-gradient(90deg, rgba(130, 171, 255, 0.16) 0 1px, transparent 1px 22px),
    repeating-linear-gradient(0deg, rgba(130, 171, 255, 0.08) 0 1px, transparent 1px 20px);
}

.visual h4 {
  margin: 0 0 0.45rem 0;
  color: #c7ddff;
}

.panel {
  border: 1px solid var(--panel-border);
  border-radius: 16px;
  padding: 0.95rem 1rem;
  background: var(--panel);
  margin-bottom: 0.8rem;
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.24);
}

.panel-title {
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.14rem;
  color: #b6cdff;
  margin-bottom: 0.55rem;
}

.small {
  color: var(--muted);
  font-size: 0.87rem;
}

.ticker {
  white-space: nowrap;
  overflow: hidden;
  border: 1px solid rgba(124, 175, 255, 0.22);
  border-radius: 12px;
  padding: 0.53rem 0.65rem;
  margin-bottom: 0.65rem;
  color: #d5e8ff;
  background: rgba(26, 39, 64, 0.78);
}

.badge {
  display: inline-block;
  margin: 0.2rem 0.38rem 0.2rem 0;
  border-radius: 999px;
  padding: 0.2rem 0.55rem;
  border: 1px solid rgba(137, 187, 255, 0.3);
  color: #cbe0ff;
  font-size: 0.78rem;
  background: rgba(67, 92, 142, 0.32);
}

hr {
  border: none;
  border-top: 1px solid rgba(138, 177, 243, 0.2);
  margin: 0.7rem 0;
}

[data-testid="stTextInput"] input {
  background: #0a1324;
  border: 1px solid rgba(132, 181, 255, 0.3);
  color: #d8ebff;
}

.stButton > button {
  border-radius: 10px;
  border: 1px solid rgba(138, 183, 255, 0.4);
  background: linear-gradient(130deg, #163769, #1f5f8d);
  color: #eaf5ff;
  font-weight: 700;
}

.stTextArea textarea {
  background: #091325;
  border: 1px solid rgba(132, 181, 255, 0.3);
  color: #d8ebff;
}
</style>
""",
    unsafe_allow_html=True,
)

if "summary" not in st.session_state:
    st.session_state.summary = ""
if "cleaned_text" not in st.session_state:
    st.session_state.cleaned_text = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""

now_utc = datetime.now(timezone.utc)

st.markdown('<section class="hero">', unsafe_allow_html=True)
hero_left, hero_right = st.columns([2.2, 1.4], gap="large")
with hero_left:
    st.markdown('<p class="brand">PROPHET</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Global intelligence synthesis for anticipatory decisions and scenario forecasting.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="meta-chip">DATE • {now_utc:%Y-%m-%d}</span>'
        f'<span class="meta-chip">UTC • {now_utc:%H:%M:%S}</span>'
        '<span class="meta-chip">STATUS • LIVE MONITORING</span>',
        unsafe_allow_html=True,
    )

with hero_right:
    st.markdown(
        """
<div class="visual">
  <h4>Orbital Signal View</h4>
  <div class="small">AI sentinel + world-state light-curve placeholder.</div>
  <div style="margin-top:8px;color:#9ec7ff;font-size:0.83rem;">
    ◉ Hemisphere activity gradient enabled • Synthetic geogrid online
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
st.markdown("</section>", unsafe_allow_html=True)

left, right = st.columns([1.25, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Live Intelligence Feed</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ticker">🛰️ Arctic shipping lane anomalies rise 12% • 📈 Commodity volatility pulse elevated in APAC • 🌐 Fiber outage risk watch in EMEA • ⚠️ Election disinformation chatter trending in LATAM</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span class='badge'>Global Events</span><span class='badge'>Markets</span><span class='badge'>Regional Signals</span><span class='badge'>Scenario Engine</span>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Current World State</div>', unsafe_allow_html=True)
        st.markdown(
            """
- **Geopolitical heat:** Medium-high in 3 corridors.
- **Supply chain stress:** Elevated around two maritime chokepoints.
- **Macro sentiment:** Risk-on opening, but defensive hedging increasing.
"""
        )
        st.markdown('<div class="small">Last synthetic update: 4 min ago.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Trend Watchlist</div>', unsafe_allow_html=True)
        st.markdown(
            """
1. **Energy corridor stability index** ↘
2. **FX shock probability (7d)** ↗
3. **Urban protest emergence score** ↗
4. **AI policy breakpoint risk** ↔
"""
        )
        st.markdown('<div class="small">Confidence bands are currently simulated.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Probabilistic Scenario Summary</div>', unsafe_allow_html=True)
    st.markdown(
        """
- **Scenario A (52%)**: controlled volatility with localized disruptions.
- **Scenario B (31%)**: synchronized market + policy shock in two regions.
- **Scenario C (17%)**: low-probability cascading infrastructure incident.
"""
    )
    st.markdown(
        '<div class="small">Note: Scenario engine is currently stubbed for UI and workflow validation.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Operator Console</div>', unsafe_allow_html=True)
    st.caption("Ingest a source URL, generate synthesis, then query the intelligence stack.")

    url = st.text_input("Source URL", placeholder="https://news.example.com/article")
    b1, b2 = st.columns(2)
    with b1:
        scrape_clicked = st.button("Run Scrape + Synthesize", use_container_width=True)
    with b2:
        clear_clicked = st.button("Reset Session", use_container_width=True)

    if clear_clicked:
        st.session_state.summary = ""
        st.session_state.cleaned_text = ""
        st.session_state.answer = ""

    if scrape_clicked:
        if not url.strip():
            st.warning("Please provide a valid URL before running ingestion.")
        else:
            with st.spinner("Running fetch → clean → summarize pipeline..."):
                result = run_pipeline(url.strip())

            if result["ok"] == "false":
                st.error(result["error"])
            else:
                st.session_state.summary = result["summary"]
                st.session_state.cleaned_text = result["cleaned_text"]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Synthesis Output**")
    if st.session_state.summary:
        st.success(st.session_state.summary)
    else:
        st.info("No synthesis yet. Ingest a source to populate this panel.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Ask The Prophet</div>', unsafe_allow_html=True)

    question = st.text_area(
        "Analyst Query",
        placeholder="What second-order effects should we monitor over the next 2 weeks?",
        height=110,
    )

    controls = st.columns(3)
    controls[0].selectbox("Analysis Mode", ["Strategic", "Market", "Regional"], index=0)
    controls[1].selectbox("Source Scope", ["Ingested only", "Global stub mix", "Hybrid"], index=0)
    controls[2].selectbox("Time Horizon", ["24h", "7d", "30d"], index=1)

    if st.button("Query Prophet", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question to query the system.")
        elif not st.session_state.summary:
            st.warning("Run ingestion first so Prophet has source context.")
        else:
            st.session_state.answer = (
                "Based on the ingested source and current synthetic world-state signals, watch for "
                "policy narrative acceleration, supply-side response lag, and correlated sentiment shocks "
                "across adjacent sectors. (Stubbed response layer.)"
            )

    if st.session_state.answer:
        st.markdown("**Prophet Response**")
        st.write(st.session_state.answer)
    else:
        st.markdown('<div class="small">Response channel idle. Submit a query to generate output.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
