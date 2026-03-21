"""Minimal Streamlit frontend for Milestone 1."""

import streamlit as st

from mcp_server.server import run_pipeline

st.set_page_config(page_title="URL Runner", page_icon="🔗")
st.title("Milestone 1: URL to Summary")

url = st.text_input("Enter a URL", placeholder="https://example.com")

if st.button("Run"):
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        with st.spinner("Fetching and processing..."):
            result = run_pipeline(url.strip())

        if result["ok"] == "false":
            st.error(result["error"])
        else:
            st.subheader("Summary")
            st.write(result["summary"])
