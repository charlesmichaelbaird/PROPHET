"""Minimal MCP-oriented server surface for Milestone 1."""

from __future__ import annotations

from mcp_server.tools import clean_text, fetch_url, summarize_text


def run_pipeline(url: str) -> str:
    """Run the tiny URL -> cleaned text -> summary pipeline."""
    try:
        html = fetch_url(url)
        text = clean_text(html)
        return summarize_text(text)
    except Exception as exc:  # Keep error handling simple for milestone 1.
        return f"Error: {exc}"


# MCP-style registry to show intended tool exposure.
TOOLS = {
    "fetch_url": fetch_url,
    "clean_text": clean_text,
    "summarize_text": summarize_text,
    "run_pipeline": run_pipeline,
}
