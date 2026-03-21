"""Minimal MCP-oriented server surface for Milestone 1."""

from __future__ import annotations

from mcp_server.tools import clean_text, fetch_url, summarize_text


def run_pipeline(url: str) -> dict[str, str]:
    """Run URL -> cleaned text -> summary with simple error reporting."""
    try:
        html = fetch_url(url)
        cleaned_text = clean_text(html)
        summary = summarize_text(cleaned_text)
        return {
            "ok": "true",
            "cleaned_text": cleaned_text,
            "summary": summary,
            "error": "",
        }
    except ValueError as exc:
        return {
            "ok": "false",
            "cleaned_text": "",
            "summary": "",
            "error": f"Invalid URL: {exc}",
        }
    except Exception as exc:
        return {
            "ok": "false",
            "cleaned_text": "",
            "summary": "",
            "error": f"Request failed: {exc}",
        }


# MCP-style registry to show intended tool exposure.
TOOLS = {
    "fetch_url": fetch_url,
    "clean_text": clean_text,
    "summarize_text": summarize_text,
    "run_pipeline": run_pipeline,
}
