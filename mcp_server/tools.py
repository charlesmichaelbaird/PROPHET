"""Tool helpers for MCP-style backend wiring."""

from __future__ import annotations

import re
from html import unescape

import requests


def fetch_url(url: str, timeout: int = 10) -> str:
    """Fetch raw HTML for a URL."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def clean_text(html: str) -> str:
    """Very small HTML-to-text cleaner."""
    no_tags = re.sub(r"<[^>]+>", " ", html)
    normalized = re.sub(r"\s+", " ", unescape(no_tags)).strip()
    return normalized


def summarize_text(text: str, limit: int = 500) -> str:
    """Stub summary: return the first chunk of cleaned text."""
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."
