"""Tool helpers for MCP-style backend wiring."""

from __future__ import annotations

import re
import os
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests
from openai import OpenAI


class VisibleTextParser(HTMLParser):
    """Very small HTML parser that collects likely-visible text."""

    _HIDDEN_TAGS = {"script", "style", "noscript", "head", "title", "meta", "link"}

    def __init__(self) -> None:
        super().__init__()
        self._hidden_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag.lower() in self._HIDDEN_TAGS:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag.lower() in self._HIDDEN_TAGS and self._hidden_depth > 0:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._hidden_depth == 0 and data.strip():
            self._chunks.append(data)

    def get_text(self) -> str:
        return " ".join(self._chunks)


def fetch_url(url: str, timeout: int = 10) -> str:
    """Fetch raw HTML for a URL with basic validation and error messages."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must include http:// or https:// and a valid host")

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "PROPHET-M1/0.1 (+https://example.local)"},
        )
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Failed to fetch URL: {exc}") from exc


def clean_text(html: str) -> str:
    """Extract likely-visible text from HTML in a simple, dependency-free way."""
    parser = VisibleTextParser()
    parser.feed(html)
    parser.close()

    text = unescape(parser.get_text())
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized


def summarize_text(text: str) -> str:
    """Summarize text with OpenAI. Requires OPENAI_API_KEY and the openai package."""
    if not text.strip():
        return ""

    # Setup note: set OPENAI_API_KEY in your environment.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"Summarize this text in 2-4 short sentences:\n\n{text[:12000]}",
    )
    return response.output_text.strip()
