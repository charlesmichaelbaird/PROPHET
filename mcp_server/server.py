"""Minimal server surface for zero-cost homepage analysis."""

from __future__ import annotations

from mcp_server.tools import analyze_homepage


def run_pipeline(
    homepage_url: str,
    max_articles: int = 20,
    keyword: str = "",
    keyword_filter_enabled: bool = False,
) -> dict:
    """Run homepage -> article scrape -> word-frequency analysis."""
    try:
        analysis = analyze_homepage(
            homepage_url=homepage_url,
            max_articles=max_articles,
            keyword=keyword,
            keyword_filter_enabled=keyword_filter_enabled,
        )
        return {"ok": "true", "error": "", **analysis}
    except ValueError as exc:
        return {
            "ok": "false",
            "error": f"Invalid URL: {exc}",
            "links_found": 0,
            "articles_attempted": 0,
            "articles_scraped": 0,
            "articles_failed": 0,
            "scraped_preview": [],
            "top_words": [],
            "summary": "",
            "keyword_filter_enabled": keyword_filter_enabled,
            "keyword": keyword.strip(),
        }
    except Exception as exc:
        return {
            "ok": "false",
            "error": f"Request failed: {exc}",
            "links_found": 0,
            "articles_attempted": 0,
            "articles_scraped": 0,
            "articles_failed": 0,
            "scraped_preview": [],
            "top_words": [],
            "summary": "",
            "keyword_filter_enabled": keyword_filter_enabled,
            "keyword": keyword.strip(),
        }


TOOLS = {
    "analyze_homepage": analyze_homepage,
    "run_pipeline": run_pipeline,
}
