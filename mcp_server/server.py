"""Minimal server surface for zero-cost homepage analysis."""

from __future__ import annotations

from mcp_server.tools import analyze_homepage, ask_the_prophet


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
            "representative_line": "",
            "representative_source_url": "",
            "keyword_filter_enabled": keyword_filter_enabled,
            "keyword": keyword.strip(),
            "article_corpus": [],
            "persisted_articles": [],
            "run_index_path": "",
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
            "representative_line": "",
            "representative_source_url": "",
            "keyword_filter_enabled": keyword_filter_enabled,
            "keyword": keyword.strip(),
            "article_corpus": [],
            "persisted_articles": [],
            "run_index_path": "",
        }


def run_ask_the_prophet(question: str, article_corpus: list[dict[str, str]]) -> dict:
    """Run grounded Q&A over the scraped article corpus."""
    try:
        return ask_the_prophet(question=question, article_corpus=article_corpus)
    except Exception as exc:
        return {
            "ok": "false",
            "error": f"Ask The Prophet failed: {exc}",
            "answer": "",
            "citations": [],
            "engine": "",
        }


TOOLS = {
    "analyze_homepage": analyze_homepage,
    "run_pipeline": run_pipeline,
    "run_ask_the_prophet": run_ask_the_prophet,
}
