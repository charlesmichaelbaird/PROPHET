"""Minimal server surface for zero-cost homepage analysis."""

from __future__ import annotations

from mcp_server.rag import ingest_new_articles
from mcp_server.tools import analyze_homepage, ask_the_prophet, query_site_article_count


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
            "articles_new_saved": 0,
            "articles_duplicates_skipped": 0,
            "scraped_preview": [],
            "top_words": [],
            "summary": "",
            "representative_line": "",
            "representative_source_url": "",
            "keyword_filter_enabled": keyword_filter_enabled,
            "keyword": keyword.strip(),
            "article_corpus": [],
            "persisted_articles": [],
            "duplicate_articles": [],
            "article_manifest_path": "",
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
            "articles_new_saved": 0,
            "articles_duplicates_skipped": 0,
            "scraped_preview": [],
            "top_words": [],
            "summary": "",
            "representative_line": "",
            "representative_source_url": "",
            "keyword_filter_enabled": keyword_filter_enabled,
            "keyword": keyword.strip(),
            "article_corpus": [],
            "persisted_articles": [],
            "duplicate_articles": [],
            "article_manifest_path": "",
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


def run_article_count_query(homepage_url: str, max_links: int = 200) -> dict:
    """Run lightweight discovery-only count query for candidate article links."""
    try:
        result = query_site_article_count(homepage_url=homepage_url, max_links=max_links)
        return {"ok": "true", "error": "", **result}
    except ValueError as exc:
        return {"ok": "false", "error": f"Invalid URL: {exc}", "homepage_url": homepage_url, "links_found": 0, "preview": []}
    except Exception as exc:
        return {"ok": "false", "error": f"Count query failed: {exc}", "homepage_url": homepage_url, "links_found": 0, "preview": []}


def run_index_data() -> dict:
    """Run explicit index-only pass over locally saved corpus."""
    try:
        indexing = ingest_new_articles()
        inspected = int(indexing.get("processed_articles_total", 0))
        new_articles = int(indexing.get("new_articles_indexed", 0))
        already_indexed = max(inspected - new_articles, 0)
        return {
            "ok": "true",
            "error": "",
            **indexing,
            "inspected_articles": inspected,
            "already_indexed_articles": already_indexed,
        }
    except Exception as exc:
        return {
            "ok": "false",
            "error": f"Indexing failed: {exc}",
            "inspected_articles": 0,
            "already_indexed_articles": 0,
            "new_articles_indexed": 0,
            "new_chunks_indexed": 0,
        }


TOOLS = {
    "analyze_homepage": analyze_homepage,
    "run_pipeline": run_pipeline,
    "run_article_count_query": run_article_count_query,
    "run_ask_the_prophet": run_ask_the_prophet,
    "run_index_data": run_index_data,
}
