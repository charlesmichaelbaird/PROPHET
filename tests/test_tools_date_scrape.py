from __future__ import annotations

import unittest
from unittest.mock import patch

from mcp_server.tools import query_source_article_count_by_date, request_scrape_stop, scrape_source_articles_by_date


class TestDateScrapeBehavior(unittest.TestCase):
    def test_uses_query_date_directory_when_no_published_metadata(self) -> None:
        persisted: list[dict] = []

        def _fake_discover(source_name: str, query_date, max_links: int):  # type: ignore[no-untyped-def]
            return ([{"url": "https://example.com/a1", "title": "A1", "publication_date": "", "lastmod": ""}], [], "")

        def _fake_persist(**kwargs):  # type: ignore[no-untyped-def]
            persisted.append(kwargs)
            return {"status": "saved", "content_hash": "h1", "written_paths": {}, "manifest_path": "manifest.json"}

        with (
            patch("mcp_server.tools._discover_articles_by_date", side_effect=_fake_discover),
            patch("mcp_server.tools.fetch_url", return_value="<html><p>Alpha story content today.</p></html>"),
            patch("mcp_server.tools.persist_article_if_new", side_effect=_fake_persist),
            patch("mcp_server.tools.write_run_index", return_value="run-index.json"),
        ):
            result = scrape_source_articles_by_date(source_name="ap", date_str="03/25/2026", max_articles=1)

        self.assertEqual(result["selected_date"], "03/25/2026")
        self.assertEqual(result["articles_scraped"], 1)
        self.assertEqual(persisted[0]["published_at"], "2026-03-25T00:00:00+00:00")
        self.assertEqual(persisted[0]["published_date_source"], "query_date_fallback")

    def test_stop_request_breaks_loop_cleanly(self) -> None:
        links = [{"url": f"https://example.com/{idx}", "title": f"T{idx}", "publication_date": "", "lastmod": ""} for idx in range(3)]
        persisted: list[dict] = []
        stop_requested = {"value": False}

        def _fake_discover(source_name: str, query_date, max_links: int):  # type: ignore[no-untyped-def]
            return (links, [], "")

        def _fake_fetch(url: str, timeout: int = 0, session=None):  # type: ignore[no-untyped-def]
            if not stop_requested["value"]:
                request_scrape_stop("ap")
                stop_requested["value"] = True
            return "<html><p>Article body text here.</p></html>"

        def _fake_persist(**kwargs):  # type: ignore[no-untyped-def]
            persisted.append(kwargs)
            return {"status": "saved", "content_hash": "h1", "written_paths": {}, "manifest_path": "manifest.json"}

        with (
            patch("mcp_server.tools._discover_articles_by_date", side_effect=_fake_discover),
            patch("mcp_server.tools.fetch_url", side_effect=_fake_fetch),
            patch("mcp_server.tools.persist_article_if_new", side_effect=_fake_persist),
            patch("mcp_server.tools.write_run_index", return_value="run-index.json"),
        ):
            result = scrape_source_articles_by_date(source_name="ap", date_str="03/25/2026", max_articles=3)

        self.assertTrue(result["scrape_stopped_by_user"])
        self.assertEqual(result["articles_attempted"], 1)
        self.assertEqual(len(persisted), 1)

    def test_ap_scrape_skips_non_english_lang_pages(self) -> None:
        def _fake_discover(source_name: str, query_date, max_links: int):  # type: ignore[no-untyped-def]
            return ([{"url": "https://example.com/es-story", "title": "ES", "publication_date": "", "lastmod": ""}], [], "")

        with (
            patch("mcp_server.tools._discover_articles_by_date", side_effect=_fake_discover),
            patch("mcp_server.tools.fetch_url", return_value='<html lang="es"><p>Hola mundo noticia.</p></html>'),
            patch("mcp_server.tools.persist_article_if_new") as persist_mock,
            patch("mcp_server.tools.write_run_index", return_value="run-index.json"),
        ):
            result = scrape_source_articles_by_date(source_name="ap", date_str="03/25/2026", max_articles=1)

        self.assertEqual(result["articles_scraped"], 0)
        self.assertFalse(persist_mock.called)
        self.assertTrue(
            any("filtered_non_english_lang_decl" in row for row in result.get("fetch_diagnostics", []))
        )

    def test_propublica_scrape_skips_non_english_page(self) -> None:
        def _fake_discover(source_name: str, query_date, max_links: int):  # type: ignore[no-untyped-def]
            return ([{"url": "https://www.propublica.org/espanol/story", "title": "ES", "publication_date": "", "lastmod": ""}], [], "")

        with (
            patch("mcp_server.tools._discover_articles_by_date", side_effect=_fake_discover),
            patch("mcp_server.tools.fetch_url", return_value='<html lang="es"><p>Hola noticia.</p></html>'),
            patch("mcp_server.tools.persist_article_if_new") as persist_mock,
            patch("mcp_server.tools.write_run_index", return_value="run-index.json"),
        ):
            result = scrape_source_articles_by_date(source_name="propublica", date_str="03/25/2026", max_articles=1)

        self.assertEqual(result["articles_scraped"], 0)
        self.assertFalse(persist_mock.called)
        self.assertTrue(any("filtered_non_english_propublica" in row for row in result.get("fetch_diagnostics", [])))

    def test_aljazeera_query_returns_zero_results_without_exception(self) -> None:
        with patch("mcp_server.tools._discover_articles_by_date", return_value=([], ["no_date_matches:sitemap"], "index.xml")):
            result = query_source_article_count_by_date(source_name="aljazeera", date_str="03/25/2026", max_links=10)

        self.assertEqual(result["links_found"], 0)
        self.assertEqual(result["discovery_status"], "no_results")


if __name__ == "__main__":
    unittest.main()
