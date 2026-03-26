from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from mcp_server.storage import _safe_source_name, compute_content_hash, persist_article_if_new


class TestStorageDeduplication(unittest.TestCase):
    def test_source_override_maps_propublica_to_normalized_slug(self) -> None:
        self.assertEqual(_safe_source_name("https://www.propublica.org/"), "propublica")

    def test_content_hash_is_stable_for_whitespace_variants(self) -> None:
        left = compute_content_hash("Alpha   beta\n\n gamma")
        right = compute_content_hash("Alpha beta gamma")
        self.assertEqual(left, right)

    def test_persist_article_if_new_saves_once_and_then_marks_duplicate(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)

            first = persist_article_if_new(
                homepage_url="https://example.com",
                source_homepage_url="https://example.com",
                article_url="https://example.com/article/1",
                title="Example Story",
                scrape_timestamp=now,
                clean_text="Same normalized content.",
                article_html="<html><p>Same normalized content.</p></html>",
                article_metadata={"url": "https://example.com/article/1"},
                data_root=root,
            )

            second = persist_article_if_new(
                homepage_url="https://example.com",
                source_homepage_url="https://example.com",
                article_url="https://example.com/article/1?utm=tracking",
                title="Example Story",
                scrape_timestamp=now,
                clean_text="Same normalized    content.",
                article_html="<html><p>Same normalized content.</p></html>",
                article_metadata={"url": "https://example.com/article/1?utm=tracking"},
                data_root=root,
            )

            self.assertEqual(first["status"], "saved")
            self.assertEqual(second["status"], "duplicate")
            self.assertEqual(first["content_hash"], second["content_hash"])

            manifest_path = Path(first["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entries = manifest.get("entries", {})
            self.assertEqual(len(entries), 1)

            only_entry = entries[first["content_hash"]]
            self.assertEqual(only_entry["duplicate_hits"], 1)
            self.assertIn("https://example.com/article/1", only_entry["seen_urls"])
            self.assertIn("https://example.com/article/1?utm=tracking", only_entry["seen_urls"])


if __name__ == "__main__":
    unittest.main()
