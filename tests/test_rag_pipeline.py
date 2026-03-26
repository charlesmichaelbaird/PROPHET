from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from mcp_server.rag import (
    LocalVectorIndex,
    OllamaClient,
    chunk_text,
    get_indexing_status,
    index_missing_articles,
    ingest_new_articles,
)
from mcp_server.storage import persist_article_if_new


class _FakeEmbedClient:
    def embed(self, text: str) -> list[float]:
        length = float(len(text.split()))
        ascii_signal = float(sum(ord(ch) for ch in text[:25]) % 997)
        return [length, ascii_signal, 1.0]


class TestRagPipeline(unittest.TestCase):
    def test_chunk_text_deterministic_overlap(self) -> None:
        text = " ".join([f"word{i}" for i in range(1, 421)])
        chunks = chunk_text(text, max_chunk_words=100, overlap_words=20)

        self.assertGreaterEqual(len(chunks), 5)
        first = chunks[0].split()
        second = chunks[1].split()
        self.assertEqual(first[-20:], second[:20])

    def test_ingest_new_articles_is_incremental(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)

            saved = persist_article_if_new(
                homepage_url="https://example.com",
                source_homepage_url="https://example.com",
                article_url="https://example.com/article/alpha",
                title="Alpha",
                scrape_timestamp=now,
                clean_text=" ".join(["alpha"] * 260),
                data_root=root,
            )
            self.assertEqual(saved["status"], "saved")

            vector_path = root / "index" / "vector_store.sqlite"
            manifest_path = root / "index" / "vector_manifest.json"

            first = ingest_new_articles(
                client=_FakeEmbedClient(),
                index=LocalVectorIndex(db_path=vector_path),
                manifest_path=manifest_path,
                data_root=root,
            )
            self.assertEqual(first["new_articles_indexed"], 1)
            self.assertGreater(first["new_chunks_indexed"], 0)

            second = ingest_new_articles(
                client=_FakeEmbedClient(),
                index=LocalVectorIndex(db_path=vector_path),
                manifest_path=manifest_path,
                data_root=root,
            )
            self.assertEqual(second["new_articles_indexed"], 0)
            self.assertEqual(second["new_chunks_indexed"], 0)

    def test_embed_uses_native_api_embed_shape(self) -> None:
        client = OllamaClient(base_url="http://localhost:11434", embed_model="nomic-embed-text")
        good_response = Mock(status_code=200)
        good_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        with patch("mcp_server.rag.requests.post") as mock_post:
            mock_post.return_value = good_response
            vector = client.embed("test text")

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/embed",
            json={"model": "nomic-embed-text", "input": ["test text"]},
            timeout=60.0,
        )

    def test_status_and_index_missing_articles_workflow(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            saved = persist_article_if_new(
                homepage_url="https://example.com",
                source_homepage_url="https://example.com",
                article_url="https://example.com/article/beta",
                title="Beta",
                scrape_timestamp=now,
                clean_text=" ".join(["beta"] * 210),
                data_root=root,
            )
            self.assertEqual(saved["status"], "saved")

            vector_path = root / "index" / "vector_store.sqlite"
            manifest_path = root / "index" / "vector_manifest.json"
            initial = get_indexing_status(
                index=LocalVectorIndex(db_path=vector_path),
                manifest_path=manifest_path,
                data_root=root,
            )
            self.assertEqual(initial["missing_articles_total"], 1)

            indexed = index_missing_articles(
                missing_content_hashes=initial["missing_content_hashes"],
                client=_FakeEmbedClient(),
                index=LocalVectorIndex(db_path=vector_path),
                manifest_path=manifest_path,
                data_root=root,
            )
            self.assertEqual(indexed["new_articles_indexed"], 1)
            self.assertTrue(indexed["is_index_up_to_date"])


if __name__ == "__main__":
    unittest.main()
