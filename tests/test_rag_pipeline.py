from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from mcp_server.rag import LocalVectorIndex, OllamaClient, chunk_text, ingest_new_articles
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

    def test_embed_falls_back_to_next_model_when_first_missing(self) -> None:
        client = OllamaClient(base_url="http://localhost:11434", embed_model="missing-model", chat_model="llama3.1")

        missing_model_response = Mock(status_code=404)
        missing_model_response.raise_for_status.side_effect = RuntimeError("model missing")

        good_response = Mock(status_code=200)
        good_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        with patch("mcp_server.rag.requests.post") as mock_post:
            mock_post.side_effect = [
                missing_model_response,  # /api/embed missing-model
                missing_model_response,  # /api/embeddings missing-model
                good_response,  # /api/embed llama3.1
            ]
            vector = client.embed("test text")

        self.assertEqual(vector, [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
