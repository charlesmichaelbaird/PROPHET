from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import requests

from mcp_server.rag import (
    LocalVectorIndex,
    OllamaClient,
    answer_question,
    chunk_text,
    get_indexing_status,
    index_missing_articles,
    ingest_new_articles,
)
from mcp_server.storage import persist_article_if_new


class _FakeEmbedClient:
    embedding_mode = "test-fake"

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
        tags_response = Mock(status_code=200)
        tags_response.json.return_value = {"models": [{"name": "nomic-embed-text:latest"}]}
        good_response = Mock(status_code=200)
        good_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        with patch("mcp_server.rag.requests.get") as mock_get, patch("mcp_server.rag.requests.post") as mock_post:
            mock_get.return_value = tags_response
            mock_post.return_value = good_response
            vector = client.embed("test text")

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=60.0)
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/embed",
            json={"model": "nomic-embed-text", "input": ["test text"]},
            timeout=60.0,
        )

    def test_embed_falls_back_to_installed_model_list(self) -> None:
        client = OllamaClient(base_url="http://localhost:11434", embed_model="missing-model")
        tags_response = Mock(status_code=200)
        tags_response.json.return_value = {"models": [{"name": "nomic-embed-text:latest"}]}

        fail_response = Mock(status_code=404)
        fail_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 missing")
        success_response = Mock(status_code=200)
        success_response.json.return_value = {"embeddings": [[0.7, 0.8, 0.9]]}

        with patch("mcp_server.rag.requests.get") as mock_get, patch("mcp_server.rag.requests.post") as mock_post:
            mock_get.return_value = tags_response
            mock_post.side_effect = [
                fail_response,  # configured missing model on /api/embed
                fail_response,  # configured missing model on legacy /api/embeddings
                success_response,  # discovered embed model on /api/embed
            ]
            vector = client.embed("fallback text")

        self.assertEqual(vector, [0.7, 0.8, 0.9])
        self.assertEqual(client.embedding_mode, "current:/api/embed")

    def test_embed_uses_legacy_variant_when_current_is_unavailable(self) -> None:
        client = OllamaClient(base_url="http://localhost:11434", embed_model="nomic-embed-text")
        tags_response = Mock(status_code=200)
        tags_response.json.return_value = {"models": []}

        current_missing = Mock(status_code=404)
        legacy_success = Mock(status_code=200)
        legacy_success.json.return_value = {"embedding": [0.11, 0.22, 0.33]}

        with patch("mcp_server.rag.requests.get") as mock_get, patch("mcp_server.rag.requests.post") as mock_post:
            mock_get.return_value = tags_response
            mock_post.side_effect = [current_missing, legacy_success]
            vector = client.embed("legacy path text")

        self.assertEqual(vector, [0.11, 0.22, 0.33])
        self.assertEqual(client.embedding_mode, "legacy:/api/embeddings")

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

    def test_answer_requires_explicit_index_run_when_corpus_not_indexed(self) -> None:
        with patch("mcp_server.rag.get_indexing_status") as mock_status:
            mock_status.return_value = {
                "processed_articles_total": 4,
                "indexed_articles_total": 0,
                "missing_articles_total": 4,
                "is_index_up_to_date": False,
                "missing_content_hashes": ["a", "b", "c", "d"],
            }
            result = answer_question("What happened?", client=_FakeEmbedClient(), index=LocalVectorIndex(Path(':memory:')))

        self.assertIn("Click 'Index Data'", result["answer"])
        self.assertEqual(result["indexing_triggered"], False)

    def test_indexing_emits_progress_events_and_processed_scan_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            persist_article_if_new(
                homepage_url="https://example.com",
                source_homepage_url="https://example.com",
                article_url="https://example.com/article/delta",
                title="Delta",
                scrape_timestamp=now,
                clean_text=" ".join(["delta"] * 220),
                data_root=root,
            )

            events: list[dict] = []
            result = ingest_new_articles(
                client=_FakeEmbedClient(),
                index=LocalVectorIndex(db_path=root / "index" / "vector_store.sqlite"),
                manifest_path=root / "index" / "vector_manifest.json",
                data_root=root,
                progress_callback=events.append,
            )

            self.assertTrue(any(event.get("event") == "scan" for event in events))
            self.assertTrue(any(event.get("event") == "step" for event in events))
            self.assertTrue(str(root / "processed") in result.get("processed_scan_directory", ""))


if __name__ == "__main__":
    unittest.main()
