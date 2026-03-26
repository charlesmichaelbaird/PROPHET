"""Local RAG pipeline for PROPHET using Ollama + persistent SQLite vector index."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Any

import requests

from mcp_server.storage import ensure_data_directories, load_article_index

DEFAULT_OLLAMA_BASE_URL = os.getenv("PROPHET_OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_EMBED_MODEL = os.getenv("PROPHET_OLLAMA_EMBED_MODEL", "nomic-embed-text")
DEFAULT_OLLAMA_CHAT_MODEL = os.getenv("PROPHET_OLLAMA_MODEL", "llama3.1")
DEFAULT_OLLAMA_TIMEOUT_SECONDS = float(os.getenv("PROPHET_OLLAMA_TIMEOUT_SECONDS", "60"))
DEFAULT_VECTOR_INDEX_PATH = Path(
    os.getenv("PROPHET_VECTOR_INDEX_PATH", str(Path(__file__).resolve().parents[1] / "data" / "index" / "vector_store.sqlite"))
)
DEFAULT_VECTOR_MANIFEST_PATH = Path(
    os.getenv("PROPHET_VECTOR_MANIFEST_PATH", str(Path(__file__).resolve().parents[1] / "data" / "index" / "vector_manifest.json"))
)


@dataclass
class RetrievalChunk:
    score: float
    text: str
    title: str
    url: str
    source: str
    clean_text_path: str
    scrape_timestamp: str
    content_hash: str


class OllamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        embed_model: str = DEFAULT_OLLAMA_EMBED_MODEL,
        chat_model: str = DEFAULT_OLLAMA_CHAT_MODEL,
        timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.timeout_seconds = timeout_seconds

    def embed(self, text: str) -> list[float]:
        """Embed text using native Ollama /api/embed endpoint."""
        candidate_models = self._candidate_embedding_models()
        last_error = ""
        for model_name in candidate_models:
            try:
                return self._embed_with_model(text=text, model_name=model_name)
            except RuntimeError as exc:
                last_error = str(exc)
                continue

        raise RuntimeError(
            "Ollama embedding request failed for all candidate models. "
            f"Tried: {', '.join(candidate_models)}. Last error: {last_error}"
        )

    def _candidate_embedding_models(self) -> list[str]:
        configured = [self.embed_model]
        discovered = self._discover_installed_models()
        embed_like = [name for name in discovered if any(marker in name.lower() for marker in ("embed", "nomic", "mxbai"))]
        fallback = [self.chat_model]
        all_names = configured + embed_like + discovered + fallback
        deduped: list[str] = []
        for name in all_names:
            clean = str(name).strip()
            if clean and clean not in deduped:
                deduped.append(clean)
        return deduped

    def _discover_installed_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout_seconds)
            response.raise_for_status()
            models = response.json().get("models", [])
            names = [str(model.get("name", "")).strip() for model in models if model.get("name")]
            return [name for name in names if name]
        except Exception:
            return []

    def _embed_with_model(self, text: str, model_name: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": model_name, "input": [text]},
            timeout=self.timeout_seconds,
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"model '{model_name}' failed on /api/embed: {exc}") from exc

        payload = response.json()
        embeddings = payload.get("embeddings", [])
        if not embeddings:
            raise RuntimeError(f"model '{model_name}' returned no embeddings")
        return [float(x) for x in embeddings[0]]

    def chat(self, question: str, chunks: list[RetrievalChunk]) -> str:
        context_lines: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            context_lines.append(
                f"[Source {idx}] {chunk.title}\nURL: {chunk.url}\nSource: {chunk.source}\nExcerpt: {chunk.text}"
            )
        context = "\n\n".join(context_lines)
        system_prompt = (
            "You are Ask The Prophet. You must answer ONLY from the provided corpus excerpts. "
            "If the excerpts are insufficient, reply exactly: "
            "'Not enough relevant scraped data to answer this question yet.'"
        )
        user_prompt = (
            f"Question: {question}\n\n"
            "Use only the evidence below. Do not use outside/world knowledge. "
            "If evidence is weak, decline exactly as instructed.\n\n"
            f"Evidence:\n{context}\n\n"
            "Return a concise answer and include brief source references like [Source 1]."
        )
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.chat_model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload.get("message", {})
        return str(message.get("content", "")).strip()


class LocalVectorIndex:
    def __init__(self, db_path: Path = DEFAULT_VECTOR_INDEX_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS article_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    title TEXT,
                    url TEXT,
                    source TEXT,
                    clean_text_path TEXT,
                    scrape_timestamp TEXT,
                    UNIQUE(content_hash, chunk_index)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_article_chunks_hash ON article_chunks(content_hash)")
            conn.commit()

    def add_chunks(
        self,
        content_hash: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata: dict[str, str],
    ) -> int:
        if not chunks or not embeddings:
            return 0
        inserted = 0
        with self._connect() as conn:
            for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
                if not vector:
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO article_chunks (
                        content_hash, chunk_index, chunk_text, embedding_json,
                        title, url, source, clean_text_path, scrape_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        content_hash,
                        idx,
                        chunk,
                        json.dumps(vector),
                        metadata.get("title", ""),
                        metadata.get("url", ""),
                        metadata.get("source", ""),
                        metadata.get("clean_text_path", ""),
                        metadata.get("scrape_timestamp", ""),
                    ),
                )
                inserted += 1
            conn.commit()
        return inserted

    def similarity_search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievalChunk]:
        if not query_vector:
            return []
        query_norm = _vector_norm(query_vector)
        if query_norm == 0:
            return []

        scored: list[RetrievalChunk] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT content_hash, chunk_text, embedding_json, title, url, source, clean_text_path, scrape_timestamp
                FROM article_chunks
                """
            ).fetchall()

        for row in rows:
            vector = [float(x) for x in json.loads(row[2])]
            score = _cosine_similarity(query_vector, query_norm, vector)
            if score <= 0:
                continue
            scored.append(
                RetrievalChunk(
                    score=score,
                    content_hash=row[0],
                    text=row[1],
                    title=row[3] or row[4] or "Untitled",
                    url=row[4] or "",
                    source=row[5] or "",
                    clean_text_path=row[6] or "",
                    scrape_timestamp=row[7] or "",
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            total_chunks = conn.execute("SELECT COUNT(*) FROM article_chunks").fetchone()[0]
            distinct_articles = conn.execute("SELECT COUNT(DISTINCT content_hash) FROM article_chunks").fetchone()[0]
        return {"chunks": int(total_chunks), "articles": int(distinct_articles)}


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


def _cosine_similarity(query: list[float], query_norm: float, vector: list[float]) -> float:
    if len(query) != len(vector):
        return 0.0
    denom = query_norm * _vector_norm(vector)
    if denom == 0:
        return 0.0
    dot = sum(q * v for q, v in zip(query, vector))
    return dot / denom


def chunk_text(text: str, max_chunk_words: int = 180, overlap_words: int = 35) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_chunk_words:
        return [" ".join(words)]

    chunks: list[str] = []
    step = max(max_chunk_words - overlap_words, 1)
    start = 0
    while start < len(words):
        end = min(start + max_chunk_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += step
    return chunks


def _load_vector_manifest(path: Path = DEFAULT_VECTOR_MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"version": 2, "articles": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 2, "articles": {}}
    # backward-compatible read for older payload shape
    entries = payload.get("articles", payload.get("indexed_articles", {}))
    if not isinstance(entries, dict):
        entries = {}
    return {"version": payload.get("version", 2), "articles": entries}


def _save_vector_manifest(manifest: dict[str, Any], path: Path = DEFAULT_VECTOR_MANIFEST_PATH) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def ingest_new_articles(
    client: OllamaClient | None = None,
    index: LocalVectorIndex | None = None,
    manifest_path: Path = DEFAULT_VECTOR_MANIFEST_PATH,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Index missing scraped articles from flat-file corpus into persistent local vector index."""
    ensure_data_directories(data_root=data_root)
    client = client or OllamaClient()
    index = index or LocalVectorIndex()

    missing_hashes = get_indexing_status(
        index=index,
        manifest_path=manifest_path,
        data_root=data_root,
    )["missing_content_hashes"]
    return index_missing_articles(
        missing_content_hashes=missing_hashes,
        client=client,
        index=index,
        manifest_path=manifest_path,
        data_root=data_root,
    )


def get_indexing_status(
    index: LocalVectorIndex | None = None,
    manifest_path: Path = DEFAULT_VECTOR_MANIFEST_PATH,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Report corpus/index synchronization status without mutating index state."""
    _ = index or LocalVectorIndex()
    ensure_data_directories(data_root=data_root)
    corpus_manifest = load_article_index(data_root=data_root)
    corpus_entries: dict[str, Any] = corpus_manifest.get("entries", {})
    vector_manifest = _load_vector_manifest(manifest_path)
    indexed_articles: dict[str, Any] = vector_manifest.get("articles", {})

    all_hashes = set(corpus_entries.keys())
    indexed_hashes = set(indexed_articles.keys())
    missing_hashes = sorted(all_hashes - indexed_hashes)
    return {
        "processed_articles_total": len(all_hashes),
        "indexed_articles_total": len(indexed_hashes.intersection(all_hashes)),
        "missing_articles_total": len(missing_hashes),
        "is_index_up_to_date": len(missing_hashes) == 0,
        "missing_content_hashes": missing_hashes,
        "vector_manifest_path": str(manifest_path),
    }


def index_missing_articles(
    missing_content_hashes: list[str],
    client: OllamaClient | None = None,
    index: LocalVectorIndex | None = None,
    manifest_path: Path = DEFAULT_VECTOR_MANIFEST_PATH,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Incrementally index only the specified missing content hashes."""
    ensure_data_directories(data_root=data_root)
    client = client or OllamaClient()
    index = index or LocalVectorIndex()

    article_manifest = load_article_index(data_root=data_root)
    article_entries: dict[str, Any] = article_manifest.get("entries", {})
    vector_manifest = _load_vector_manifest(manifest_path)
    indexed_articles: dict[str, Any] = vector_manifest.get("articles", {})

    newly_indexed_articles = 0
    newly_indexed_chunks = 0

    for content_hash in missing_content_hashes:
        entry = article_entries.get(content_hash, {})
        if not entry or content_hash in indexed_articles:
            continue

        file_paths = entry.get("file_paths", {})
        clean_text_path = Path(file_paths.get("clean_text_path", ""))
        if not clean_text_path.exists():
            continue
        text = clean_text_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        embeddings = [client.embed(chunk) for chunk in chunks]
        meta = {
            "title": str(entry.get("title", "")),
            "url": str(entry.get("article_url", "")),
            "source": str(entry.get("source_name", "")),
            "clean_text_path": str(clean_text_path),
            "scrape_timestamp": str(entry.get("scrape_timestamp", "")),
        }
        inserted = index.add_chunks(content_hash=content_hash, chunks=chunks, embeddings=embeddings, metadata=meta)
        if inserted <= 0:
            continue

        indexed_articles[content_hash] = {
            "content_hash": content_hash,
            "source": meta["source"],
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": inserted,
            "title": meta["title"],
            "url": meta["url"],
            "clean_text_path": meta["clean_text_path"],
            "scrape_timestamp": meta["scrape_timestamp"],
            "indexed": True,
        }
        newly_indexed_articles += 1
        newly_indexed_chunks += inserted

    vector_manifest["articles"] = indexed_articles
    final_manifest_path = _save_vector_manifest(vector_manifest, manifest_path)
    stats = index.stats()
    sync_status = get_indexing_status(index=index, manifest_path=manifest_path, data_root=data_root)

    return {
        "new_articles_indexed": newly_indexed_articles,
        "new_chunks_indexed": newly_indexed_chunks,
        "total_articles_indexed": stats["articles"],
        "total_chunks_indexed": stats["chunks"],
        "processed_articles_total": sync_status["processed_articles_total"],
        "missing_articles_total": sync_status["missing_articles_total"],
        "is_index_up_to_date": sync_status["is_index_up_to_date"],
        "vector_manifest_path": final_manifest_path,
        "vector_index_path": str(index.db_path),
    }


def answer_question(
    question: str,
    top_k: int = 5,
    client: OllamaClient | None = None,
    index: LocalVectorIndex | None = None,
) -> dict[str, Any]:
    cleaned_question = question.strip()
    if not cleaned_question:
        return {"ok": "false", "error": "Please enter a question first.", "answer": "", "citations": []}

    client = client or OllamaClient()
    index = index or LocalVectorIndex()
    verification = get_indexing_status(index=index)
    indexing_triggered = False
    indexing_result: dict[str, Any] = {}
    if verification["missing_articles_total"] > 0:
        indexing_triggered = True
        indexing_result = index_missing_articles(
            missing_content_hashes=verification["missing_content_hashes"],
            client=client,
            index=index,
        )
        verification = get_indexing_status(index=index)

    query_vector = client.embed(cleaned_question)
    retrieved = index.similarity_search(query_vector, top_k=top_k)
    if not retrieved:
        return {
            "ok": "true",
            "error": "",
            "answer": "Not enough relevant scraped data to answer this question yet.",
            "citations": [],
            "engine": "ollama-rag",
            "retrieval_count": 0,
            "index_verification": verification,
            "indexing_triggered": indexing_triggered,
            "indexing_result": indexing_result,
        }

    best_score = retrieved[0].score
    if best_score < 0.20:
        return {
            "ok": "true",
            "error": "",
            "answer": "Not enough relevant scraped data to answer this question yet.",
            "citations": [],
            "engine": "ollama-rag",
            "retrieval_count": len(retrieved),
            "index_verification": verification,
            "indexing_triggered": indexing_triggered,
            "indexing_result": indexing_result,
        }

    answer = client.chat(cleaned_question, retrieved)
    if not answer:
        answer = "Not enough relevant scraped data to answer this question yet."

    citations: list[dict[str, str]] = []
    seen: set[str] = set()
    for chunk in retrieved:
        key = chunk.url or chunk.clean_text_path
        if not key or key in seen:
            continue
        citations.append({"title": chunk.title, "url": chunk.url})
        seen.add(key)
        if len(citations) >= 5:
            break

    return {
        "ok": "true",
        "error": "",
        "answer": answer,
        "citations": citations,
        "engine": "ollama-rag",
        "retrieval_count": len(retrieved),
        "index_verification": verification,
        "indexing_triggered": indexing_triggered,
        "indexing_result": indexing_result,
    }
