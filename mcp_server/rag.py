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
DEFAULT_OLLAMA_EMBED_MODEL = os.getenv("PROPHET_OLLAMA_EMBED_MODEL", "").strip()
DEFAULT_OLLAMA_CHAT_MODEL = os.getenv("PROPHET_OLLAMA_MODEL", "").strip()
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


def _base_model_name(model_name: str) -> str:
    cleaned = str(model_name).strip()
    if ":" in cleaned:
        return cleaned.split(":", 1)[0].strip()
    return cleaned


def discover_ollama_models(base_url: str = DEFAULT_OLLAMA_BASE_URL, timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Discover locally installed Ollama models and derive role-friendly candidate lists."""
    normalized_url = base_url.rstrip("/")
    try:
        response = requests.get(f"{normalized_url}/api/tags", timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        raw_models = payload.get("models", [])
    except Exception as exc:
        return {
            "available": False,
            "host": normalized_url,
            "models": [],
            "embedding_candidates": [],
            "answer_candidates": [],
            "error": f"Ollama runtime unavailable at {normalized_url}: {exc}",
        }

    discovered: list[str] = []
    for row in raw_models:
        name = _base_model_name(str(row.get("name", "")))
        if name and name not in discovered:
            discovered.append(name)

    embedding_candidates = [
        name for name in discovered if any(marker in name.lower() for marker in ("embed", "nomic", "mxbai", "e5", "bge"))
    ]
    answer_candidates = [name for name in discovered if name not in embedding_candidates]

    return {
        "available": True,
        "host": normalized_url,
        "models": discovered,
        "embedding_candidates": embedding_candidates,
        "answer_candidates": answer_candidates or discovered,
        "error": "",
    }


class OllamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        embed_model: str = DEFAULT_OLLAMA_EMBED_MODEL,
        chat_model: str = DEFAULT_OLLAMA_CHAT_MODEL,
        timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model.strip()
        self.chat_model = chat_model.strip()
        self.timeout_seconds = timeout_seconds
        self.embedding_mode = "unknown"

    def embed(self, text: str) -> list[float]:
        """Embed text using native Ollama /api/embed endpoint."""
        candidate_models = self._candidate_embedding_models()
        if not candidate_models:
            raise RuntimeError(
                "No embedding model could be resolved. Choose a valid embedding model from local Ollama discovery."
            )
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
        discovered = self._discover_installed_models()
        if self.embed_model:
            if discovered and self.embed_model not in discovered:
                raise RuntimeError(
                    f"Selected embedding model '{self.embed_model}' is not installed in local Ollama."
                )
            return [self.embed_model]

        embed_like = [name for name in discovered if any(marker in name.lower() for marker in ("embed", "nomic", "mxbai"))]
        all_names = embed_like + discovered
        deduped: list[str] = []
        for name in all_names:
            clean = str(name).strip()
            if clean and clean not in deduped:
                deduped.append(clean)
        return deduped

    def _discover_installed_models(self) -> list[str]:
        discovered = discover_ollama_models(base_url=self.base_url, timeout_seconds=self.timeout_seconds)
        return discovered.get("models", [])

    def _embed_with_model(self, text: str, model_name: str) -> list[float]:
        return self._request_embedding_vector(model_name=model_name, text=text)

    def _request_embedding_vector(self, model_name: str, text: str) -> list[float]:
        """
        Request embeddings with defensive compatibility handling.

        Preferred path:
          1) POST /api/embed      body: {"model": "...", "input": ["..."]}
        Legacy fallback:
          2) POST /api/embeddings body: {"model": "...", "prompt": "..."}
        """
        current_error = ""
        current_response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": model_name, "input": [text]},
            timeout=self.timeout_seconds,
        )
        if current_response.status_code == 200:
            payload = current_response.json()
            embeddings = payload.get("embeddings", [])
            if embeddings:
                self.embedding_mode = "current:/api/embed"
                return [float(x) for x in embeddings[0]]
            current_error = "current endpoint returned empty embeddings payload"
        else:
            current_error = f"current endpoint status={current_response.status_code}"

        legacy_error = ""
        legacy_response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model_name, "prompt": text},
            timeout=self.timeout_seconds,
        )
        if legacy_response.status_code == 200:
            payload = legacy_response.json()
            vector = payload.get("embedding", [])
            if vector:
                self.embedding_mode = "legacy:/api/embeddings"
                return [float(x) for x in vector]
            legacy_error = "legacy endpoint returned empty embedding payload"
        else:
            legacy_error = f"legacy endpoint status={legacy_response.status_code}"

        raise RuntimeError(
            f"model '{model_name}' embedding failed across endpoint variants; "
            f"{current_error}; {legacy_error}. "
            "This may indicate an Ollama version mismatch, missing model, or unavailable runtime."
        )

    def chat(self, question: str, chunks: list[RetrievalChunk]) -> str:
        discovered = self._discover_installed_models()
        requested_chat_model = self.chat_model
        if not requested_chat_model and discovered:
            requested_chat_model = discovered[0]
        if not requested_chat_model:
            raise RuntimeError("No answer model is selected and no local Ollama models were discovered.")
        if discovered and requested_chat_model not in discovered:
            raise RuntimeError(
                f"Selected answer model '{requested_chat_model}' is not installed in local Ollama."
            )

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
                "model": requested_chat_model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=self.timeout_seconds,
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"Selected answer model '{requested_chat_model}' could not serve chat/generation: {exc}"
            ) from exc
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
    progress_callback: Any | None = None,
    embedding_model: str = "",
    answer_model: str = "",
) -> dict[str, Any]:
    """Index missing scraped articles from flat-file corpus into persistent local vector index."""
    ensure_data_directories(data_root=data_root)
    client = client or OllamaClient(embed_model=embedding_model, chat_model=answer_model)
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
        progress_callback=progress_callback,
    )


def _processed_article_entries(data_root: Path | None = None) -> tuple[dict[str, Any], Path]:
    """Return manifest entries that point to valid processed clean-text files only."""
    layout = ensure_data_directories(data_root=data_root)
    processed_root = layout["processed"].resolve()
    corpus_manifest = load_article_index(data_root=data_root)
    corpus_entries: dict[str, Any] = corpus_manifest.get("entries", {})
    filtered: dict[str, Any] = {}
    for content_hash, entry in corpus_entries.items():
        clean_text_path = Path(entry.get("file_paths", {}).get("clean_text_path", ""))
        if not clean_text_path.exists():
            continue
        try:
            resolved = clean_text_path.resolve()
        except Exception:
            continue
        if processed_root not in resolved.parents:
            continue
        filtered[content_hash] = entry
    return filtered, processed_root


def get_indexing_status(
    index: LocalVectorIndex | None = None,
    manifest_path: Path = DEFAULT_VECTOR_MANIFEST_PATH,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Report corpus/index synchronization status without mutating index state."""
    _ = index or LocalVectorIndex()
    corpus_entries, processed_root = _processed_article_entries(data_root=data_root)
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
        "processed_scan_directory": str(processed_root),
    }


def index_missing_articles(
    missing_content_hashes: list[str],
    client: OllamaClient | None = None,
    index: LocalVectorIndex | None = None,
    manifest_path: Path = DEFAULT_VECTOR_MANIFEST_PATH,
    data_root: Path | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Incrementally index only the specified missing content hashes."""
    ensure_data_directories(data_root=data_root)
    client = client or OllamaClient()
    index = index or LocalVectorIndex()

    article_entries, processed_root = _processed_article_entries(data_root=data_root)
    vector_manifest = _load_vector_manifest(manifest_path)
    indexed_articles: dict[str, Any] = vector_manifest.get("articles", {})

    newly_indexed_articles = 0
    newly_indexed_chunks = 0
    total_discovered = len(article_entries)
    already_indexed = max(total_discovered - len(missing_content_hashes), 0)
    if progress_callback:
        progress_callback(
            {
                "event": "scan",
                "processed_scan_directory": str(processed_root),
                "total_discovered": total_discovered,
                "eligible_for_indexing": len(missing_content_hashes),
                "already_indexed": already_indexed,
                "remaining": len(missing_content_hashes),
            }
        )

    for article_pos, content_hash in enumerate(missing_content_hashes, start=1):
        entry = article_entries.get(content_hash, {})
        if not entry or content_hash in indexed_articles:
            continue

        file_paths = entry.get("file_paths", {})
        clean_text_path = Path(file_paths.get("clean_text_path", ""))
        if not clean_text_path.exists():
            continue
        if progress_callback:
            progress_callback(
                {
                    "event": "step",
                    "content_hash": content_hash,
                    "article_position": article_pos,
                    "total_to_index": len(missing_content_hashes),
                    "step": "loading file",
                    "title": str(entry.get("title", "")),
                    "file_path": str(clean_text_path),
                }
            )
        text = clean_text_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        if progress_callback:
            progress_callback(
                {
                    "event": "step",
                    "content_hash": content_hash,
                    "article_position": article_pos,
                    "total_to_index": len(missing_content_hashes),
                    "step": "chunking text",
                    "title": str(entry.get("title", "")),
                }
            )
        chunks = chunk_text(text)
        if not chunks:
            continue

        if progress_callback:
            progress_callback(
                {
                    "event": "step",
                    "content_hash": content_hash,
                    "article_position": article_pos,
                    "total_to_index": len(missing_content_hashes),
                    "step": "generating embeddings",
                    "title": str(entry.get("title", "")),
                    "chunk_count": len(chunks),
                }
            )
        embeddings = [client.embed(chunk) for chunk in chunks]
        meta = {
            "title": str(entry.get("title", "")),
            "url": str(entry.get("article_url", "")),
            "source": str(entry.get("source_name", "")),
            "clean_text_path": str(clean_text_path),
            "scrape_timestamp": str(entry.get("scrape_timestamp", "")),
        }
        if progress_callback:
            progress_callback(
                {
                    "event": "step",
                    "content_hash": content_hash,
                    "article_position": article_pos,
                    "total_to_index": len(missing_content_hashes),
                    "step": "writing to index",
                    "title": meta["title"],
                }
            )
        inserted = index.add_chunks(content_hash=content_hash, chunks=chunks, embeddings=embeddings, metadata=meta)
        if inserted <= 0:
            continue

        if progress_callback:
            progress_callback(
                {
                    "event": "step",
                    "content_hash": content_hash,
                    "article_position": article_pos,
                    "total_to_index": len(missing_content_hashes),
                    "step": "marking indexed",
                    "title": meta["title"],
                }
            )
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
        if progress_callback:
            progress_callback(
                {
                    "event": "article_done",
                    "indexed_so_far": newly_indexed_articles,
                    "total_to_index": len(missing_content_hashes),
                    "already_indexed": already_indexed,
                    "remaining": max(len(missing_content_hashes) - article_pos, 0),
                    "title": meta["title"],
                }
            )

    vector_manifest["articles"] = indexed_articles
    final_manifest_path = _save_vector_manifest(vector_manifest, manifest_path)
    stats = index.stats()
    sync_status = get_indexing_status(index=index, manifest_path=manifest_path, data_root=data_root)
    selected_embedding_model = getattr(client, "embed_model", "")
    selected_answer_model = getattr(client, "chat_model", "")

    return {
        "new_articles_indexed": newly_indexed_articles,
        "new_chunks_indexed": newly_indexed_chunks,
        "total_articles_indexed": stats["articles"],
        "total_chunks_indexed": stats["chunks"],
        "processed_articles_total": sync_status["processed_articles_total"],
        "missing_articles_total": sync_status["missing_articles_total"],
        "is_index_up_to_date": sync_status["is_index_up_to_date"],
        "embedding_mode": client.embedding_mode,
        "total_discovered": total_discovered,
        "eligible_for_indexing": len(missing_content_hashes),
        "already_indexed_articles": already_indexed,
        "processed_scan_directory": str(processed_root),
        "vector_manifest_path": final_manifest_path,
        "vector_index_path": str(index.db_path),
        "selected_embedding_model": selected_embedding_model,
        "selected_answer_model": selected_answer_model,
    }


def answer_question(
    question: str,
    top_k: int = 5,
    client: OllamaClient | None = None,
    index: LocalVectorIndex | None = None,
    embedding_model: str = "",
    answer_model: str = "",
) -> dict[str, Any]:
    cleaned_question = question.strip()
    if not cleaned_question:
        return {"ok": "false", "error": "Please enter a question first.", "answer": "", "citations": []}

    client = client or OllamaClient(embed_model=embedding_model, chat_model=answer_model)
    index = index or LocalVectorIndex()
    verification = get_indexing_status(index=index)
    indexing_triggered = False
    indexing_result: dict[str, Any] = {}
    selected_embedding_model = getattr(client, "embed_model", embedding_model)
    selected_answer_model = getattr(client, "chat_model", answer_model)
    if verification["missing_articles_total"] > 0:
        return {
            "ok": "true",
            "error": "",
            "answer": (
                "Your local corpus has unindexed scraped articles. "
                "Click 'Index Data' to index saved articles, then ask again."
            ),
            "citations": [],
            "engine": "ollama-rag",
            "retrieval_count": 0,
            "index_verification": verification,
            "indexing_triggered": False,
            "indexing_result": {},
            "embedding_mode": client.embedding_mode,
            "selected_embedding_model": selected_embedding_model,
            "selected_answer_model": selected_answer_model,
        }

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
            "embedding_mode": client.embedding_mode,
            "selected_embedding_model": selected_embedding_model,
            "selected_answer_model": selected_answer_model,
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
            "embedding_mode": client.embedding_mode,
            "selected_embedding_model": selected_embedding_model,
            "selected_answer_model": selected_answer_model,
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
        "embedding_mode": client.embedding_mode,
        "selected_embedding_model": selected_embedding_model,
        "selected_answer_model": selected_answer_model,
    }
