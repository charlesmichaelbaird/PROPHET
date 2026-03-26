"""Local RAG pipeline for PROPHET using Ollama + persistent SQLite vector index."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any
from urllib.parse import urlparse

import requests

from mcp_server.storage import ensure_data_directories, load_article_index

DEFAULT_OLLAMA_BASE_URL = os.getenv("PROPHET_OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_EMBED_MODEL = os.getenv("PROPHET_OLLAMA_EMBED_MODEL", "").strip()
DEFAULT_OLLAMA_CHAT_MODEL = os.getenv("PROPHET_OLLAMA_MODEL", "").strip()
DEFAULT_OLLAMA_TIMEOUT_SECONDS = float(os.getenv("PROPHET_OLLAMA_TIMEOUT_SECONDS", "60"))
VECTOR_INDEX_FILENAME = "vector_store.sqlite"
VECTOR_MANIFEST_FILENAME = "vector_manifest.json"


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


def _slugify_fs(value: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-._")
    if not text:
        return "unknown"
    return text[:max_len].strip("-._") or "unknown"


def _normalize_embedding_model_name(embedding_model: str) -> str:
    return _slugify_fs(embedding_model or "default-embedding-model", max_len=80)


def _normalize_source_partition(entry: dict[str, Any]) -> str:
    source_name = str(entry.get("source_name", "")).strip().lower()
    source_url = str(entry.get("source_homepage_url", "")).strip().lower()
    article_url = str(entry.get("article_url", "")).strip().lower()
    source_hint = source_name or source_url or article_url
    if "apnews" in source_hint or "ap-news" in source_hint:
        return "ap-news"
    if "bbc" in source_hint:
        return "bbc"
    if "reuters" in source_hint:
        return "reuters"
    parsed = urlparse(source_url or article_url)
    host = parsed.netloc.lower().replace("www.", "")
    if host:
        return _slugify_fs(host.replace(".", "-"), max_len=60)
    return _slugify_fs(source_name or "unknown-source", max_len=60)


def _model_index_root(data_root: Path | None, embedding_model: str) -> Path:
    layout = ensure_data_directories(data_root=data_root)
    return layout["index"] / _normalize_embedding_model_name(embedding_model)


def _source_index_paths(data_root: Path | None, embedding_model: str, source: str) -> tuple[Path, Path]:
    source_dir = _model_index_root(data_root, embedding_model) / _slugify_fs(source, max_len=60)
    return source_dir / VECTOR_INDEX_FILENAME, source_dir / VECTOR_MANIFEST_FILENAME


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
        name = str(row.get("name", "")).strip()
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
        self.chat_timeout_seconds = float(
            os.getenv("PROPHET_OLLAMA_CHAT_TIMEOUT_SECONDS", str(max(timeout_seconds, 180.0)))
        )
        self.embedding_mode = "unknown"

    @staticmethod
    def _resolve_model_name(discovered: list[str], requested: str) -> str:
        requested_clean = requested.strip()
        if not requested_clean:
            return ""
        if requested_clean in discovered:
            return requested_clean
        for installed in discovered:
            if installed.startswith(f"{requested_clean}:") or requested_clean.startswith(f"{installed}:"):
                return installed
        return ""

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
            if discovered and not self._resolve_model_name(discovered, self.embed_model):
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
        resolved_chat_model = self._resolve_model_name(discovered, requested_chat_model) if discovered else requested_chat_model
        if discovered and not resolved_chat_model:
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
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": resolved_chat_model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=self.chat_timeout_seconds,
            )
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Ollama chat timed out after {self.chat_timeout_seconds:.0f}s for model "
                f"'{resolved_chat_model}'. Try a smaller/faster model (for example llama3) "
                "or increase PROPHET_OLLAMA_CHAT_TIMEOUT_SECONDS."
            ) from exc
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"Selected answer model '{resolved_chat_model}' could not serve chat/generation: {exc}"
            ) from exc
        payload = response.json()
        message = payload.get("message", {})
        return str(message.get("content", "")).strip()


class LocalVectorIndex:
    def __init__(self, db_path: Path) -> None:
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


def _load_vector_manifest(path: Path) -> dict[str, Any]:
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


def _save_vector_manifest(manifest: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def ingest_new_articles(
    client: OllamaClient | None = None,
    index: LocalVectorIndex | None = None,
    manifest_path: Path | None = None,
    data_root: Path | None = None,
    progress_callback: Any | None = None,
    embedding_model: str = "",
    answer_model: str = "",
    source_partition: str = "",
) -> dict[str, Any]:
    """Index missing scraped articles from flat-file corpus into persistent local vector index."""
    ensure_data_directories(data_root=data_root)
    client = client or OllamaClient(embed_model=embedding_model, chat_model=answer_model)
    if index is not None and manifest_path is not None:
        missing_hashes = get_indexing_status(
            index=index,
            manifest_path=manifest_path,
            data_root=data_root,
            embedding_model=embedding_model,
        )["missing_content_hashes"]
        return index_missing_articles(
            missing_content_hashes=missing_hashes,
            client=client,
            index=index,
            manifest_path=manifest_path,
            data_root=data_root,
            progress_callback=progress_callback,
            embedding_model=embedding_model,
        )

    selected_embedding_model = embedding_model or DEFAULT_OLLAMA_EMBED_MODEL or "default-embedding-model"

    missing_hashes = get_indexing_status(
        index=None,
        manifest_path=None,
        data_root=data_root,
        embedding_model=selected_embedding_model,
    )["missing_content_hashes"]
    if source_partition.strip():
        normalized_source = _slugify_fs(source_partition.strip().lower(), max_len=60)
        source_aliases = {
            "apnews-com": "ap-news",
            "www-bbc-com": "bbc",
            "bbc-com": "bbc",
        }
        normalized_source = source_aliases.get(normalized_source, normalized_source)
        article_entries, _ = _processed_article_entries(data_root=data_root)
        missing_hashes = [
            content_hash
            for content_hash in missing_hashes
            if _normalize_source_partition(article_entries.get(content_hash, {})) == normalized_source
        ]
    return index_missing_articles(
        missing_content_hashes=missing_hashes,
        client=client,
        index=None,
        manifest_path=None,
        data_root=data_root,
        progress_callback=progress_callback,
        embedding_model=selected_embedding_model,
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


def _group_entries_by_source(entries: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for content_hash, entry in entries.items():
        source_partition = _normalize_source_partition(entry)
        grouped.setdefault(source_partition, {})[content_hash] = entry
    return grouped


def get_indexing_status(
    index: LocalVectorIndex | None = None,
    manifest_path: Path | None = None,
    data_root: Path | None = None,
    embedding_model: str = "",
) -> dict[str, Any]:
    """Report corpus/index synchronization status without mutating index state."""
    if manifest_path is not None:
        _ = index
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

    explicit_index = index
    corpus_entries, processed_root = _processed_article_entries(data_root=data_root)
    grouped_entries = _group_entries_by_source(corpus_entries)
    selected_embedding_model = embedding_model or DEFAULT_OLLAMA_EMBED_MODEL or "default-embedding-model"
    model_root = _model_index_root(data_root=data_root, embedding_model=selected_embedding_model)

    indexed_hashes: set[str] = set()
    source_counts: dict[str, int] = {}
    source_directories: list[str] = []
    for source_name, source_entries in grouped_entries.items():
        _, source_manifest_path = _source_index_paths(
            data_root=data_root,
            embedding_model=selected_embedding_model,
            source=source_name,
        )
        vector_manifest = _load_vector_manifest(source_manifest_path)
        indexed_articles: dict[str, Any] = vector_manifest.get("articles", {})
        source_indexed_hashes = set(indexed_articles.keys()).intersection(set(source_entries.keys()))
        indexed_hashes.update(source_indexed_hashes)
        source_counts[source_name] = len(source_indexed_hashes)
        source_directories.append(str(source_manifest_path.parent))

    all_hashes = set(corpus_entries.keys())
    missing_hashes = sorted(all_hashes - indexed_hashes)
    return {
        "processed_articles_total": len(all_hashes),
        "indexed_articles_total": len(indexed_hashes.intersection(all_hashes)),
        "missing_articles_total": len(missing_hashes),
        "is_index_up_to_date": len(missing_hashes) == 0,
        "missing_content_hashes": missing_hashes,
        "vector_manifest_path": str(model_root / "*" / VECTOR_MANIFEST_FILENAME),
        "active_embedding_model": selected_embedding_model,
        "active_model_index_root": str(model_root),
        "source_directories": sorted(set(source_directories)),
        "indexed_counts_by_source": source_counts,
        "processed_scan_directory": str(processed_root),
    }


def index_missing_articles(
    missing_content_hashes: list[str],
    client: OllamaClient | None = None,
    index: LocalVectorIndex | None = None,
    manifest_path: Path | None = None,
    data_root: Path | None = None,
    progress_callback: Any | None = None,
    embedding_model: str = "",
) -> dict[str, Any]:
    """Incrementally index only the specified missing content hashes."""
    ensure_data_directories(data_root=data_root)
    client = client or OllamaClient()
    if index is not None and manifest_path is not None:
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
            clean_text_path = Path(entry.get("file_paths", {}).get("clean_text_path", ""))
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
            chunks = chunk_text(text)
            if not chunks:
                continue
            embeddings = [client.embed(chunk) for chunk in chunks]
            inserted = index.add_chunks(
                content_hash=content_hash,
                chunks=chunks,
                embeddings=embeddings,
                metadata={
                    "title": str(entry.get("title", "")),
                    "url": str(entry.get("article_url", "")),
                    "source": str(entry.get("source_name", "")),
                    "clean_text_path": str(clean_text_path),
                    "scrape_timestamp": str(entry.get("scrape_timestamp", "")),
                },
            )
            if inserted <= 0:
                continue
            indexed_articles[content_hash] = {
                "content_hash": content_hash,
                "source": str(entry.get("source_name", "")),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
                "chunk_count": inserted,
                "title": str(entry.get("title", "")),
                "url": str(entry.get("article_url", "")),
                "clean_text_path": str(clean_text_path),
                "scrape_timestamp": str(entry.get("scrape_timestamp", "")),
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
            "embedding_mode": client.embedding_mode,
            "total_discovered": total_discovered,
            "eligible_for_indexing": len(missing_content_hashes),
            "already_indexed_articles": already_indexed,
            "processed_scan_directory": str(processed_root),
            "vector_manifest_path": final_manifest_path,
            "vector_index_path": str(index.db_path),
            "selected_embedding_model": embedding_model or getattr(client, "embed_model", ""),
            "selected_answer_model": getattr(client, "chat_model", ""),
        }

    _ = index
    _ = manifest_path
    selected_embedding_model = (
        embedding_model
        or getattr(client, "embed_model", "")
        or DEFAULT_OLLAMA_EMBED_MODEL
        or "default-embedding-model"
    )

    article_entries, processed_root = _processed_article_entries(data_root=data_root)
    source_entries = _group_entries_by_source(article_entries)
    source_indexes: dict[str, LocalVectorIndex] = {}
    source_manifests: dict[str, dict[str, Any]] = {}
    source_manifest_paths: dict[str, Path] = {}

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
        if not entry:
            continue

        source_partition = _normalize_source_partition(entry)
        if source_partition not in source_indexes:
            source_index_path, source_manifest_path = _source_index_paths(
                data_root=data_root,
                embedding_model=selected_embedding_model,
                source=source_partition,
            )
            source_indexes[source_partition] = LocalVectorIndex(db_path=source_index_path)
            source_manifest_paths[source_partition] = source_manifest_path
            source_manifests[source_partition] = _load_vector_manifest(source_manifest_path)

        indexed_articles: dict[str, Any] = source_manifests[source_partition].get("articles", {})
        if content_hash in indexed_articles:
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
                    "source_partition": source_partition,
                }
            )
        inserted = source_indexes[source_partition].add_chunks(
            content_hash=content_hash,
            chunks=chunks,
            embeddings=embeddings,
            metadata=meta,
        )
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
        source_manifests[source_partition]["articles"] = indexed_articles
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

    source_manifest_outputs: dict[str, str] = {}
    aggregate_articles = 0
    aggregate_chunks = 0
    for source_partition, source_manifest in source_manifests.items():
        source_manifest_outputs[source_partition] = _save_vector_manifest(
            source_manifest,
            source_manifest_paths[source_partition],
        )
        source_stats = source_indexes[source_partition].stats()
        aggregate_articles += source_stats["articles"]
        aggregate_chunks += source_stats["chunks"]

    sync_status = get_indexing_status(
        index=None,
        manifest_path=None,
        data_root=data_root,
        embedding_model=selected_embedding_model,
    )
    selected_answer_model = getattr(client, "chat_model", "")

    return {
        "new_articles_indexed": newly_indexed_articles,
        "new_chunks_indexed": newly_indexed_chunks,
        "total_articles_indexed": aggregate_articles,
        "total_chunks_indexed": aggregate_chunks,
        "processed_articles_total": sync_status["processed_articles_total"],
        "missing_articles_total": sync_status["missing_articles_total"],
        "is_index_up_to_date": sync_status["is_index_up_to_date"],
        "embedding_mode": client.embedding_mode,
        "total_discovered": total_discovered,
        "eligible_for_indexing": len(missing_content_hashes),
        "already_indexed_articles": already_indexed,
        "processed_scan_directory": str(processed_root),
        "vector_manifest_path": str(_model_index_root(data_root=data_root, embedding_model=selected_embedding_model)),
        "vector_manifest_paths_by_source": source_manifest_outputs,
        "vector_index_path": str(_model_index_root(data_root=data_root, embedding_model=selected_embedding_model)),
        "active_model_index_root": str(_model_index_root(data_root=data_root, embedding_model=selected_embedding_model)),
        "indexed_counts_by_source": sync_status.get("indexed_counts_by_source", {}),
        "selected_embedding_model": selected_embedding_model,
        "selected_answer_model": selected_answer_model,
        "sources_discovered": sorted(source_entries.keys()),
    }


def answer_question(
    question: str,
    top_k: int = 5,
    client: OllamaClient | None = None,
    index: LocalVectorIndex | None = None,
    embedding_model: str = "",
    answer_model: str = "",
) -> dict[str, Any]:
    started_at = time.perf_counter()

    def _diag(
        *,
        stage: str,
        retrieval_count: int = 0,
        top_score: float = 0.0,
        embed_ms: int = 0,
        retrieval_ms: int = 0,
        chat_ms: int = 0,
    ) -> dict[str, Any]:
        return {
            "stage": stage,
            "retrieval_count": retrieval_count,
            "top_score": top_score,
            "embed_ms": embed_ms,
            "retrieval_ms": retrieval_ms,
            "chat_ms": chat_ms,
            "total_ms": int((time.perf_counter() - started_at) * 1000),
        }

    cleaned_question = question.strip()
    if not cleaned_question:
        return {
            "ok": "false",
            "error": "Please enter a question first.",
            "answer": "",
            "citations": [],
            "diagnostics": _diag(stage="idle"),
        }

    client = client or OllamaClient(embed_model=embedding_model, chat_model=answer_model)
    explicit_index = index
    selected_embedding_model = (
        embedding_model
        or getattr(client, "embed_model", "")
        or DEFAULT_OLLAMA_EMBED_MODEL
        or "default-embedding-model"
    )
    selected_answer_model = answer_model or getattr(client, "chat_model", answer_model)
    verification = get_indexing_status(index=None, manifest_path=None, embedding_model=selected_embedding_model)
    indexing_triggered = False
    indexing_result: dict[str, Any] = {}
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
            "diagnostics": _diag(stage="indexing_required"),
        }

    embed_started = time.perf_counter()
    query_vector = client.embed(cleaned_question)
    embed_ms = int((time.perf_counter() - embed_started) * 1000)
    retrieval_started = time.perf_counter()
    retrieved: list[RetrievalChunk] = []
    if explicit_index is not None:
        retrieved = explicit_index.similarity_search(query_vector, top_k=top_k)
    else:
        model_root = _model_index_root(data_root=None, embedding_model=selected_embedding_model)
        source_dirs = [
            item for item in model_root.iterdir() if item.is_dir() and (item / VECTOR_INDEX_FILENAME).exists()
        ] if model_root.exists() else []
        for source_dir in source_dirs:
            source_index = LocalVectorIndex(db_path=source_dir / VECTOR_INDEX_FILENAME)
            retrieved.extend(source_index.similarity_search(query_vector, top_k=top_k))
        retrieved.sort(key=lambda item: item.score, reverse=True)
        retrieved = retrieved[:top_k]
    retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)
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
            "diagnostics": _diag(stage="retrieved_none", embed_ms=embed_ms, retrieval_ms=retrieval_ms),
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
            "diagnostics": _diag(
                stage="retrieved_low_confidence",
                retrieval_count=len(retrieved),
                top_score=best_score,
                embed_ms=embed_ms,
                retrieval_ms=retrieval_ms,
            ),
        }

    chat_started = time.perf_counter()
    answer = client.chat(cleaned_question, retrieved)
    chat_ms = int((time.perf_counter() - chat_started) * 1000)
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
        "diagnostics": _diag(
            stage="completed",
            retrieval_count=len(retrieved),
            top_score=best_score,
            embed_ms=embed_ms,
            retrieval_ms=retrieval_ms,
            chat_ms=chat_ms,
        ),
    }
