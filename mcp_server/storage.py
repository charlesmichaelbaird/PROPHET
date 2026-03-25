"""Local file-based persistence helpers for scraped article data."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"



def get_data_root(data_root: Path | None = None) -> Path:
    """Return the local data root, optionally overridden for tests/future config."""
    return data_root or DEFAULT_DATA_DIR



def ensure_data_directories(data_root: Path | None = None) -> dict[str, Path]:
    """Create and return standard data directories used by local persistence."""
    root = get_data_root(data_root)
    layout = {
        "root": root,
        "raw": root / "raw",
        "processed": root / "processed",
        "index": root / "index",
        "cache": root / "cache",
        "exports": root / "exports",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout



def _slugify(value: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    if not text:
        return "untitled"
    return text[:max_len].strip("-") or "untitled"



def _safe_source_name(homepage_url: str) -> str:
    netloc = urlparse(homepage_url).netloc.lower() or "unknown-source"
    return _slugify(netloc.replace(".", "-"), max_len=60)



def _article_id(article_url: str, title: str) -> str:
    digest = hashlib.sha256(article_url.encode("utf-8")).hexdigest()[:12]
    return f"{digest}-{_slugify(title, max_len=50)}"



def build_article_paths(
    homepage_url: str,
    article_url: str,
    article_title: str,
    scrape_time: datetime,
    data_root: Path | None = None,
) -> dict[str, Path]:
    """Build deterministic article file paths under raw/ and processed/."""
    layout = ensure_data_directories(data_root)
    source = _safe_source_name(homepage_url)
    day_bucket = scrape_time.strftime("%Y-%m-%d")
    article_key = _article_id(article_url, article_title or article_url)

    raw_base = layout["raw"] / source / day_bucket
    processed_base = layout["processed"] / source / day_bucket / article_key
    raw_base.mkdir(parents=True, exist_ok=True)
    processed_base.mkdir(parents=True, exist_ok=True)

    return {
        "raw_html": raw_base / f"{article_key}.html",
        "metadata": processed_base / "metadata.json",
        "clean_text": processed_base / "clean_text.txt",
        "processed_dir": processed_base,
        "article_key": Path(article_key),
    }



def write_article_files(
    homepage_url: str,
    source_homepage_url: str,
    article_url: str,
    title: str,
    scrape_timestamp: datetime,
    clean_text: str,
    article_html: str = "",
    article_metadata: dict[str, Any] | None = None,
    data_root: Path | None = None,
) -> dict[str, str]:
    """Persist a scraped article to local flat files and return written paths."""
    paths = build_article_paths(
        homepage_url=homepage_url,
        article_url=article_url,
        article_title=title,
        scrape_time=scrape_timestamp,
        data_root=data_root,
    )

    if article_html:
        paths["raw_html"].write_text(article_html, encoding="utf-8")

    paths["clean_text"].write_text(clean_text, encoding="utf-8")

    payload = {
        "source_name": _safe_source_name(homepage_url),
        "source_homepage_url": source_homepage_url,
        "article_url": article_url,
        "title": title,
        "scrape_timestamp": scrape_timestamp.astimezone(timezone.utc).isoformat(),
        "raw_html_path": str(paths["raw_html"]),
        "clean_text_path": str(paths["clean_text"]),
    }
    if article_metadata:
        payload["article_metadata"] = article_metadata

    paths["metadata"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "metadata_path": str(paths["metadata"]),
        "clean_text_path": str(paths["clean_text"]),
        "raw_html_path": str(paths["raw_html"]),
        "processed_dir": str(paths["processed_dir"]),
    }



def write_run_index(
    homepage_url: str,
    scrape_timestamp: datetime,
    summary_payload: dict[str, Any],
    data_root: Path | None = None,
) -> str:
    """Persist a lightweight run-level index record for future retrieval/exports."""
    layout = ensure_data_directories(data_root)
    source = _safe_source_name(homepage_url)
    day_bucket = scrape_timestamp.strftime("%Y-%m-%d")
    stamp = scrape_timestamp.strftime("%Y%m%dT%H%M%SZ")
    run_hash = hashlib.sha256(f"{homepage_url}-{stamp}".encode("utf-8")).hexdigest()[:10]

    run_dir = layout["index"] / source / day_bucket
    run_dir.mkdir(parents=True, exist_ok=True)
    run_file = run_dir / f"run-{stamp}-{run_hash}.json"
    run_file.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(run_file)
