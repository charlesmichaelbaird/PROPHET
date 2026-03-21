"""Tool helpers for zero-cost homepage article analysis."""

from __future__ import annotations

import re
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, urlunparse

import requests

_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
    "yourselves", "will", "also", "said", "says", "say", "get", "got", "like", "one", "two", "new",
}


class ArticleLinkParser(HTMLParser):
    """Collect anchor URLs and visible anchor text."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self._current_href = ""
        self._text_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag.lower() != "a":
            return

        href = ""
        for key, value in attrs:
            if key.lower() == "href" and value:
                href = value.strip()
                break

        self._current_href = href
        self._text_chunks = []

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._current_href:
            cleaned = data.strip()
            if cleaned:
                self._text_chunks.append(cleaned)

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag.lower() != "a" or not self._current_href:
            return

        title = " ".join(self._text_chunks).strip()
        self.links.append({"url": self._current_href, "title": title})
        self._current_href = ""
        self._text_chunks = []


class ArticleTextParser(HTMLParser):
    """Extract paragraph-like text while skipping hidden or noisy tags."""

    _SKIP_TAGS = {"script", "style", "noscript", "header", "footer", "nav", "aside", "form", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._in_paragraph = False
        self._paragraph_parts: list[str] = []
        self._paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        lowered = tag.lower()
        if lowered in self._SKIP_TAGS:
            self._skip_depth += 1
        if lowered == "p" and self._skip_depth == 0:
            self._in_paragraph = True
            self._paragraph_parts = []

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        lowered = tag.lower()
        if lowered in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if lowered == "p" and self._in_paragraph:
            paragraph_text = " ".join(self._paragraph_parts).strip()
            if paragraph_text:
                self._paragraphs.append(paragraph_text)
            self._in_paragraph = False
            self._paragraph_parts = []

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth > 0 or not self._in_paragraph:
            return

        cleaned = data.strip()
        if cleaned:
            self._paragraph_parts.append(cleaned)

    def get_text(self) -> str:
        return " ".join(self._paragraphs)


def fetch_url(url: str, timeout: int = 10) -> str:
    """Fetch raw HTML for a URL with basic validation and error messages."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must include http:// or https:// and a valid host")

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "PROPHET-ZeroCost/1.0 (+https://example.local)"},
        )
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Failed to fetch URL: {exc}") from exc


def _normalize_url(candidate_url: str, base_url: str) -> str:
    full = urljoin(base_url, candidate_url)
    parsed = urlparse(full)
    clean = parsed._replace(fragment="", query="")
    return urlunparse(clean)


def _is_probable_article_link(url: str, source_host: str) -> bool:
    parsed = urlparse(url)
    if not parsed.netloc:
        return False

    if source_host.endswith("apnews.com"):
        return parsed.path.startswith("/article/")

    blocked_fragments = ("/video", "/live", "/sports", "/gallery", "/tag/")
    if any(fragment in parsed.path for fragment in blocked_fragments):
        return False

    path_parts = [part for part in parsed.path.split("/") if part]
    return len(path_parts) >= 2


def extract_article_links(homepage_html: str, homepage_url: str, max_links: int = 40) -> list[dict[str, str]]:
    """Extract likely article links from a homepage."""
    source_host = urlparse(homepage_url).netloc.lower()
    parser = ArticleLinkParser()
    parser.feed(homepage_html)
    parser.close()

    links: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for raw in parser.links:
        href = raw["url"]
        if not href:
            continue
        normalized = _normalize_url(href, homepage_url)
        parsed = urlparse(normalized)
        if not parsed.netloc.lower().endswith(source_host):
            continue
        if not _is_probable_article_link(normalized, source_host):
            continue
        if normalized in seen_urls:
            continue

        links.append({"url": normalized, "title": raw["title"]})
        seen_urls.add(normalized)
        if len(links) >= max_links:
            break

    return links


def extract_article_text(article_html: str) -> str:
    """Extract article body text with a simple paragraph strategy."""
    parser = ArticleTextParser()
    parser.feed(article_html)
    parser.close()

    clean = re.sub(r"\s+", " ", unescape(parser.get_text())).strip()
    return clean


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]{3,}", text.lower())


def analyze_homepage(homepage_url: str, max_articles: int = 20) -> dict:
    """Scrape homepage article links and compute top-word metrics without LLMs."""
    homepage_html = fetch_url(homepage_url)
    link_entries = extract_article_links(homepage_html, homepage_url, max_links=max_articles * 3)

    word_totals: Counter[str] = Counter()
    article_frequency: Counter[str] = Counter()
    scraped_articles: list[dict[str, str]] = []

    for link in link_entries:
        if len(scraped_articles) >= max_articles:
            break

        try:
            article_html = fetch_url(link["url"], timeout=12)
            article_text = extract_article_text(article_html)
            if not article_text:
                continue

            tokens = [token for token in _tokenize(article_text) if token not in _STOPWORDS]
            if not tokens:
                continue

            token_counts = Counter(tokens)
            word_totals.update(token_counts)
            article_frequency.update(token_counts.keys())
            scraped_articles.append({"url": link["url"], "title": link["title"] or link["url"]})
        except Exception:
            continue

    article_count = len(scraped_articles)
    table_rows = []
    for word, total_occurrences in word_totals.most_common(10):
        containing_articles = article_frequency[word]
        coverage = (containing_articles / article_count * 100) if article_count else 0.0
        table_rows.append(
            {
                "word": word,
                "total_occurrences": total_occurrences,
                "article_count": containing_articles,
                "article_coverage_pct": round(coverage, 1),
            }
        )

    return {
        "links_found": len(link_entries),
        "articles_scraped": article_count,
        "scraped_preview": scraped_articles[:8],
        "top_words": table_rows,
    }
