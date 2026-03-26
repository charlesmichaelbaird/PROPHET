"""Tool helpers for zero-cost homepage article analysis."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import re
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, urlunparse
import xml.etree.ElementTree as ET

import requests

from mcp_server.rag import answer_question
from mcp_server.storage import ensure_data_directories, persist_article_if_new, write_run_index

_BASE_STOPWORDS = {
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
    "yourselves", "will", "also", "says", "say", "get", "got", "like", "one", "two", "new",
}

_MONTH_STOPWORDS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

_DAY_STOPWORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
    "today", "tomorrow", "yesterday",
}

_DATE_AND_FORMAT_STOPWORDS = {
    "photo", "photos", "photograph", "file", "files", "ap", "news", "said",
    "copyright", "published", "updated", "newsletter", "breaking", "read", "story",
}

_STOPWORDS = (
    _BASE_STOPWORDS
    | _MONTH_STOPWORDS
    | _DAY_STOPWORDS
    | _DATE_AND_FORMAT_STOPWORDS
)

REUTERS_DISCOVERY_PATHS = (
    "/",
    "/world/",
    "/business/",
    "/markets/",
)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

SOURCE_DATE_CONFIG = {
    "ap": {
        "label": "AP News",
        "canonical_home": "https://apnews.com/",
        "sitemap_indexes": (
            "https://apnews.com/sitemap.xml",
            "https://apnews.com/news-sitemap.xml",
        ),
    },
    "bbc": {
        "label": "BBC",
        "canonical_home": "https://www.bbc.com/",
        "sitemap_indexes": (
            "https://www.bbc.com/sitemaps/https-index-com-news.xml",
            "https://www.bbc.com/sitemaps/https-index-com-archive.xml",
            "https://www.bbc.com/sitemaps/https-news.xml",
        ),
    },
}

BBC_NON_ENGLISH_PATH_MARKERS = (
    "/mundo/",
    "/zhongwen/",
    "/arabic/",
    "/russian/",
    "/hindi/",
    "/afrique/",
    "/korean/",
    "/japanese/",
    "/portuguese/",
    "/ukrainian/",
    "/urdu/",
    "/uzbek/",
    "/persian/",
    "/serbian/",
    "/telugu/",
    "/tamil/",
    "/pidgin/",
    "/indonesia/",
    "/bengali/",
    "/hausa/",
    "/swahili/",
    "/gahuza/",
    "/nepali/",
    "/kyrgyz/",
    "/azeri/",
    "/marathi/",
    "/gujarati/",
    "/pashto/",
    "/somali/",
    "/burmese/",
    "/thai/",
    "/uz/",
)


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


class DocumentTitleParser(HTMLParser):
    """Extract a best-effort page title from HTML metadata or <title>."""

    def __init__(self) -> None:
        super().__init__()
        self._in_title = False
        self._title_parts: list[str] = []
        self.meta_title = ""

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        lowered = tag.lower()
        if lowered == "title":
            self._in_title = True
            return

        if lowered != "meta":
            return

        attr_map = {str(key).lower(): str(value) for key, value in attrs if value}
        marker = attr_map.get("property") or attr_map.get("name")
        content = attr_map.get("content", "").strip()
        if marker and marker.lower() in {"og:title", "twitter:title"} and content and not self.meta_title:
            self.meta_title = content

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._in_title:
            cleaned = data.strip()
            if cleaned:
                self._title_parts.append(cleaned)

    def get_title(self) -> str:
        title_tag = " ".join(self._title_parts).strip()
        return self.meta_title or title_tag


def _is_browser_header_host(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return "reuters.com" in host or "bbc.com" in host


def _is_reuters_host(url: str) -> bool:
    return "reuters.com" in urlparse(url).netloc.lower()


def _new_browser_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(_BROWSER_HEADERS)
    return session


def fetch_url(
    url: str,
    timeout: int = 10,
    session: requests.Session | None = None,
    headers: dict[str, str] | None = None,
) -> str:
    """Fetch raw HTML for a URL with basic validation and error messages."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must include http:// or https:// and a valid host")

    request_headers = {"User-Agent": "PROPHET-ZeroCost/1.0 (+https://example.local)"}
    if _is_browser_header_host(url):
        request_headers.update(_BROWSER_HEADERS)
    if headers:
        request_headers.update(headers)

    try:
        client = session or requests
        response = client.get(
            url,
            timeout=timeout,
            headers=request_headers,
        )
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code in {401, 403, 429}:
            raise RuntimeError(
                f"Request blocked by target site (HTTP {status_code}). "
                "Try alternate discovery pages or retry later."
            ) from exc
        raise RuntimeError(f"Failed to fetch URL (HTTP {status_code}): {exc}") from exc
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


def _discover_reuters_links(max_links: int) -> tuple[list[dict[str, str]], str, list[str]]:
    """Try Reuters homepage/discovery pages and return first link set found."""
    base_url = "https://www.reuters.com/"
    session = _new_browser_session()
    diagnostics: list[str] = []

    for discovery_path in REUTERS_DISCOVERY_PATHS:
        candidate_url = urljoin(base_url, discovery_path)
        try:
            homepage_html = fetch_url(candidate_url, timeout=12, session=session)
            links = extract_article_links(homepage_html, candidate_url, max_links=max_links)
        except Exception as exc:
            diagnostics.append(f"blocked_or_fetch_failed:{candidate_url}:{exc}")
            continue

        if links:
            diagnostics.append(f"discovery_success:{candidate_url}:links={len(links)}")
            return links, candidate_url, diagnostics
        diagnostics.append(f"no_links_found:{candidate_url}")

    return [], base_url, diagnostics


def _parse_mmddyyyy(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%m/%d/%Y")


def _tag_local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _parse_sitemap_document(xml_text: str) -> dict[str, list[dict[str, str]] | list[str]]:
    root = ET.fromstring(xml_text)
    root_name = _tag_local_name(root.tag).lower()
    if root_name == "sitemapindex":
        sitemap_locs: list[str] = []
        for node in root.iter():
            if _tag_local_name(node.tag).lower() == "loc" and node.text:
                sitemap_locs.append(node.text.strip())
        return {"type": ["sitemapindex"], "sitemap_locs": sitemap_locs}

    if root_name == "urlset":
        entries: list[dict[str, str]] = []
        for url_node in root:
            if _tag_local_name(url_node.tag).lower() != "url":
                continue
            loc = ""
            lastmod = ""
            publication_date = ""
            title = ""
            for child in url_node.iter():
                local = _tag_local_name(child.tag).lower()
                text = (child.text or "").strip()
                if local == "loc" and text and not loc:
                    loc = text
                elif local == "lastmod" and text and not lastmod:
                    lastmod = text
                elif local == "publication_date" and text and not publication_date:
                    publication_date = text
                elif local == "title" and text and not title:
                    title = text
            if loc:
                entries.append({"url": loc, "title": title, "lastmod": lastmod, "publication_date": publication_date})
        return {"type": ["urlset"], "entries": entries}

    return {"type": [root_name]}


def _entry_matches_date(entry: dict[str, str], target_date: datetime) -> bool:
    expected = target_date.date().isoformat()
    for key in ("publication_date", "lastmod"):
        value = entry.get(key, "")
        if value.startswith(expected):
            return True

    url = entry.get("url", "")
    if not url:
        return False
    date_path = target_date.strftime("%Y/%m/%d")
    return f"/{date_path}/" in url


def _is_english_candidate_url(source_name: str, candidate_url: str) -> bool:
    if source_name != "bbc":
        return True

    parsed = urlparse(candidate_url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if "bbc.com" not in host:
        return False
    return not any(marker in path for marker in BBC_NON_ENGLISH_PATH_MARKERS)


def _is_probably_english_text(text: str) -> bool:
    sample = (text or "").strip()
    if not sample:
        return False
    alpha_chars = [char for char in sample if char.isalpha()]
    if not alpha_chars:
        return False
    ascii_alpha = [char for char in alpha_chars if "a" <= char.lower() <= "z"]
    return (len(ascii_alpha) / len(alpha_chars)) >= 0.85


def _discover_articles_by_date(
    source_name: str,
    query_date: datetime,
    max_links: int,
) -> tuple[list[dict[str, str]], list[str], str]:
    config = SOURCE_DATE_CONFIG[source_name]
    diagnostics: list[str] = []
    session = _new_browser_session() if source_name == "bbc" else None

    discovered: list[dict[str, str]] = []
    seen: set[str] = set()
    selected_index = config["sitemap_indexes"][0]
    month_token = query_date.strftime("%Y-%m")

    for sitemap_index_url in config["sitemap_indexes"]:
        selected_index = sitemap_index_url
        try:
            index_xml = fetch_url(sitemap_index_url, timeout=14, session=session)
            parsed_index = _parse_sitemap_document(index_xml)
        except Exception as exc:
            diagnostics.append(f"index_fetch_failed:{sitemap_index_url}:{exc}")
            continue

        if parsed_index.get("type", [""])[0] == "urlset":
            candidate_sitemaps = [sitemap_index_url]
        else:
            locs = parsed_index.get("sitemap_locs", [])
            filtered_locs = [loc for loc in locs if month_token in loc] or list(locs)
            candidate_sitemaps = filtered_locs[:20]

        if not candidate_sitemaps:
            diagnostics.append(f"no_candidate_sitemaps:{sitemap_index_url}")
            continue

        for sitemap_url in candidate_sitemaps:
            try:
                sitemap_xml = index_xml if sitemap_url == sitemap_index_url else fetch_url(sitemap_url, timeout=14, session=session)
                parsed_sitemap = _parse_sitemap_document(sitemap_xml)
                entries = parsed_sitemap.get("entries", [])
            except Exception as exc:
                diagnostics.append(f"sitemap_parse_failed:{sitemap_url}:{exc}")
                continue

            matched = 0
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                url = entry.get("url", "")
                if not url or url in seen:
                    continue
                if not _is_english_candidate_url(source_name, url):
                    diagnostics.append(f"filtered_non_english_url:{url}")
                    continue
                if not _entry_matches_date(entry, query_date):
                    continue
                seen.add(url)
                discovered.append({"url": url, "title": entry.get("title", "") or url})
                matched += 1
                if len(discovered) >= max_links:
                    diagnostics.append(f"matched_limit_reached:{sitemap_url}")
                    return discovered, diagnostics, selected_index
            if matched == 0:
                diagnostics.append(f"no_date_matches:{sitemap_url}")
            else:
                diagnostics.append(f"date_matches:{sitemap_url}:{matched}")

    return discovered, diagnostics, selected_index


def extract_article_text(article_html: str) -> str:
    """Extract article body text with a simple paragraph strategy."""
    parser = ArticleTextParser()
    parser.feed(article_html)
    parser.close()

    clean = re.sub(r"\s+", " ", unescape(parser.get_text())).strip()
    return _strip_ap_photo_captions(clean)


def _strip_ap_photo_captions(text: str) -> str:
    """
    Remove AP photo caption lead-ins from scraped article text.

    Example caption marker pattern:
    `(AP Photo/John Doe)` or `(AP Photo/John Doe, File)`.

    If present, keep only the text that appears after the closing parenthesis.
    """
    cleaned = text
    marker_pattern = re.compile(r"\(AP Photo/[^)]*\)", re.IGNORECASE)

    while True:
        match = marker_pattern.search(cleaned)
        if not match:
            break
        cleaned = cleaned[match.end():].lstrip(" -:;,\n\t")

    return re.sub(r"\s+", " ", cleaned).strip()


def extract_document_title(article_html: str) -> str:
    """Extract a human-readable title from a page."""
    parser = DocumentTitleParser()
    parser.feed(article_html)
    parser.close()
    return re.sub(r"\s+", " ", unescape(parser.get_title())).strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]{3,}", text.lower())


def _filter_tokens(text: str) -> list[str]:
    """Tokenize and remove low-value words before frequency calculations."""
    return [token for token in _tokenize(text) if token not in _STOPWORDS]


def _split_sentences(text: str) -> list[str]:
    """Split text into rough sentences using punctuation boundaries."""
    chunks = re.split(r"(?<=[.!?])\s+", text)
    cleaned = [chunk.strip() for chunk in chunks if chunk and len(chunk.strip()) >= 40]
    return cleaned


def _extract_top_phrases(article_texts: list[str], max_phrases: int = 3) -> list[str]:
    """Extract recurring two-word phrases from filtered article tokens."""
    phrase_counts: Counter[str] = Counter()
    for article_text in article_texts:
        tokens = _filter_tokens(article_text)
        for first, second in zip(tokens, tokens[1:]):
            phrase = f"{first} {second}"
            phrase_counts[phrase] += 1

    recurring = [phrase for phrase, count in phrase_counts.most_common(30) if count >= 2]
    return recurring[:max_phrases]


def _select_representative_line(article_texts: list[str], focus_words: list[str]) -> str:
    """Pick a representative sentence using overlap with top focus words."""
    candidate_sentences: list[str] = []
    for article_text in article_texts:
        candidate_sentences.extend(_split_sentences(article_text))

    if not candidate_sentences:
        return ""

    scored: list[tuple[int, str]] = []
    focus_set = set(focus_words)
    for sentence in candidate_sentences:
        sentence_tokens = set(_filter_tokens(sentence))
        score = len(sentence_tokens.intersection(focus_set))
        if 10 <= len(sentence.split()) <= 34:
            scored.append((score, sentence))

    if not scored:
        return ""

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _select_supporting_article_url(
    representative_line: str,
    article_sources: list[dict[str, str]],
) -> str:
    """
    Choose the most useful source article via token overlap with the representative line.

    Deterministic tie-break: retain first seen article among equal scores.
    """
    if not representative_line:
        return ""

    line_tokens = set(_filter_tokens(representative_line))
    if not line_tokens:
        return ""

    best_url = ""
    best_score = 0
    for source in article_sources:
        url = source.get("url", "")
        text = source.get("text", "")
        if not url or not text:
            continue

        article_tokens = set(_filter_tokens(text))
        if not article_tokens:
            continue

        overlap_score = len(line_tokens.intersection(article_tokens))
        if overlap_score > best_score:
            best_score = overlap_score
            best_url = url

    return best_url


def build_summary(
    article_texts: list[str],
    top_words: list[dict[str, float]],
    articles_scraped: int,
    links_found: int,
) -> tuple[str, str]:
    """
    Build a short, heuristic narrative summary of the scraped corpus.

    The summary combines: (1) broad coverage stats, (2) recurring high-value
    terms, (3) repeated short phrases, and (4) one representative sentence
    selected with a term-overlap score.
    """
    if not article_texts or not top_words:
        return (
            "Summary unavailable because there was not enough clean article text "
            "to extract recurring terms.",
            "",
        )

    focus_words = [row["word"] for row in top_words[:5]]
    focus_terms = ", ".join(focus_words[:-1]) + f", and {focus_words[-1]}" if len(focus_words) > 1 else focus_words[0]

    phrase_list = _extract_top_phrases(article_texts, max_phrases=3)
    phrase_text = ""
    if phrase_list:
        joined_phrases = ", ".join(f'"{phrase}"' for phrase in phrase_list)
        phrase_text = f" Repeated phrase signals include {joined_phrases}."

    best_sentence = _select_representative_line(article_texts, focus_words)

    coverage_rate = (articles_scraped / links_found * 100) if links_found else 0.0
    summary = (
        f"From {articles_scraped} scraped articles out of {links_found} discovered links "
        f"({coverage_rate:.0f}% coverage), the dominant themes center on {focus_terms}."
        f"{phrase_text}"
    )
    if best_sentence:
        summary += f" A representative line from the corpus is: {best_sentence}"

    return summary, best_sentence


def _match_keyword(page_title: str, article_text: str, keyword: str) -> bool:
    """Simple case-insensitive keyword matching against title/body content."""
    normalized_keyword = keyword.strip().lower()
    if not normalized_keyword:
        return False

    haystack = f"{page_title} {article_text}".lower()
    return normalized_keyword in haystack


def _suggest_alternative_keywords(
    candidate_article_texts: list[str],
    keyword: str,
    max_suggestions: int = 5,
) -> list[str]:
    """Derive simple alternative keyword suggestions from broad candidate text."""
    suggestion_counts: Counter[str] = Counter()
    keyword_lower = keyword.strip().lower()

    for text in candidate_article_texts:
        suggestion_counts.update(_filter_tokens(text))

    suggestions: list[str] = []
    for word, _count in suggestion_counts.most_common(50):
        if word == keyword_lower:
            continue
        suggestions.append(word)
        if len(suggestions) >= max_suggestions:
            break
    return suggestions


def analyze_homepage(
    homepage_url: str,
    max_articles: int = 20,
    keyword: str = "",
    keyword_filter_enabled: bool = False,
) -> dict:
    """Scrape homepage article links and compute top-word metrics without LLMs."""
    scrape_started_at = datetime.now(timezone.utc)
    ensure_data_directories()
    diagnostics: list[str] = []
    reuters_mode = _is_reuters_host(homepage_url)
    reuters_session: requests.Session | None = _new_browser_session() if reuters_mode else None
    discovery_url = homepage_url

    if reuters_mode:
        link_entries, discovery_url, diagnostics = _discover_reuters_links(max_links=max_articles * 3)
        if not link_entries:
            diag_summary = " | ".join(diagnostics[:4]) if diagnostics else "no diagnostics"
            raise RuntimeError(
                "Reuters blocked discovery requests or returned no candidate article links. "
                f"Try again later or use an alternate Reuters discovery page. Diagnostics: {diag_summary}"
            )
    else:
        homepage_html = fetch_url(homepage_url)
        link_entries = extract_article_links(homepage_html, homepage_url, max_links=max_articles * 3)

    keyword_clean = keyword.strip()
    keyword_mode_active = bool(keyword_filter_enabled)

    word_totals: Counter[str] = Counter()
    article_frequency: Counter[str] = Counter()
    scraped_articles: list[dict[str, str]] = []
    article_texts: list[str] = []
    candidate_article_texts: list[str] = []
    article_sources: list[dict[str, str]] = []
    failed_urls: list[str] = []
    persisted_articles: list[dict[str, str]] = []
    duplicate_articles: list[dict[str, str]] = []
    manifest_path = ""
    new_articles_saved = 0
    duplicate_articles_skipped = 0
    attempted_articles = 0
    candidate_articles_considered = 0

    for link in link_entries:
        if attempted_articles >= max_articles:
            break

        attempted_articles += 1
        try:
            article_html = fetch_url(link["url"], timeout=12, session=reuters_session)
            page_title = extract_document_title(article_html)
            article_text = extract_article_text(article_html)
            if not article_text:
                continue

            tokens = _filter_tokens(article_text)
            if not tokens:
                continue

            candidate_articles_considered += 1
            candidate_article_texts.append(f"{page_title} {article_text}")

            if keyword_mode_active and not _match_keyword(page_title, article_text, keyword_clean):
                continue

            token_counts = Counter(tokens)
            word_totals.update(token_counts)
            article_frequency.update(token_counts.keys())
            article_texts.append(article_text)
            article_sources.append(
                {
                    "url": link["url"],
                    "text": article_text,
                }
            )
            preview_entry = (
                {
                    "url": link["url"],
                    "title": page_title or link["title"] or link["url"],
                    "word_count": len(tokens),
                }
            )
            scraped_articles.append(preview_entry)
            persist_result = persist_article_if_new(
                homepage_url=homepage_url,
                source_homepage_url=homepage_url,
                article_url=link["url"],
                title=preview_entry["title"],
                scrape_timestamp=datetime.now(timezone.utc),
                clean_text=article_text,
                article_html=article_html,
                article_metadata=preview_entry,
            )
            manifest_path = persist_result.get("manifest_path", manifest_path)
            if persist_result.get("status") == "saved":
                new_articles_saved += 1
                written_paths = persist_result.get("written_paths", {})
                persisted_articles.append(
                    {
                        "url": preview_entry["url"],
                        "title": preview_entry["title"],
                        "content_hash": persist_result.get("content_hash", ""),
                        **written_paths,
                    }
                )
            else:
                duplicate_articles_skipped += 1
                duplicate_articles.append(
                    {
                        "url": preview_entry["url"],
                        "title": preview_entry["title"],
                        "content_hash": persist_result.get("content_hash", ""),
                        "status": "duplicate",
                    }
                )
        except Exception as exc:
            if reuters_mode:
                diagnostics.append(f"article_fetch_failed:{link['url']}:{exc}")
            failed_urls.append(link["url"])
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

    if keyword_mode_active:
        representative_line = ""
        if not keyword_clean:
            summary = (
                "Keyword filtering is enabled, but no keyword was provided. "
                "Enter a keyword to run a filtered analysis."
            )
        elif article_count == 0:
            summary = (
                f"No matching articles were found for keyword '{keyword_clean}' "
                f"across {candidate_articles_considered} candidate article pages. "
                "Try a different keyword."
            )
        elif article_count < 2:
            summary = (
                f"Only {article_count} matching article was found for keyword '{keyword_clean}' "
                f"from {candidate_articles_considered} candidate article pages. "
                "That is too little signal for a useful keyword-focused summary; try a broader keyword."
            )
        else:
            base_summary, representative_line = build_summary(
                article_texts=article_texts,
                top_words=table_rows,
                articles_scraped=article_count,
                links_found=max(candidate_articles_considered, 1),
            )
            summary = (
                f"Keyword focus: '{keyword_clean}'. "
                f"Matched {article_count} of {candidate_articles_considered} candidate articles. "
                f"{base_summary}"
            )
    else:
        summary, representative_line = build_summary(
            article_texts=article_texts,
            top_words=table_rows,
            articles_scraped=article_count,
            links_found=len(link_entries),
        )

    representative_source_url = _select_supporting_article_url(
        representative_line=representative_line,
        article_sources=article_sources,
    )

    keyword_suggestions: list[str] = []
    if keyword_mode_active and keyword_clean and article_count < 2:
        keyword_suggestions = _suggest_alternative_keywords(
            candidate_article_texts=candidate_article_texts,
            keyword=keyword_clean,
            max_suggestions=5,
        )

    result = {
        "links_found": len(link_entries),
        "articles_attempted": attempted_articles,
        "articles_scraped": article_count,
        "articles_failed": len(failed_urls),
        "articles_new_saved": new_articles_saved,
        "articles_duplicates_skipped": duplicate_articles_skipped,
        "scraped_preview": scraped_articles[:8],
        "top_words": table_rows,
        "summary": summary,
        "representative_line": representative_line,
        "representative_source_url": representative_source_url,
        "keyword_filter_enabled": keyword_mode_active,
        "keyword": keyword_clean,
        "candidate_articles_considered": candidate_articles_considered,
        "matching_articles": article_count if keyword_mode_active else 0,
        "keyword_suggestions": keyword_suggestions,
        "article_corpus": [
            {
                "url": article["url"],
                "title": article["title"],
                "text": source["text"],
            }
            for article, source in zip(scraped_articles, article_sources)
        ],
        "persisted_articles": persisted_articles[:8],
        "duplicate_articles": duplicate_articles[:8],
        "article_manifest_path": manifest_path,
        "fetch_diagnostics": diagnostics[:30],
    }

    run_index_path = write_run_index(
        homepage_url=discovery_url,
        scrape_timestamp=scrape_started_at,
        summary_payload={
            "homepage_url": homepage_url,
            "discovery_url": discovery_url,
            "scrape_started_at": scrape_started_at.isoformat(),
            "links_found": result["links_found"],
            "articles_attempted": result["articles_attempted"],
            "articles_scraped": result["articles_scraped"],
            "articles_failed": result["articles_failed"],
            "articles_new_saved": result["articles_new_saved"],
            "articles_duplicates_skipped": result["articles_duplicates_skipped"],
            "keyword_filter_enabled": result["keyword_filter_enabled"],
            "keyword": result["keyword"],
            "summary": result["summary"],
            "top_words": result["top_words"],
            "persisted_articles": persisted_articles,
            "duplicate_articles": duplicate_articles,
            "article_manifest_path": manifest_path,
            "fetch_diagnostics": diagnostics[:100],
        },
    )
    result["run_index_path"] = run_index_path
    return result


def query_site_article_count(
    homepage_url: str,
    max_links: int = 200,
) -> dict:
    """Run lightweight article-link discovery/count without scraping article bodies."""
    diagnostics: list[str] = []
    discovery_url = homepage_url
    if _is_reuters_host(homepage_url):
        link_entries, discovery_url, diagnostics = _discover_reuters_links(max_links=max_links)
        if not link_entries:
            diag_summary = " | ".join(diagnostics[:4]) if diagnostics else "no diagnostics"
            raise RuntimeError(
                "Reuters blocked lightweight discovery requests (or returned no article links). "
                f"An alternate discovery path may be needed. Diagnostics: {diag_summary}"
            )
    else:
        homepage_html = fetch_url(homepage_url)
        link_entries = extract_article_links(homepage_html, homepage_url, max_links=max_links)

    preview = []
    for entry in link_entries[:10]:
        preview.append(
            {
                "url": entry.get("url", ""),
                "title": entry.get("title", "") or entry.get("url", ""),
            }
        )

    return {
        "homepage_url": homepage_url,
        "discovery_url": discovery_url,
        "links_found": len(link_entries),
        "preview": preview,
        "diagnostics": diagnostics[:20],
    }


def query_source_article_count_by_date(
    source_name: str,
    date_str: str,
    max_links: int = 250,
) -> dict:
    """Lightweight metadata-only date-based article discovery for supported sources."""
    source_key = source_name.strip().lower()
    if source_key not in SOURCE_DATE_CONFIG:
        raise ValueError(f"Unsupported source '{source_name}'. Expected one of: {', '.join(SOURCE_DATE_CONFIG)}")

    query_date = _parse_mmddyyyy(date_str)
    links, diagnostics, discovery_index = _discover_articles_by_date(
        source_name=source_key,
        query_date=query_date,
        max_links=max_links,
    )
    if not links:
        label = SOURCE_DATE_CONFIG[source_key]["label"]
        raise RuntimeError(
            f"{label} date-based archive discovery returned no article links for {query_date:%m/%d/%Y}. "
            "The archive endpoint may be temporarily unavailable or blocked."
        )

    return {
        "source_name": source_key,
        "selected_date": query_date.strftime("%m/%d/%Y"),
        "links_found": len(links),
        "preview": links[:10],
        "diagnostics": diagnostics[:20],
        "discovery_index": discovery_index,
    }


def scrape_source_articles_by_date(
    source_name: str,
    date_str: str,
    max_articles: int = 200,
) -> dict:
    """Full date-based scrape workflow for supported sources."""
    source_key = source_name.strip().lower()
    if source_key not in SOURCE_DATE_CONFIG:
        raise ValueError(f"Unsupported source '{source_name}'. Expected one of: {', '.join(SOURCE_DATE_CONFIG)}")

    query_date = _parse_mmddyyyy(date_str)
    scrape_started_at = datetime.now(timezone.utc)
    ensure_data_directories()
    links, diagnostics, discovery_index = _discover_articles_by_date(
        source_name=source_key,
        query_date=query_date,
        max_links=max_articles * 3,
    )
    if not links:
        label = SOURCE_DATE_CONFIG[source_key]["label"]
        raise RuntimeError(
            f"{label} date-based archive discovery returned no article links for {query_date:%m/%d/%Y}. "
            "Try a different date or retry later."
        )

    session = _new_browser_session() if source_key == "bbc" else None
    source_home = SOURCE_DATE_CONFIG[source_key]["canonical_home"]
    attempted_articles = 0
    new_articles_saved = 0
    duplicate_articles_skipped = 0
    failed_urls: list[str] = []
    scraped_articles: list[dict[str, str]] = []
    article_texts: list[str] = []
    article_sources: list[dict[str, str]] = []
    persisted_articles: list[dict[str, str]] = []
    duplicate_articles: list[dict[str, str]] = []
    manifest_path = ""
    word_totals: Counter[str] = Counter()
    article_frequency: Counter[str] = Counter()

    for link in links:
        if attempted_articles >= max_articles:
            break
        attempted_articles += 1
        try:
            article_html = fetch_url(link["url"], timeout=14, session=session)
            page_title = extract_document_title(article_html) or link.get("title", "") or link["url"]
            article_text = extract_article_text(article_html)
            if not article_text:
                diagnostics.append(f"article_parse_empty:{link['url']}")
                continue
            if source_key == "bbc" and not _is_probably_english_text(f"{page_title} {article_text[:1000]}"):
                diagnostics.append(f"filtered_non_english_content:{link['url']}")
                continue

            tokens = _filter_tokens(article_text)
            if not tokens:
                diagnostics.append(f"article_tokens_empty:{link['url']}")
                continue

            token_counts = Counter(tokens)
            word_totals.update(token_counts)
            article_frequency.update(token_counts.keys())
            article_texts.append(article_text)
            article_sources.append({"url": link["url"], "text": article_text})
            preview_entry = {"url": link["url"], "title": page_title, "word_count": len(tokens)}
            scraped_articles.append(preview_entry)

            persist_result = persist_article_if_new(
                homepage_url=source_home,
                source_homepage_url=source_home,
                article_url=link["url"],
                title=preview_entry["title"],
                scrape_timestamp=datetime.now(timezone.utc),
                clean_text=article_text,
                article_html=article_html,
                article_metadata=preview_entry,
            )
            manifest_path = persist_result.get("manifest_path", manifest_path)
            if persist_result.get("status") == "saved":
                new_articles_saved += 1
                persisted_articles.append(
                    {
                        "url": preview_entry["url"],
                        "title": preview_entry["title"],
                        "content_hash": persist_result.get("content_hash", ""),
                        **persist_result.get("written_paths", {}),
                    }
                )
            else:
                duplicate_articles_skipped += 1
                duplicate_articles.append(
                    {
                        "url": preview_entry["url"],
                        "title": preview_entry["title"],
                        "content_hash": persist_result.get("content_hash", ""),
                        "status": "duplicate",
                    }
                )
        except Exception as exc:
            diagnostics.append(f"article_fetch_failed:{link['url']}:{exc}")
            failed_urls.append(link["url"])

    table_rows = []
    article_count = len(scraped_articles)
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

    summary, representative_line = build_summary(
        article_texts=article_texts,
        top_words=table_rows,
        articles_scraped=article_count,
        links_found=len(links),
    )
    representative_source_url = _select_supporting_article_url(
        representative_line=representative_line,
        article_sources=article_sources,
    )

    result = {
        "source_name": source_key,
        "selected_date": query_date.strftime("%m/%d/%Y"),
        "links_found": len(links),
        "articles_attempted": attempted_articles,
        "articles_scraped": article_count,
        "articles_failed": len(failed_urls),
        "articles_new_saved": new_articles_saved,
        "articles_duplicates_skipped": duplicate_articles_skipped,
        "scraped_preview": scraped_articles[:8],
        "top_words": table_rows,
        "summary": summary,
        "representative_line": representative_line,
        "representative_source_url": representative_source_url,
        "keyword_filter_enabled": False,
        "keyword": "",
        "candidate_articles_considered": attempted_articles,
        "matching_articles": 0,
        "keyword_suggestions": [],
        "article_corpus": [
            {"url": article["url"], "title": article["title"], "text": source["text"]}
            for article, source in zip(scraped_articles, article_sources)
        ],
        "persisted_articles": persisted_articles[:8],
        "duplicate_articles": duplicate_articles[:8],
        "article_manifest_path": manifest_path,
        "fetch_diagnostics": diagnostics[:40],
        "discovery_index": discovery_index,
    }

    run_index_path = write_run_index(
        homepage_url=source_home,
        scrape_timestamp=scrape_started_at,
        summary_payload={
            "source_name": source_key,
            "selected_date": query_date.strftime("%m/%d/%Y"),
            "discovery_index": discovery_index,
            "links_found": result["links_found"],
            "articles_attempted": result["articles_attempted"],
            "articles_scraped": result["articles_scraped"],
            "articles_failed": result["articles_failed"],
            "articles_new_saved": result["articles_new_saved"],
            "articles_duplicates_skipped": result["articles_duplicates_skipped"],
            "summary": result["summary"],
            "top_words": result["top_words"],
            "article_manifest_path": manifest_path,
            "fetch_diagnostics": diagnostics[:100],
        },
    )
    result["run_index_path"] = run_index_path
    return result


def ask_the_prophet(
    question: str,
    article_corpus: list[dict[str, str]],
    embedding_model: str = "",
    answer_model: str = "",
) -> dict:
    """Answer a question using local persistent vector retrieval + Ollama generation."""
    _ = article_corpus  # Legacy arg retained for API compatibility with UI/server wiring.
    return answer_question(
        question=question,
        top_k=5,
        embedding_model=embedding_model,
        answer_model=answer_model,
    )
