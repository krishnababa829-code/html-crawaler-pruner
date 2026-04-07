"""Site crawler module.

Starting from a seed URL, discovers and follows:
- Internal hyperlinks (same domain)
- Pagination controls (next page, page numbers)
- Dynamically loaded sections (tabs, accordions)
- Sidebar navigation links

Respects max_depth and max_pages limits from config.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

from web2md.config import PipelineConfig
from web2md.fetcher import DynamicFetcher, FetchResult, StaticFetcher, create_fetcher
from web2md.pruner import PruneStats, prune_html

logger = logging.getLogger(__name__)

# URL patterns to skip
SKIP_EXTENSIONS = frozenset([
    ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z",
    ".exe", ".dmg", ".msi", ".deb", ".rpm",
    ".mp3", ".mp4", ".avi", ".mkv", ".mov", ".wmv",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".woff", ".woff2", ".ttf", ".eot",
    ".css", ".js", ".map",
])

SKIP_SCHEMES = frozenset(["mailto", "tel", "javascript", "data", "ftp"])


@dataclass
class CrawledPage:
    """A single crawled and pruned page."""

    url: str
    final_url: str
    pruned_html: str
    prune_stats: PruneStats
    fetch_result: FetchResult
    depth: int
    discovered_links: list[str] = field(default_factory=list)
    media_urls: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class CrawlResult:
    """Complete crawl result."""

    seed_url: str
    pages: list[CrawledPage]
    total_pages_discovered: int
    total_pages_crawled: int
    skipped_urls: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)


def _normalize_url(url: str) -> str:
    """Normalize a URL for deduplication."""
    parsed = urlparse(url)
    # Remove fragment, normalize trailing slash
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((
        parsed.scheme,
        parsed.netloc.lower(),
        path,
        parsed.params,
        parsed.query,
        "",  # drop fragment
    ))


def _is_same_domain(url: str, seed_domain: str) -> bool:
    """Check if URL belongs to the same domain as the seed."""
    parsed = urlparse(url)
    url_domain = parsed.netloc.lower().lstrip("www.")
    return url_domain == seed_domain


def _should_skip(url: str) -> bool:
    """Check if a URL should be skipped."""
    parsed = urlparse(url)
    if parsed.scheme in SKIP_SCHEMES:
        return True
    path_lower = parsed.path.lower()
    return any(path_lower.endswith(ext) for ext in SKIP_EXTENSIONS)


def _extract_links(html: str, base_url: str) -> list[str]:
    """Extract all href links from HTML, resolved against base URL."""
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if not href or href.startswith("#"):
            continue
        absolute = urljoin(base_url, href)
        links.append(absolute)
    return links


def _extract_pagination_links(html: str, base_url: str) -> list[str]:
    """Detect pagination controls and extract their URLs."""
    soup = BeautifulSoup(html, "lxml")
    pagination_links = []

    # Common pagination selectors
    pagination_selectors = [
        "[class*='pagination'] a[href]",
        "[class*='pager'] a[href]",
        "[class*='page-numbers'] a[href]",
        "[aria-label*='pagination'] a[href]",
        "[aria-label*='page'] a[href]",
        "a[rel='next']",
        "a[rel='prev']",
        "[class*='next-page'] a[href]",
        "[class*='prev-page'] a[href]",
    ]

    for selector in pagination_selectors:
        try:
            for el in soup.select(selector):
                href = el.get("href", "").strip()
                if href and not href.startswith("#"):
                    pagination_links.append(urljoin(base_url, href))
        except Exception:
            continue

    # Also detect "?page=N" or "/page/N" patterns in all links
    page_pattern = re.compile(r"[?&]page=\d+|/page/\d+", re.IGNORECASE)
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if page_pattern.search(href):
            pagination_links.append(urljoin(base_url, href))

    return list(set(pagination_links))


def _extract_media_urls(html: str, base_url: str) -> dict[str, list[str]]:
    """Extract all media URLs (images, video, audio) from HTML."""
    soup = BeautifulSoup(html, "lxml")
    media: dict[str, list[str]] = {
        "images": [],
        "videos": [],
        "audio": [],
    }

    # Images
    for img in soup.find_all("img", src=True):
        src = urljoin(base_url, img["src"].strip())
        if src not in media["images"]:
            media["images"].append(src)
    # srcset
    for el in soup.find_all(srcset=True):
        for entry in el["srcset"].split(","):
            parts = entry.strip().split()
            if parts:
                src = urljoin(base_url, parts[0])
                if src not in media["images"]:
                    media["images"].append(src)
    # picture > source
    for source in soup.find_all("source", src=True):
        parent = source.parent
        src = urljoin(base_url, source["src"].strip())
        if parent and parent.name == "picture":
            if src not in media["images"]:
                media["images"].append(src)
        elif parent and parent.name == "video":
            if src not in media["videos"]:
                media["videos"].append(src)
        elif parent and parent.name == "audio":
            if src not in media["audio"]:
                media["audio"].append(src)

    # Video tags
    for video in soup.find_all("video", src=True):
        src = urljoin(base_url, video["src"].strip())
        if src not in media["videos"]:
            media["videos"].append(src)
    # Video embeds (YouTube, Vimeo, etc.)
    for iframe in soup.find_all("iframe", src=True):
        src = iframe["src"].strip()
        if any(d in src for d in ["youtube", "vimeo", "dailymotion", "wistia"]):
            if src not in media["videos"]:
                media["videos"].append(src)

    # Audio tags
    for audio in soup.find_all("audio", src=True):
        src = urljoin(base_url, audio["src"].strip())
        if src not in media["audio"]:
            media["audio"].append(src)

    return media


def crawl(
    seed_url: str,
    config: PipelineConfig,
    static: bool = False,
) -> CrawlResult:
    """Crawl a website starting from seed_url.

    Uses BFS traversal with depth limiting. At each page:
    1. Fetch the full HTML
    2. Extract media URLs from raw HTML (before pruning removes img tags)
    3. Prune the HTML
    4. Discover links for further crawling

    Args:
        seed_url: The starting URL.
        config: Pipeline configuration.
        static: Use static (httpx) fetcher instead of Playwright.

    Returns:
        CrawlResult with all crawled pages.
    """
    seed_parsed = urlparse(seed_url)
    seed_domain = seed_parsed.netloc.lower().lstrip("www.")

    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque()  # (url, depth)
    queue.append((_normalize_url(seed_url), 0))

    pages: list[CrawledPage] = []
    skipped: list[str] = []
    errors: list[dict[str, str]] = []
    total_discovered = 0

    fetcher = create_fetcher(config, static=static)
    fetcher.start()

    try:
        while queue and len(pages) < config.max_pages:
            url, depth = queue.popleft()
            normalized = _normalize_url(url)

            if normalized in visited:
                continue
            visited.add(normalized)

            if _should_skip(url):
                skipped.append(url)
                continue

            logger.info("Crawling [depth=%d]: %s", depth, url)

            # Fetch
            result = fetcher.fetch(url)
            if result.error:
                errors.append({"url": url, "error": result.error})
                logger.warning("Fetch error for %s: %s", url, result.error)
                continue

            if not result.html:
                continue

            # Extract media from raw HTML before pruning
            media = _extract_media_urls(result.html, result.final_url)

            # Prune
            pruned, prune_stats = prune_html(result.html)

            # Discover links
            discovered = _extract_links(result.html, result.final_url)
            total_discovered += len(discovered)

            # Also get pagination links
            if config.follow_pagination:
                pagination = _extract_pagination_links(
                    result.html, result.final_url
                )
                discovered.extend(pagination)

            page = CrawledPage(
                url=url,
                final_url=result.final_url,
                pruned_html=pruned,
                prune_stats=prune_stats,
                fetch_result=result,
                depth=depth,
                discovered_links=discovered,
                media_urls=media,
            )
            pages.append(page)

            # Enqueue child links if within depth limit
            if depth < config.max_depth:
                for link in discovered:
                    norm_link = _normalize_url(link)
                    if norm_link in visited:
                        continue
                    if config.same_domain_only and not _is_same_domain(
                        link, seed_domain
                    ):
                        continue
                    queue.append((link, depth + 1))

    finally:
        fetcher.stop()

    return CrawlResult(
        seed_url=seed_url,
        pages=pages,
        total_pages_discovered=total_discovered,
        total_pages_crawled=len(pages),
        skipped_urls=skipped,
        errors=errors,
    )
