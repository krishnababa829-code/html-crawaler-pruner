"""HTML fetcher module.

Two strategies:
1. Dynamic (Playwright): Full browser rendering for JS-heavy SPAs.
   Scrolls to trigger lazy-loaded content, expands dropdowns, waits
   for dynamic sections.
2. Static (httpx): Lightweight HTTP GET for server-rendered pages.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from web2md.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result of fetching a URL."""

    url: str
    final_url: str
    html: str
    status_code: int
    content_type: str
    fetch_time_ms: float
    method: str  # "dynamic" or "static"
    error: Optional[str] = None


class DynamicFetcher:
    """Playwright-based fetcher for JavaScript-rendered pages.

    Handles:
    - Single Page Applications (React, Vue, Angular)
    - Lazy-loaded content (infinite scroll)
    - Click-to-expand sections
    - Dynamic tab/dropdown content
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._playwright = None
        self._browser = None
        self._context = None

    def start(self) -> None:
        """Launch the browser."""
        if self._browser is not None:
            return

        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.config.headless,
        )
        self._context = self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=self.config.user_agent,
            java_script_enabled=True,
        )
        # Block heavy resources that don't affect content
        self._context.route(
            "**/*.{png,jpg,jpeg,gif,webp,svg,ico,woff,woff2,ttf,eot}",
            lambda route: route.abort(),
        )
        logger.info("Playwright browser started")

    def stop(self) -> None:
        """Shut down the browser."""
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass
        self._browser = None
        self._context = None
        self._playwright = None

    def fetch(self, url: str) -> FetchResult:
        """Fetch a URL with full JS rendering."""
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        if self._context is None:
            self.start()

        page = self._context.new_page()
        start_time = time.monotonic()
        error = None
        html = ""
        final_url = url
        status = 0
        content_type = ""

        try:
            logger.info("Fetching (dynamic): %s", url)
            response = page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=self.config.timeout * 1000,
            )

            # Wait for network to settle
            try:
                page.wait_for_load_state("networkidle", timeout=10_000)
            except PlaywrightTimeout:
                pass  # Some pages never fully settle

            # Scroll to trigger lazy-loaded content
            self._scroll_to_bottom(page)

            # Expand dynamic content sections
            self._expand_dynamic_content(page)

            html = page.content()
            final_url = page.url
            status = response.status if response else 0
            content_type = (
                response.headers.get("content-type", "") if response else ""
            )
            logger.info("Fetched %d chars from %s (status %d)", len(html), final_url, status)

        except PlaywrightTimeout:
            error = f"Timeout after {self.config.timeout}s"
            try:
                html = page.content()
            except Exception:
                html = ""
            logger.warning("Timeout fetching %s", url)
        except Exception as exc:
            error = str(exc)
            logger.error("Error fetching %s: %s", url, exc)
        finally:
            try:
                page.close()
            except Exception:
                pass

        elapsed = (time.monotonic() - start_time) * 1000

        return FetchResult(
            url=url,
            final_url=final_url,
            html=html,
            status_code=status,
            content_type=content_type,
            fetch_time_ms=round(elapsed, 2),
            method="dynamic",
            error=error,
        )

    def _scroll_to_bottom(self, page, max_scrolls: int = 15) -> None:
        """Incrementally scroll to trigger lazy-loaded content."""
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        previous_height = 0
        for _ in range(max_scrolls):
            try:
                current_height = page.evaluate("document.body.scrollHeight")
                if current_height == previous_height:
                    break
                previous_height = current_height
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(800)
            except (PlaywrightTimeout, Exception):
                break

    def _expand_dynamic_content(self, page) -> None:
        """Click common expand/show-more buttons and accordion triggers."""
        expand_selectors = [
            "button[aria-expanded='false']",
            "[class*='show-more']",
            "[class*='load-more']",
            "[class*='expand']",
            "[class*='read-more']",
            "[class*='see-more']",
            "details:not([open]) > summary",
            "[role='tab'][aria-selected='false']",
        ]
        for selector in expand_selectors:
            try:
                elements = page.query_selector_all(selector)
                for el in elements[:10]:  # Limit to avoid infinite loops
                    try:
                        el.click(timeout=1000)
                        page.wait_for_timeout(300)
                    except Exception:
                        continue
            except Exception:
                continue

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class StaticFetcher:
    """httpx-based fetcher for server-rendered pages.

    Faster and lighter than Playwright. Use when pages don't
    require JavaScript execution.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._client: Optional[httpx.Client] = None

    def start(self) -> None:
        """Create the HTTP client."""
        if self._client is not None:
            return
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.config.timeout, connect=10.0),
            follow_redirects=True,
            headers={
                "User-Agent": self.config.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
            },
            limits=httpx.Limits(max_connections=self.config.concurrency),
        )

    def stop(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
        self._client = None

    def fetch(self, url: str) -> FetchResult:
        """Fetch a URL via HTTP GET."""
        if self._client is None:
            self.start()

        start_time = time.monotonic()
        error = None

        try:
            logger.info("Fetching (static): %s", url)
            resp = self._client.get(url)
            html = resp.text
            final_url = str(resp.url)
            status = resp.status_code
            content_type = resp.headers.get("content-type", "")
            logger.info("Fetched %d chars from %s (status %d)", len(html), final_url, status)
        except httpx.TimeoutException:
            error = f"Timeout after {self.config.timeout}s"
            html = ""
            final_url = url
            status = 0
            content_type = ""
        except Exception as exc:
            error = str(exc)
            html = ""
            final_url = url
            status = 0
            content_type = ""

        elapsed = (time.monotonic() - start_time) * 1000

        return FetchResult(
            url=url,
            final_url=final_url,
            html=html,
            status_code=status,
            content_type=content_type,
            fetch_time_ms=round(elapsed, 2),
            method="static",
            error=error,
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def create_fetcher(
    config: PipelineConfig, static: bool = False
) -> DynamicFetcher | StaticFetcher:
    """Factory to create the appropriate fetcher."""
    if static:
        return StaticFetcher(config)
    return DynamicFetcher(config)
