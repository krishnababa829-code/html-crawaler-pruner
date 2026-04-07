"""HTML pruning module.

Removes non-content elements from raw HTML to reduce noise before
LLM processing. This dramatically cuts token count and improves
extraction quality.

Pruning stages:
1. Remove invisible/structural tags (script, style, nav, footer, etc.)
2. Remove elements by class/id patterns (sidebar, widget, ad, popup, etc.)
3. Strip event handler attributes and data-* attributes
4. Remove empty containers that hold no text or meaningful children
5. Collapse excessive whitespace
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Comment, Tag

logger = logging.getLogger(__name__)

# Tags that never contain user-visible content
INVISIBLE_TAGS = frozenset([
    "script", "noscript", "style", "meta", "link", "iframe", "object",
    "embed", "svg", "path", "head", "template", "slot",
    "base", "basefont", "applet", "param", "source", "track",
    "map", "area", "canvas",
])

# Structural/chrome tags to remove (navigation, footers, etc.)
CHROME_TAGS = frozenset([
    "nav", "footer", "aside", "header",
])

# Class/ID substrings that indicate non-content elements
GARBAGE_PATTERNS = frozenset([
    "sidebar", "widget", "menu", "archive", "comment", "advertisement",
    "promo", "popup", "modal", "overlay", "cookie", "consent", "banner",
    "newsletter", "subscribe", "social", "share", "related", "recommend",
    "breadcrumb", "pagination", "pager", "toolbar", "tooltip",
    "dropdown", "carousel", "slider", "lightbox", "gallery-nav",
    "ad-", "ads-", "advert", "sponsor", "tracking", "analytics",
    "gdpr", "privacy-notice", "notification", "alert-banner",
    "skip-link", "screen-reader", "sr-only", "visually-hidden",
    "print-only", "no-print",
])

# Attributes to strip from remaining tags
STRIP_ATTRS = frozenset([
    "class", "style", "onclick", "onload", "onmouseover", "onfocus",
    "onblur", "onchange", "onsubmit", "onerror", "onkeydown",
    "onkeyup", "onkeypress", "onmouseout", "onmouseenter",
    "onmouseleave", "ondblclick", "oncontextmenu", "onscroll",
    "onresize", "ontouchstart", "ontouchend", "ontouchmove",
    "role", "aria-hidden", "tabindex", "draggable",
])

# Self-closing tags that should not be removed even if "empty"
SELF_CLOSING = frozenset([
    "br", "hr", "img", "input", "col", "wbr",
])

# Tags whose content (code, math, pre) must be preserved verbatim
PRESERVE_CONTENT_TAGS = frozenset([
    "code", "pre", "kbd", "samp", "var",
    "math", "mrow", "mi", "mo", "mn", "msup", "msub", "mfrac",
    "annotation", "semantics",
    "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption",
])


@dataclass
class PruneStats:
    """Statistics from a pruning operation."""

    original_chars: int
    pruned_chars: int
    tags_removed: int
    attrs_stripped: int

    @property
    def reduction_pct(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return (1 - self.pruned_chars / self.original_chars) * 100


def _matches_garbage(value: str) -> bool:
    """Check if a class or ID value matches any garbage pattern."""
    lower = value.lower()
    return any(pat in lower for pat in GARBAGE_PATTERNS)


def prune_html(raw_html: str, keep_chrome: bool = False) -> tuple[str, PruneStats]:
    """Prune non-content elements from HTML.

    Args:
        raw_html: The raw HTML string to prune.
        keep_chrome: If True, keep nav/footer/header/aside tags.

    Returns:
        Tuple of (pruned HTML string, pruning statistics).
    """
    if not raw_html or not isinstance(raw_html, str):
        return "", PruneStats(0, 0, 0, 0)

    original_len = len(raw_html)
    soup = BeautifulSoup(raw_html, "lxml")
    tags_removed = 0
    attrs_stripped = 0

    # Stage 1: Remove invisible tags
    for tag_name in INVISIBLE_TAGS:
        for el in soup.find_all(tag_name):
            el.decompose()
            tags_removed += 1

    # Stage 2: Remove chrome tags (nav, footer, etc.)
    if not keep_chrome:
        for tag_name in CHROME_TAGS:
            for el in soup.find_all(tag_name):
                el.decompose()
                tags_removed += 1

    # Stage 3: Remove elements by class/id garbage patterns
    for el in soup.find_all(True):
        if not isinstance(el, Tag):
            continue
        # Check class attribute
        classes = el.get("class", [])
        if isinstance(classes, list) and any(
            _matches_garbage(c) for c in classes
        ):
            el.decompose()
            tags_removed += 1
            continue
        # Check id attribute
        el_id = el.get("id", "")
        if el_id and _matches_garbage(el_id):
            el.decompose()
            tags_removed += 1
            continue

    # Stage 4: Remove HTML comments
    for comment in soup.find_all(string=lambda x: isinstance(x, Comment)):
        comment.extract()
        tags_removed += 1

    # Stage 5: Strip junk attributes but preserve href, src, alt, title
    for tag in soup.find_all(True):
        if not isinstance(tag, Tag):
            continue
        # Skip content-preservation tags
        if tag.name in PRESERVE_CONTENT_TAGS:
            # Only strip event handlers from these
            removable = [
                a for a in tag.attrs
                if a.startswith("on") or a.startswith("data-")
            ]
        else:
            removable = [
                a for a in tag.attrs
                if a in STRIP_ATTRS or a.startswith("data-")
            ]
        for attr in removable:
            del tag[attr]
            attrs_stripped += 1

    # Stage 6: Remove empty containers (but keep self-closing and content tags)
    changed = True
    while changed:
        changed = False
        for tag in soup.find_all(True):
            if not isinstance(tag, Tag):
                continue
            if tag.name in SELF_CLOSING or tag.name in PRESERVE_CONTENT_TAGS:
                continue
            if (
                not tag.get_text(strip=True)
                and not tag.find_all(list(SELF_CLOSING | PRESERVE_CONTENT_TAGS))
            ):
                tag.decompose()
                tags_removed += 1
                changed = True

    # Stage 7: Collapse whitespace
    result = str(soup)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = result.strip()

    stats = PruneStats(
        original_chars=original_len,
        pruned_chars=len(result),
        tags_removed=tags_removed,
        attrs_stripped=attrs_stripped,
    )
    logger.info(
        "Pruned %d -> %d chars (%.1f%% reduction, %d tags removed)",
        original_len, len(result), stats.reduction_pct, tags_removed,
    )
    return result, stats


def prune_file(input_path: str, output_path: str) -> PruneStats:
    """Prune an HTML file and write the result."""
    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()
    pruned, stats = prune_html(raw)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pruned)
    return stats
