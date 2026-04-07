"""End-to-end pipeline orchestrator.

Coordinates fetching, pruning, crawling, and extraction into a single
workflow. Handles output directory structure and report generation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from web2md.config import PipelineConfig
from web2md.crawler import CrawlResult, CrawledPage, crawl

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class PageOutput:
    """Output for a single processed page."""

    url: str
    output_path: str
    markdown_length: int
    input_tokens: int
    output_tokens: int
    prune_reduction_pct: float
    fetch_time_ms: float
    error: Optional[str] = None


@dataclass
class PipelineReport:
    """Complete pipeline execution report."""

    seed_url: str
    started_at: str
    completed_at: str
    duration_seconds: float
    model_name: str
    pages_crawled: int
    pages_extracted: int
    total_input_tokens: int
    total_output_tokens: int
    total_media_found: dict[str, int]
    page_outputs: list[PageOutput] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)


def _url_to_filepath(url: str, base_dir: Path) -> Path:
    """Convert a URL to a filesystem path for the output .md file."""
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        path = "index"
    # Remove file extension if present
    if "." in path.split("/")[-1]:
        path = path.rsplit(".", 1)[0]
    # Sanitize
    path = path.replace("..", "_").replace("~", "_")
    return base_dir / f"{path}.md"


def _fallback_text_extract(
    pruned_html: str, media_urls: Optional[dict[str, list[str]]] = None
) -> str:
    """Extract text from pruned HTML using BeautifulSoup (no LLM needed).

    This is the fallback when --skip-llm is used or when the LLM
    is unavailable. Produces reasonable Markdown from the HTML structure.
    """
    soup = BeautifulSoup(pruned_html, "lxml")
    lines: list[str] = []

    for el in soup.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "code",
         "li", "th", "td", "blockquote", "a", "img"]
    ):
        tag = el.name

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            text = el.get_text(strip=True)
            if text:
                lines.append(f"\n{'#' * level} {text}\n")

        elif tag == "p":
            text = el.get_text(strip=True)
            if text:
                lines.append(f"\n{text}\n")

        elif tag == "pre":
            code_el = el.find("code")
            code_text = code_el.get_text() if code_el else el.get_text()
            # Try to detect language from class
            lang = ""
            classes = el.get("class", []) or []
            if code_el:
                classes = code_el.get("class", []) or []
            for cls in classes:
                if isinstance(cls, str) and cls.startswith(("language-", "lang-")):
                    lang = cls.split("-", 1)[1]
                    break
            lines.append(f"\n```{lang}\n{code_text}\n```\n")

        elif tag == "code" and el.parent and el.parent.name != "pre":
            text = el.get_text()
            if text:
                lines.append(f"`{text}`")

        elif tag == "li":
            text = el.get_text(strip=True)
            if text:
                parent = el.parent
                if parent and parent.name == "ol":
                    lines.append(f"1. {text}")
                else:
                    lines.append(f"- {text}")

        elif tag == "blockquote":
            text = el.get_text(strip=True)
            if text:
                lines.append(f"\n> {text}\n")

        elif tag == "img":
            src = el.get("src", "")
            alt = el.get("alt", "image")
            if src:
                lines.append(f"![{alt}]({src})")

    markdown = "\n".join(lines)

    # Append media section
    if media_urls:
        media_parts = []
        if media_urls.get("images"):
            media_parts.append("\n## Media: Images\n")
            for url in media_urls["images"]:
                media_parts.append(f"- ![image]({url})")
        if media_urls.get("videos"):
            media_parts.append("\n## Media: Videos\n")
            for url in media_urls["videos"]:
                media_parts.append(f"- [Video]({url})")
        if media_urls.get("audio"):
            media_parts.append("\n## Media: Audio\n")
            for url in media_urls["audio"]:
                media_parts.append(f"- [Audio]({url})")
        if media_parts:
            markdown += "\n" + "\n".join(media_parts)

    return markdown.strip()


def run_pipeline(
    seed_url: str,
    config: PipelineConfig,
    static: bool = False,
    output_dir: Optional[Path] = None,
    skip_llm: bool = False,
) -> PipelineReport:
    """Run the complete web-to-markdown pipeline.

    Steps:
    1. Crawl the site starting from seed_url
    2. Load the LLM (unless skip_llm=True)
    3. Extract Markdown from each crawled page
    4. Write output files and report

    Args:
        seed_url: Starting URL to process.
        config: Pipeline configuration.
        static: Use static fetcher (no browser).
        output_dir: Override output directory.
        skip_llm: If True, skip LLM and use fallback text extraction.

    Returns:
        PipelineReport with execution details.
    """
    start_time = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()

    # Determine output directory
    domain = urlparse(seed_url).netloc.lower().replace("www.", "")
    out_dir = (output_dir or config.output_dir) / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]Output directory:[/bold] {out_dir.resolve()}")

    # ── Step 1: Crawl ──────────────────────────────────────────────
    console.print(f"\n[bold cyan]Step 1/3: Crawling[/bold cyan] {seed_url}")
    console.print(f"  Max depth: {config.max_depth}, Max pages: {config.max_pages}")
    console.print(f"  Fetcher: {'static (httpx)' if static else 'dynamic (Playwright)'}")

    crawl_result: CrawlResult = crawl(seed_url, config, static=static)

    console.print(
        f"  [green]Crawled {crawl_result.total_pages_crawled} pages[/green] "
        f"({crawl_result.total_pages_discovered} links discovered)"
    )
    if crawl_result.errors:
        console.print(f"  [yellow]{len(crawl_result.errors)} errors during crawl[/yellow]")
        for err in crawl_result.errors[:5]:
            console.print(f"    [dim]{err['url']}: {err['error']}[/dim]")

    if not crawl_result.pages:
        console.print("[bold red]No pages were successfully crawled. Aborting.[/bold red]")
        return PipelineReport(
            seed_url=seed_url,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=round(time.monotonic() - start_time, 2),
            model_name=config.model_name,
            pages_crawled=0,
            pages_extracted=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_media_found={"images": 0, "videos": 0, "audio": 0},
            errors=crawl_result.errors,
        )

    # Aggregate media
    all_media: dict[str, list[str]] = {"images": [], "videos": [], "audio": []}
    for page in crawl_result.pages:
        for media_type, urls in page.media_urls.items():
            for u in urls:
                if u not in all_media[media_type]:
                    all_media[media_type].append(u)

    # ── Step 2: Load LLM (or skip) ────────────────────────────────
    extractor = None
    use_llm = False

    if skip_llm:
        console.print(
            f"\n[bold cyan]Step 2/3: LLM[/bold cyan] [yellow]SKIPPED[/yellow] (--skip-llm)"
        )
        console.print("  Using fallback BeautifulSoup text extraction")
    else:
        console.print(f"\n[bold cyan]Step 2/3: Loading LLM[/bold cyan] ({config.model_name})")
        try:
            from web2md.extractor import ContentExtractor
            extractor = ContentExtractor(config)
            extractor.load_model()
            use_llm = True
            console.print("  [green]Model loaded[/green]")
        except ImportError as exc:
            console.print(
                f"  [yellow]LLM dependencies not available:[/yellow] {exc}\n"
                f"  [yellow]Falling back to BeautifulSoup text extraction.[/yellow]\n"
                f"  To enable LLM: pip install torch transformers accelerate bitsandbytes"
            )
        except Exception as exc:
            console.print(
                f"  [yellow]Failed to load LLM:[/yellow] {exc}\n"
                f"  [yellow]Falling back to BeautifulSoup text extraction.[/yellow]"
            )

    # ── Step 3: Extract ────────────────────────────────────────────
    console.print(f"\n[bold cyan]Step 3/3: Extracting content[/bold cyan]")
    if not use_llm:
        console.print("  [dim]Mode: fallback (BS4 text extraction)[/dim]")

    page_outputs: list[PageOutput] = []
    total_input_tokens = 0
    total_output_tokens = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting...", total=len(crawl_result.pages))

        for page in crawl_result.pages:
            short_url = page.url[:60] + ("..." if len(page.url) > 60 else "")
            progress.update(task, description=f"Processing {short_url}")

            try:
                if use_llm and extractor is not None:
                    result = extractor.extract(
                        page.pruned_html,
                        media_urls=page.media_urls,
                    )
                    markdown = result.markdown
                    input_tok = result.input_tokens
                    output_tok = result.output_tokens
                    error = result.error
                else:
                    markdown = _fallback_text_extract(
                        page.pruned_html, media_urls=page.media_urls,
                    )
                    input_tok = 0
                    output_tok = 0
                    error = None

                # Write markdown file
                md_path = _url_to_filepath(page.final_url, out_dir)
                md_path.parent.mkdir(parents=True, exist_ok=True)
                md_path.write_text(markdown, encoding="utf-8")

                total_input_tokens += input_tok
                total_output_tokens += output_tok

                page_outputs.append(PageOutput(
                    url=page.url,
                    output_path=str(md_path.relative_to(out_dir)),
                    markdown_length=len(markdown),
                    input_tokens=input_tok,
                    output_tokens=output_tok,
                    prune_reduction_pct=page.prune_stats.reduction_pct,
                    fetch_time_ms=page.fetch_result.fetch_time_ms,
                    error=error,
                ))

            except Exception as exc:
                logger.error("Failed to extract %s: %s", page.url, exc)
                page_outputs.append(PageOutput(
                    url=page.url,
                    output_path="",
                    markdown_length=0,
                    input_tokens=0,
                    output_tokens=0,
                    prune_reduction_pct=page.prune_stats.reduction_pct,
                    fetch_time_ms=page.fetch_result.fetch_time_ms,
                    error=str(exc),
                ))

            progress.advance(task)

    # Unload model
    if extractor is not None:
        extractor.unload_model()

    # Write media manifest
    media_path = out_dir / "media_manifest.json"
    media_path.write_text(json.dumps(all_media, indent=2), encoding="utf-8")

    # Write pruned HTML for debugging
    for page in crawl_result.pages:
        pruned_path = _url_to_filepath(page.final_url, out_dir).with_suffix(".pruned.html")
        pruned_path.parent.mkdir(parents=True, exist_ok=True)
        pruned_path.write_text(page.pruned_html, encoding="utf-8")

    # Build report
    elapsed = time.monotonic() - start_time
    completed_at = datetime.now(timezone.utc).isoformat()

    report = PipelineReport(
        seed_url=seed_url,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=round(elapsed, 2),
        model_name=config.model_name if use_llm else "fallback-bs4",
        pages_crawled=crawl_result.total_pages_crawled,
        pages_extracted=sum(1 for p in page_outputs if not p.error),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_media_found={
            k: len(v) for k, v in all_media.items()
        },
        page_outputs=page_outputs,
        errors=crawl_result.errors,
    )

    # Write report
    report_path = out_dir / "crawl_report.json"
    report_path.write_text(
        json.dumps(asdict(report), indent=2, default=str),
        encoding="utf-8",
    )

    # Summary
    console.print(f"\n[bold green]Pipeline complete![/bold green]")
    console.print(f"  Output directory: {out_dir.resolve()}")
    console.print(f"  Pages extracted: {report.pages_extracted}/{report.pages_crawled}")
    if use_llm:
        console.print(f"  Total tokens: {total_input_tokens} in / {total_output_tokens} out")
    console.print(
        f"  Media found: {report.total_media_found.get('images', 0)} images, "
        f"{report.total_media_found.get('videos', 0)} videos, "
        f"{report.total_media_found.get('audio', 0)} audio"
    )
    console.print(f"  Duration: {elapsed:.1f}s")

    # List output files
    console.print(f"\n[bold]Output files:[/bold]")
    for po in page_outputs:
        if po.output_path:
            status = "[green]OK[/green]" if not po.error else f"[yellow]{po.error}[/yellow]"
            console.print(f"  {po.output_path} ({po.markdown_length:,} chars) {status}")

    return report
