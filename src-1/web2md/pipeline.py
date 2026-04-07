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
from web2md.extractor import ContentExtractor, ExtractionResult

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


def run_pipeline(
    seed_url: str,
    config: PipelineConfig,
    static: bool = False,
    output_dir: Optional[Path] = None,
) -> PipelineReport:
    """Run the complete web-to-markdown pipeline.

    Steps:
    1. Crawl the site starting from seed_url
    2. Load the LLM
    3. Extract Markdown from each crawled page
    4. Write output files and report

    Args:
        seed_url: Starting URL to process.
        config: Pipeline configuration.
        static: Use static fetcher (no browser).
        output_dir: Override output directory.

    Returns:
        PipelineReport with execution details.
    """
    start_time = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()

    # Determine output directory
    domain = urlparse(seed_url).netloc.lower().replace("www.", "")
    out_dir = (output_dir or config.output_dir) / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Crawl
    console.print(f"\n[bold cyan]Step 1/3: Crawling[/bold cyan] {seed_url}")
    console.print(f"  Max depth: {config.max_depth}, Max pages: {config.max_pages}")

    crawl_result: CrawlResult = crawl(seed_url, config, static=static)

    console.print(
        f"  [green]Crawled {crawl_result.total_pages_crawled} pages[/green] "
        f"({crawl_result.total_pages_discovered} links discovered)"
    )
    if crawl_result.errors:
        console.print(f"  [yellow]{len(crawl_result.errors)} errors during crawl[/yellow]")

    # Aggregate media
    all_media: dict[str, list[str]] = {"images": [], "videos": [], "audio": []}
    for page in crawl_result.pages:
        for media_type, urls in page.media_urls.items():
            for u in urls:
                if u not in all_media[media_type]:
                    all_media[media_type].append(u)

    # Step 2: Load LLM
    console.print(f"\n[bold cyan]Step 2/3: Loading LLM[/bold cyan] ({config.model_name})")
    extractor = ContentExtractor(config)
    extractor.load_model()
    console.print("  [green]Model loaded[/green]")

    # Step 3: Extract
    console.print(f"\n[bold cyan]Step 3/3: Extracting content[/bold cyan]")

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
            progress.update(task, description=f"Processing {page.url[:60]}...")

            try:
                result: ExtractionResult = extractor.extract(
                    page.pruned_html,
                    media_urls=page.media_urls,
                )

                # Write markdown file
                md_path = _url_to_filepath(page.final_url, out_dir)
                md_path.parent.mkdir(parents=True, exist_ok=True)
                md_path.write_text(result.markdown, encoding="utf-8")

                total_input_tokens += result.input_tokens
                total_output_tokens += result.output_tokens

                page_outputs.append(PageOutput(
                    url=page.url,
                    output_path=str(md_path.relative_to(out_dir)),
                    markdown_length=len(result.markdown),
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    prune_reduction_pct=page.prune_stats.reduction_pct,
                    fetch_time_ms=page.fetch_result.fetch_time_ms,
                    error=result.error,
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
    extractor.unload_model()

    # Write media manifest
    media_path = out_dir / "media_manifest.json"
    media_path.write_text(json.dumps(all_media, indent=2), encoding="utf-8")

    # Build report
    elapsed = time.monotonic() - start_time
    completed_at = datetime.now(timezone.utc).isoformat()

    report = PipelineReport(
        seed_url=seed_url,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=round(elapsed, 2),
        model_name=config.model_name,
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
    console.print(f"  Output directory: {out_dir}")
    console.print(f"  Pages extracted: {report.pages_extracted}/{report.pages_crawled}")
    console.print(f"  Total tokens: {total_input_tokens} in / {total_output_tokens} out")
    console.print(
        f"  Media found: {report.total_media_found.get('images', 0)} images, "
        f"{report.total_media_found.get('videos', 0)} videos, "
        f"{report.total_media_found.get('audio', 0)} audio"
    )
    console.print(f"  Duration: {elapsed:.1f}s")

    return report
