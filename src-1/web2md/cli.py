"""Command-line interface for the web-to-markdown pipeline.

Usage:
    python -m web2md.cli extract <url> [options]
    python -m web2md.cli batch <file> [options]
    python -m web2md.cli prune <input> [options]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

from web2md.config import PipelineConfig

console = Console()


def _setup_logging(verbose: bool) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool) -> None:
    """Web-to-Markdown extraction pipeline.

    Fetch any website, prune junk HTML, and extract clean Markdown
    using a local LLM.
    """
    _setup_logging(verbose)


@main.command()
@click.argument("url")
@click.option("--depth", "-d", default=1, help="Max crawl depth (default: 1)")
@click.option("--max-pages", "-n", default=50, help="Max pages to crawl (default: 50)")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output directory")
@click.option("--static", is_flag=True, help="Use static fetcher (no browser, faster)")
@click.option("--model", "-m", default=None, help="HuggingFace model ID")
@click.option("--no-quantize", is_flag=True, help="Disable 4-bit quantization")
@click.option("--timeout", "-t", default=30, help="Page load timeout in seconds")
@click.option("--no-pagination", is_flag=True, help="Don't follow pagination links")
def extract(
    url: str,
    depth: int,
    max_pages: int,
    output: str | None,
    static: bool,
    model: str | None,
    no_quantize: bool,
    timeout: int,
    no_pagination: bool,
) -> None:
    """Extract content from a URL into Markdown.

    Fetches the page (with full JS rendering by default), prunes HTML,
    crawls linked pages up to --depth, and uses an LLM to produce
    structured Markdown.

    Examples:

        python -m web2md.cli extract "https://example.com"

        python -m web2md.cli extract "https://docs.example.com" --depth 2 --max-pages 100

        python -m web2md.cli extract "https://blog.example.com" --static --model google/gemma-3-4b-it
    """
    from web2md.pipeline import run_pipeline

    overrides = {
        "max_depth": depth,
        "max_pages": max_pages,
        "timeout": timeout,
        "follow_pagination": not no_pagination,
    }
    if model:
        overrides["model_name"] = model
    if no_quantize:
        overrides["quantize_4bit"] = False

    config = PipelineConfig(**overrides)
    output_dir = Path(output) if output else None

    try:
        run_pipeline(url, config, static=static, output_dir=output_dir)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[bold red]Pipeline failed:[/bold red] {exc}")
        logging.getLogger(__name__).debug("Full traceback:", exc_info=True)
        sys.exit(1)


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None, help="Output directory")
@click.option("--depth", "-d", default=1, help="Max crawl depth (default: 1)")
@click.option("--max-pages", "-n", default=50, help="Max pages per URL (default: 50)")
@click.option("--static", is_flag=True, help="Use static fetcher")
@click.option("--model", "-m", default=None, help="HuggingFace model ID")
def batch(
    file: str,
    output: str | None,
    depth: int,
    max_pages: int,
    static: bool,
    model: str | None,
) -> None:
    """Process multiple URLs from a file (one URL per line).

    Example:

        python -m web2md.cli batch urls.txt --output ./results/
    """
    from web2md.pipeline import run_pipeline

    urls_path = Path(file)
    urls = [
        line.strip()
        for line in urls_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not urls:
        console.print("[red]No URLs found in file[/red]")
        sys.exit(1)

    console.print(f"[bold]Processing {len(urls)} URLs[/bold]")

    overrides = {"max_depth": depth, "max_pages": max_pages}
    if model:
        overrides["model_name"] = model

    config = PipelineConfig(**overrides)
    output_dir = Path(output) if output else None

    successes = 0
    failures = 0

    for i, url in enumerate(urls, 1):
        console.print(f"\n[bold]{'='*60}[/bold]")
        console.print(f"[bold]URL {i}/{len(urls)}:[/bold] {url}")
        console.print(f"[bold]{'='*60}[/bold]")

        try:
            run_pipeline(url, config, static=static, output_dir=output_dir)
            successes += 1
        except Exception as exc:
            console.print(f"[red]Failed: {exc}[/red]")
            failures += 1

    console.print(f"\n[bold]Batch complete:[/bold] {successes} succeeded, {failures} failed")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path")
def prune(
    input_file: str,
    output: str | None,
) -> None:
    """Prune an HTML file (remove CSS, nav, scripts, etc.) without LLM.

    Useful for inspecting what the pruner produces before running
    the full extraction pipeline.

    Example:

        python -m web2md.cli prune raw_page.html --output pruned.html
    """
    from web2md.pruner import prune_file

    input_path = Path(input_file)
    output_path = Path(output) if output else input_path.with_stem(input_path.stem + "_pruned")

    stats = prune_file(str(input_path), str(output_path))

    console.print(f"[green]Pruned:[/green] {input_path} -> {output_path}")
    console.print(
        f"  {stats.original_chars:,} -> {stats.pruned_chars:,} chars "
        f"({stats.reduction_pct:.1f}% reduction)"
    )
    console.print(f"  Tags removed: {stats.tags_removed}, Attrs stripped: {stats.attrs_stripped}")


@main.command(name="fetch")
@click.argument("url")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path")
@click.option("--static", is_flag=True, help="Use static fetcher")
@click.option("--timeout", "-t", default=30, help="Timeout in seconds")
def fetch_cmd(
    url: str,
    output: str | None,
    static: bool,
    timeout: int,
) -> None:
    """Fetch raw HTML from a URL (no pruning or extraction).

    Useful for debugging or collecting HTML for manual inspection.

    Example:

        python -m web2md.cli fetch "https://example.com" --output raw.html
    """
    from web2md.fetcher import create_fetcher

    config = PipelineConfig(timeout=timeout)
    fetcher = create_fetcher(config, static=static)

    try:
        fetcher.start()
        result = fetcher.fetch(url)
    finally:
        fetcher.stop()

    if result.error:
        console.print(f"[red]Fetch error:[/red] {result.error}")
        sys.exit(1)

    if output:
        Path(output).write_text(result.html, encoding="utf-8")
        console.print(f"[green]Saved {len(result.html):,} chars to {output}[/green]")
    else:
        console.print(result.html)

    console.print(
        f"\n[dim]Method: {result.method} | Status: {result.status_code} | "
        f"Time: {result.fetch_time_ms:.0f}ms | Size: {len(result.html):,} chars[/dim]"
    )


if __name__ == "__main__":
    main()
