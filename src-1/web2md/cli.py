"""Command-line interface for the web-to-markdown pipeline.

Usage:
    python -m web2md.cli extract <url> [options]
    python -m web2md.cli batch <file> [options]
    python -m web2md.cli prune <input> [options]
    python -m web2md.cli fetch <url> [options]
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def _setup_logging(verbose: bool) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
        force=True,
    )


def _check_playwright() -> bool:
    """Check if Playwright browsers are installed."""
    try:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        try:
            browser = pw.chromium.launch(headless=True)
            browser.close()
        finally:
            pw.stop()
        return True
    except Exception as exc:
        error_msg = str(exc).lower()
        if "executable doesn't exist" in error_msg or "browsertype.launch" in error_msg:
            console.print(
                "\n[bold red]Playwright browser not installed![/bold red]\n"
                "Run this command to install it:\n\n"
                "  [bold cyan]playwright install chromium[/bold cyan]\n"
            )
        else:
            console.print(f"\n[bold red]Playwright check failed:[/bold red] {exc}\n")
        return False


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
@click.option(
    "--skip-llm", is_flag=True,
    help="Skip LLM extraction, output pruned HTML + fallback text only",
)
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
    skip_llm: bool,
) -> None:
    """Extract content from a URL into Markdown.

    Fetches the page (with full JS rendering by default), prunes HTML,
    crawls linked pages up to --depth, and uses an LLM to produce
    structured Markdown.

    Use --skip-llm to only crawl and prune (no model needed).
    Use --static to skip browser rendering (faster, but misses JS content).

    Examples:

        python -m web2md.cli extract "https://example.com"

        python -m web2md.cli extract "https://example.com" --skip-llm

        python -m web2md.cli extract "https://example.com" --static

        python -m web2md.cli extract "https://docs.example.com" --depth 2
    """
    from web2md.config import PipelineConfig
    from web2md.pipeline import run_pipeline

    # Pre-flight check: Playwright installed?
    if not static:
        console.print("[dim]Checking Playwright installation...[/dim]")
        if not _check_playwright():
            console.print(
                "[yellow]Tip: Use --static flag to skip browser and use HTTP fetch instead[/yellow]"
            )
            sys.exit(1)
        console.print("[dim]Playwright OK[/dim]")

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
        run_pipeline(
            url, config, static=static, output_dir=output_dir, skip_llm=skip_llm,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[bold red]Pipeline failed:[/bold red] {exc}")
        console.print(f"\n[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


@main.command()
@click.argument("file", type=click.Path())  # removed exists=True
@click.option("--output", "-o", type=click.Path(), default=None, help="Output directory")
@click.option("--depth", "-d", default=1, help="Max crawl depth (default: 1)")
@click.option("--max-pages", "-n", default=50, help="Max pages per URL (default: 50)")
@click.option("--static", is_flag=True, help="Use static fetcher")
@click.option("--model", "-m", default=None, help="HuggingFace model ID")
@click.option("--skip-llm", is_flag=True, help="Skip LLM, use fallback text extraction")
def batch(
    file: str,
    output: str | None,
    depth: int,
    max_pages: int,
    static: bool,
    model: str | None,
    skip_llm: bool,
) -> None:
    """Process multiple URLs from a file (one URL per line).

    If the file doesn't exist, a template will be created for you.

    Example:

        python -m web2md.cli batch urls.txt --output ./results/
    """
    from web2md.config import PipelineConfig
    from web2md.pipeline import run_pipeline

    urls_path = Path(file).resolve()

    # If file doesn't exist, create a template
    if not urls_path.exists():
        urls_path.parent.mkdir(parents=True, exist_ok=True)
        template = (
            "# Web-to-Markdown batch URL list\n"
            "# Add one URL per line. Lines starting with # are ignored.\n"
            "#\n"
            "# Example:\n"
            "# https://docs.pytorch.org/xla/release/r2.8/index.html\n"
            "# https://example.com/blog\n"
        )
        urls_path.write_text(template, encoding="utf-8")
        console.print(
            f"[yellow]File not found. Created template at:[/yellow]\n"
            f"  [bold]{urls_path}[/bold]\n\n"
            f"Add your URLs (one per line) and run the command again."
        )
        sys.exit(0)

    urls = [
        line.strip()
        for line in urls_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not urls:
        console.print(
            f"[red]No URLs found in {urls_path}[/red]\n"
            f"Add URLs (one per line) and run again."
        )
        sys.exit(1)

    console.print(f"[bold]Processing {len(urls)} URLs from {urls_path}[/bold]")

    overrides = {"max_depth": depth, "max_pages": max_pages}
    if model:
        overrides["model_name"] = model

    config = PipelineConfig(**overrides)
    output_dir = Path(output).resolve() if output else None

    successes = 0
    failures = 0

    for i, url in enumerate(urls, 1):
        console.print(f"\n[bold]{'='*60}[/bold]")
        console.print(f"[bold]URL {i}/{len(urls)}:[/bold] {url}")
        console.print(f"[bold]{'='*60}[/bold]")

        try:
            run_pipeline(
                url, config, static=static, output_dir=output_dir,
                skip_llm=skip_llm,
            )
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
    from web2md.config import PipelineConfig
    from web2md.fetcher import create_fetcher

    if not static:
        if not _check_playwright():
            console.print("[yellow]Tip: Use --static to skip browser[/yellow]")
            sys.exit(1)

    config = PipelineConfig(timeout=timeout)
    fetcher = create_fetcher(config, static=static)

    console.print(f"[dim]Fetching {url} ({'static' if static else 'dynamic'} mode)...[/dim]")

    try:
        fetcher.start()
        result = fetcher.fetch(url)
    finally:
        fetcher.stop()

    if result.error:
        console.print(f"[red]Fetch error:[/red] {result.error}")
        sys.exit(1)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.html, encoding="utf-8")
        console.print(f"[green]Saved {len(result.html):,} chars to {output}[/green]")
    else:
        # Print first 2000 chars to avoid flooding terminal
        preview = result.html[:2000]
        console.print(preview)
        if len(result.html) > 2000:
            console.print(f"\n[dim]... ({len(result.html) - 2000:,} more chars, use -o to save full output)[/dim]")

    console.print(
        f"\n[dim]Method: {result.method} | Status: {result.status_code} | "
        f"Time: {result.fetch_time_ms:.0f}ms | Size: {len(result.html):,} chars[/dim]"
    )


if __name__ == "__main__":
    main()
