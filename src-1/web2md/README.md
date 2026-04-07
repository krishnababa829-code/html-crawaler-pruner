# Web-to-Markdown Extraction Pipeline

A complete pipeline that takes a URL, fetches the full HTML (including JavaScript-rendered content), prunes unnecessary elements, crawls linked sub-pages, and uses an LLM to produce clean structured Markdown.

## Architecture

```
URL Input
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Fetcher     │────▶│  Pruner      │────▶│  Crawler     │────▶│  Extractor   │
│ (Playwright  │     │ (Remove CSS, │     │ (Follow      │     │ (LLM-based   │
│  + httpx)    │     │  nav, junk)  │     │  sub-pages,  │     │  Markdown    │
│              │     │              │     │  pagination) │     │  conversion) │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                    │
                                                                    ▼
                                                            ┌──────────────┐
                                                            │  Output      │
                                                            │  .md files   │
                                                            │  + media.json│
                                                            └──────────────┘
```

## Pipeline Stages

1. **Fetcher** (`fetcher.py`): Retrieves full HTML from URLs using Playwright (handles JS-rendered SPAs) with httpx as a lightweight fallback for static pages.
2. **Pruner** (`pruner.py`): Strips CSS, scripts, navbars, footers, ads, tracking pixels, empty containers, and event handlers.
3. **Crawler** (`crawler.py`): Discovers and follows internal links, pagination controls, dropdown-loaded content, and sidebar navigation to capture the entire site.
4. **Extractor** (`extractor.py`): Uses a local LLM (Google Gemma via Hugging Face Transformers) to convert pruned HTML into structured Markdown preserving code blocks, tables, math, media URLs, and hyperlinks.
5. **CLI** (`cli.py`): Command-line interface for single-URL and batch processing.

## Installation

```bash
cd src-1
pip install -r requirements.txt

# Install browser engine (first time only)
playwright install chromium
```

### GPU Support (recommended)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Usage

```bash
# Single URL extraction
python -m web2md.cli extract "https://example.com/article"

# With recursive crawling (depth=2)
python -m web2md.cli extract "https://example.com" --depth 2

# Batch URLs from file
python -m web2md.cli batch urls.txt --output ./results/

# Use static fetcher only (no browser, faster)
python -m web2md.cli extract "https://example.com" --static

# Custom LLM model
python -m web2md.cli extract "https://example.com" --model "google/gemma-3-4b-it"

# Prune-only mode (no LLM, just HTML cleanup)
python -m web2md.cli prune input.html --output pruned.html
```

## Configuration

Set via environment variables or `.env` file:

| Variable | Default | Description |
|---|---|---|
| `W2MD_MODEL` | `google/gemma-3-4b-it` | HuggingFace model ID |
| `W2MD_MAX_DEPTH` | `1` | Max crawl depth |
| `W2MD_TIMEOUT` | `30` | Page load timeout (seconds) |
| `W2MD_CONCURRENCY` | `3` | Parallel page fetches |
| `W2MD_OUTPUT_DIR` | `./output` | Default output directory |
| `W2MD_HEADLESS` | `true` | Run browser headless |
| `HF_TOKEN` | `` | HuggingFace token (for gated models) |

## Output Structure

```
output/
├── example.com/
│   ├── index.md              # Main page markdown
│   ├── about.md              # Sub-page markdown
│   ├── blog/
│   │   ├── post-1.md
│   │   └── post-2.md
│   ├── media_manifest.json   # All discovered media URLs
│   └── crawl_report.json     # Crawl metadata and stats
```
