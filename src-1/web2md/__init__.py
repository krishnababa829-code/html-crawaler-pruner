"""Web-to-Markdown extraction pipeline.

Fetches HTML from URLs (including JS-rendered pages), prunes unnecessary
elements, crawls sub-pages, and uses an LLM to produce structured Markdown.
"""

__version__ = "0.1.0"
