"""Pipeline configuration.

All settings can be overridden via environment variables prefixed with W2MD_
or through a .env file in the project root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseSettings):
    """Configuration for the web-to-markdown pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="W2MD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM settings
    model_name: str = Field(
        default="google/gemma-3-4b-it",
        description="HuggingFace model ID for content extraction",
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="HuggingFace API token for gated models",
        alias="HF_TOKEN",
    )
    max_input_tokens: int = Field(
        default=6144,
        description="Maximum tokens to send to the LLM per chunk",
    )
    max_output_tokens: int = Field(
        default=4096,
        description="Maximum tokens the LLM can generate",
    )
    quantize_4bit: bool = Field(
        default=True,
        description="Load model in 4-bit quantization to reduce VRAM",
    )

    # Fetcher settings
    timeout: int = Field(default=30, description="Page load timeout in seconds")
    headless: bool = Field(default=True, description="Run browser in headless mode")
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        description="Browser user agent string",
    )

    # Crawler settings
    max_depth: int = Field(default=1, description="Maximum crawl depth from seed URL")
    max_pages: int = Field(default=50, description="Maximum total pages to crawl")
    concurrency: int = Field(default=3, description="Parallel page fetch limit")
    same_domain_only: bool = Field(
        default=True,
        description="Only follow links on the same domain",
    )
    follow_pagination: bool = Field(
        default=True,
        description="Detect and follow pagination controls",
    )

    # Output settings
    output_dir: Path = Field(
        default=Path("./output"),
        description="Default output directory",
    )


def get_config(**overrides) -> PipelineConfig:
    """Create a PipelineConfig with optional overrides."""
    return PipelineConfig(**overrides)
