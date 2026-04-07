"""LLM-based content extraction module.

Takes pruned HTML and uses a local LLM (Google Gemma via HuggingFace
Transformers) to produce structured Markdown that preserves:
- Headings, paragraphs, and text structure
- Code blocks (verbatim, with language tags)
- Tables (as Markdown tables)
- Mathematical expressions (LaTeX notation)
- Hyperlinks and references
- Media URL references (images, video, audio)

The model is loaded once and reused across all pages in a crawl.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from web2md.config import PipelineConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Content Extraction Engine. Convert the provided HTML into clean, structured Markdown.

STRICT RULES:
1. HEADINGS: Convert all heading tags (h1-h6) to corresponding Markdown headings (#, ##, etc.).
2. PARAGRAPHS: Preserve all paragraph text. Maintain the reading order.
3. CODE BLOCKS: Any code, functions, scripts, or technical snippets MUST be preserved VERBATIM inside fenced code blocks with the correct language identifier. Do NOT summarize or paraphrase code.
4. TABLES: Convert HTML tables to Markdown table syntax. Preserve all rows and columns.
5. MATH: Convert mathematical expressions to LaTeX notation inside $...$ (inline) or $$...$$ (block).
6. LISTS: Convert ordered and unordered lists to Markdown list syntax.
7. LINKS: Preserve all hyperlinks as [text](url) format.
8. IMAGES: Output as ![alt](src) with the original src URL.
9. VIDEOS/AUDIO: List media URLs in a dedicated section at the end.
10. EMPHASIS: Preserve bold (**text**), italic (*text*), and other inline formatting.
11. BLOCKQUOTES: Convert blockquote elements to > syntax.
12. DO NOT add any commentary, explanation, or filler text. Output ONLY the Markdown content.
13. DO NOT skip or summarize any section. Extract EVERYTHING.
14. If content is truncated due to length, end with [CONTENT_CONTINUES] marker."""


@dataclass
class ExtractionResult:
    """Result of extracting content from a single page."""

    markdown: str
    input_tokens: int
    output_tokens: int
    model_name: str
    truncated: bool = False
    error: Optional[str] = None


class ContentExtractor:
    """LLM-based HTML to Markdown extractor."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._device = None

    def load_model(self) -> None:
        """Load the LLM model and tokenizer."""
        if self._model is not None:
            return

        logger.info("Loading model: %s", self.config.model_name)

        # Determine device
        if torch.cuda.is_available():
            self._device = "cuda"
            logger.info("Using CUDA GPU")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
            logger.info("Using Apple MPS")
        else:
            self._device = "cpu"
            logger.info("Using CPU (this will be slow)")

        # Tokenizer
        token_kwargs = {}
        if self.config.hf_token:
            token_kwargs["token"] = self.config.hf_token

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            **token_kwargs,
        )

        # Model with optional 4-bit quantization
        model_kwargs = {
            "torch_dtype": torch.float16 if self._device != "cpu" else torch.float32,
            **token_kwargs,
        }

        if self.config.quantize_4bit and self._device == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "auto" if self._device != "cpu" else None

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        if self._device == "cpu" or (self._device == "mps" and "device_map" not in model_kwargs):
            self._model = self._model.to(self._device)

        self._model.eval()
        logger.info("Model loaded successfully on %s", self._device)

    def unload_model(self) -> None:
        """Free model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _chunk_html(self, html: str) -> list[str]:
        """Split HTML into chunks that fit within the model's context window.

        Splits on block-level boundaries to avoid breaking tags mid-element.
        """
        if not self._tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Reserve tokens for system prompt + generation
        prompt_overhead = len(self._tokenizer.encode(SYSTEM_PROMPT)) + 100
        max_chunk_tokens = self.config.max_input_tokens - prompt_overhead

        if max_chunk_tokens <= 0:
            max_chunk_tokens = 2048

        # Split on block-level HTML boundaries
        block_pattern = re.compile(
            r"(</?(?:div|section|article|main|p|h[1-6]|ul|ol|table|pre|blockquote|figure|dl)[^>]*>)",
            re.IGNORECASE,
        )
        parts = block_pattern.split(html)

        chunks: list[str] = []
        current_chunk = ""
        current_tokens = 0

        for part in parts:
            part_tokens = len(self._tokenizer.encode(part, add_special_tokens=False))

            if current_tokens + part_tokens > max_chunk_tokens:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # If a single part exceeds the limit, force-split by characters
                if part_tokens > max_chunk_tokens:
                    char_limit = max_chunk_tokens * 3  # rough chars-per-token estimate
                    for i in range(0, len(part), char_limit):
                        sub = part[i:i + char_limit].strip()
                        if sub:
                            chunks.append(sub)
                    current_chunk = ""
                    current_tokens = 0
                else:
                    current_chunk = part
                    current_tokens = part_tokens
            else:
                current_chunk += part
                current_tokens += part_tokens

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [html[:self.config.max_input_tokens * 3]]

    def _generate(self, html_chunk: str) -> str:
        """Run a single chunk through the LLM."""
        messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nHTML:\n{html_chunk}"},
        ]

        # Try chat template first, fall back to raw prompt
        try:
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            input_text = f"{SYSTEM_PROMPT}\n\nHTML:\n{html_chunk}\n\nMarkdown:\n"

        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        )

        if self._device and "device_map" not in str(self._model.hf_device_map if hasattr(self._model, 'hf_device_map') else ""):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_output_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (skip input)
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)

        return text.strip()

    def extract(
        self,
        pruned_html: str,
        media_urls: Optional[dict[str, list[str]]] = None,
    ) -> ExtractionResult:
        """Extract structured Markdown from pruned HTML.

        Args:
            pruned_html: HTML that has already been pruned.
            media_urls: Pre-extracted media URLs to append to output.

        Returns:
            ExtractionResult with the generated Markdown.
        """
        if self._model is None:
            self.load_model()

        chunks = self._chunk_html(pruned_html)
        logger.info("Processing %d chunk(s)", len(chunks))

        markdown_parts: list[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        truncated = False
        error = None

        for i, chunk in enumerate(chunks):
            try:
                input_tokens = len(
                    self._tokenizer.encode(chunk, add_special_tokens=False)
                )
                total_input_tokens += input_tokens

                result = self._generate(chunk)

                output_tokens = len(
                    self._tokenizer.encode(result, add_special_tokens=False)
                )
                total_output_tokens += output_tokens

                if "[CONTENT_CONTINUES]" in result:
                    truncated = True
                    result = result.replace("[CONTENT_CONTINUES]", "").strip()

                markdown_parts.append(result)
                logger.info(
                    "Chunk %d/%d: %d input -> %d output tokens",
                    i + 1, len(chunks), input_tokens, output_tokens,
                )

            except Exception as exc:
                error = f"Chunk {i + 1} failed: {exc}"
                logger.error(error)
                # Include raw text as fallback
                from bs4 import BeautifulSoup
                fallback = BeautifulSoup(chunk, "lxml").get_text("\n", strip=True)
                markdown_parts.append(f"<!-- Extraction failed for this section -->\n{fallback}")

        # Combine all chunks
        markdown = "\n\n".join(markdown_parts)

        # Append media URLs section if present
        if media_urls:
            media_section = self._format_media_section(media_urls)
            if media_section:
                markdown += "\n\n" + media_section

        return ExtractionResult(
            markdown=markdown,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            model_name=self.config.model_name,
            truncated=truncated,
            error=error,
        )

    def _format_media_section(self, media_urls: dict[str, list[str]]) -> str:
        """Format media URLs into a Markdown section."""
        sections = []

        if media_urls.get("images"):
            lines = ["## Media: Images", ""]
            for url in media_urls["images"]:
                lines.append(f"- ![image]({url})")
            sections.append("\n".join(lines))

        if media_urls.get("videos"):
            lines = ["## Media: Videos", ""]
            for url in media_urls["videos"]:
                lines.append(f"- [Video]({url})")
            sections.append("\n".join(lines))

        if media_urls.get("audio"):
            lines = ["## Media: Audio", ""]
            for url in media_urls["audio"]:
                lines.append(f"- [Audio]({url})")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, *args):
        self.unload_model()
