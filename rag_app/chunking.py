"""Split a parsed document into chunks small enough to embed.

Splits on document structure (headings, tables) rather than blind character
counts, and keeps the filename, page and heading on each chunk so answers can
cite their source.

Usage:

    from rag_app.parsing import parse
    from rag_app.chunking import chunk_document

    doc = parse("data/samples/report.pdf")
    for chunk in chunk_document(doc):
        print(chunk.page, chunk.heading, chunk.token_count)
        print(chunk.text)

Each chunk's `text` has its headings prepended. That context is what makes a
chunk understandable on its own: "up to three days per week" means little
until you know it sits under "Remote Work Policy > Eligibility".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from rag_app.config import Settings, get_settings
from rag_app.parsing import ParsedDocument

log = logging.getLogger(__name__)

# Separator used when a chunk sits under nested headings.
HEADING_SEPARATOR = " > "


@dataclass(frozen=True)
class Chunk:
    """One piece of a document, ready to be embedded."""

    text: str
    """The chunk with its headings prepended. This is what gets embedded."""

    source: str
    """Filename the chunk came from."""

    page: int | None
    """First page the chunk appears on. None for formats without pages."""

    heading: str | None
    """Heading path, e.g. "Remote Work Policy > Eligibility"."""

    token_count: int
    """Length in tokens, measured with the embedding model's own tokenizer."""

    index: int
    """Position within its document, starting at 0."""


def chunk_document(
    document: ParsedDocument,
    settings: Settings | None = None,
) -> list[Chunk]:
    """Split one parsed document into chunks.

    Args:
        document: A document from `rag_app.parsing.parse`.
        settings: Overrides the app settings, mainly for tests.
    """
    settings = settings or get_settings()
    tokenizer, chunker = _chunker(settings)

    chunks: list[Chunk] = []
    for index, raw in enumerate(chunker.chunk(dl_doc=document.document)):
        text = chunker.contextualize(chunk=raw)
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                text=text,
                source=document.source,
                page=_first_page(raw),
                heading=_heading_path(raw),
                token_count=tokenizer.count_tokens(text),
                index=index,
            )
        )

    if not chunks:
        # Usually an image or scanned page read with OCR turned off. Saying so
        # beats letting the document vanish from the index without explanation.
        log.warning(
            "No text found in '%s', so it produced no chunks. If it is a "
            "scanned document or an image, switch OCR on.",
            document.source,
        )
    else:
        log.info("Split '%s' into %d chunk(s)", document.source, len(chunks))

    return chunks


def chunk_documents(
    documents: list[ParsedDocument],
    settings: Settings | None = None,
) -> list[Chunk]:
    """Split several parsed documents, returning one flat list of chunks."""
    settings = settings or get_settings()
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, settings))
    return chunks


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------

# Loading a tokenizer is slow, so reuse it per (model, size) combination.
_cache: dict[tuple[str, int], tuple[HuggingFaceTokenizer, HybridChunker]] = {}


def _chunker(settings: Settings) -> tuple[HuggingFaceTokenizer, HybridChunker]:
    """Build a chunker that measures length the way the embedding model does.

    The chunk size is capped at whatever the embedding model can actually read.
    Going over that limit would not fail loudly; the far end of every oversized
    chunk would just be ignored at embedding time, so we clamp it here instead.
    """
    key = (settings.embedding_model, settings.chunk_max_tokens)
    if key in _cache:
        return _cache[key]

    tokenizer = HuggingFaceTokenizer.from_pretrained(settings.embedding_model)
    model_limit = tokenizer.get_max_tokens()
    wanted = settings.chunk_max_tokens

    if wanted > model_limit:
        log.warning(
            "chunk_max_tokens is %d but '%s' can only read %d tokens; "
            "using %d so no text is lost.",
            wanted,
            settings.embedding_model,
            model_limit,
            model_limit,
        )
        wanted = model_limit

    if wanted != model_limit:
        tokenizer = tokenizer.model_copy(update={"max_tokens": wanted})

    chunker = HybridChunker(tokenizer=tokenizer)
    _cache[key] = (tokenizer, chunker)
    return _cache[key]


def _heading_path(raw_chunk) -> str | None:
    """Join nested headings into one readable trail."""
    headings = getattr(raw_chunk.meta, "headings", None)
    return HEADING_SEPARATOR.join(headings) if headings else None


def _first_page(raw_chunk) -> int | None:
    """Return the earliest page the chunk touches, if the format has pages."""
    pages = [
        prov.page_no
        for item in raw_chunk.meta.doc_items
        for prov in (item.prov or [])
    ]
    return min(pages) if pages else None
