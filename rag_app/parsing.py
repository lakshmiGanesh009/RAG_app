"""Turn any supported file into one common document format.

Handles PDF, DOCX, HTML and images (with OCR) so the rest of the app never
needs to care which format a document came from.

Usage:

    from rag_app.parsing import parse, parse_directory

    doc = parse("data/samples/report.pdf")
    print(doc.text)                 # the whole document as markdown
    print(doc.blocks[0].page)       # where each piece of text came from

    for doc in parse_directory("data/samples"):
        print(doc.source, doc.page_count)

Parsing a PDF for the first time downloads docling's layout models, which
takes a few minutes. Later runs reuse the cached copy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from rag_app.config import Settings, get_settings

log = logging.getLogger(__name__)

# The file types this app accepts. Docling reads more than these, but this is
# the set we have chosen to support and test.
SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".html",
        ".htm",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    }
)

# Item labels docling uses for headings.
_HEADING_LABELS = {"section_header", "title"}


class ParseError(Exception):
    """Raised when a document cannot be read."""


@dataclass(frozen=True)
class Block:
    """One piece of text, plus where it came from.

    Carrying the page and heading here is what lets answers cite their source
    later on.
    """

    text: str
    page: int | None
    heading: str | None


@dataclass(frozen=True)
class ParsedDocument:
    """A document after parsing, independent of what format it started as."""

    source: str
    """Filename, or the URL it was fetched from."""

    text: str
    """The whole document as markdown. Good for reading and for checking."""

    page_count: int
    """Number of pages. 0 for formats that have no pages, such as HTML."""

    blocks: list[Block] = field(repr=False)
    """The document split into pieces, each knowing its page and heading."""

    document: DoclingDocument = field(repr=False)
    """Docling's structured form. Used for chunking in Phase 3."""


def is_supported(source: str | Path) -> bool:
    """Return True if this looks like something we can parse."""
    if _is_url(source):
        return True
    return Path(source).suffix.lower() in SUPPORTED_SUFFIXES


def parse(source: str | Path, settings: Settings | None = None) -> ParsedDocument:
    """Read one document and return it in the common format.

    Args:
        source: Path to a local file, or an http(s) URL.
        settings: Overrides the app settings, mainly for tests.

    Raises:
        ParseError: if the file is missing, unsupported, or unreadable.
    """
    settings = settings or get_settings()
    is_url = _is_url(source)

    if not is_url:
        path = Path(source)
        if not path.exists():
            raise ParseError(f"File not found: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ParseError(
                f"Cannot read '{path.name}': {path.suffix or 'no extension'} is "
                f"not supported. Supported types: {_supported_list()}."
            )
        source_name = path.name
    else:
        source_name = str(source)

    try:
        result = _converter(settings).convert(str(source))
    except Exception as error:  # docling raises a variety of types
        raise ParseError(f"Could not read '{source_name}': {error}") from error

    document = result.document
    return ParsedDocument(
        source=source_name,
        text=document.export_to_markdown(),
        page_count=document.num_pages(),
        blocks=_blocks_of(document),
        document=document,
    )


def parse_directory(
    folder: str | Path,
    settings: Settings | None = None,
    recursive: bool = True,
) -> list[ParsedDocument]:
    """Read every supported document in a folder.

    Files that are unsupported or unreadable are reported as warnings and
    skipped, so one bad file cannot stop the whole run.

    Args:
        folder: Folder to read.
        settings: Overrides the app settings, mainly for tests.
        recursive: Also read sub-folders. On by default.

    Raises:
        ParseError: only if the folder itself does not exist.
    """
    root = Path(folder)
    if not root.is_dir():
        raise ParseError(f"Not a folder: {root}")

    settings = settings or get_settings()
    parsed: list[ParsedDocument] = []
    skipped = 0

    for path in sorted(p for p in root.glob("**/*" if recursive else "*") if p.is_file()):
        # Hidden files such as .gitkeep are housekeeping, not documents, so
        # passing over them quietly keeps the warnings meaningful.
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            log.warning("Skipping '%s': unsupported file type", path.name)
            skipped += 1
            continue
        try:
            parsed.append(parse(path, settings))
        except ParseError as error:
            log.warning("Skipping '%s': %s", path.name, error)
            skipped += 1

    log.info("Parsed %d document(s) from %s, skipped %d", len(parsed), root, skipped)
    return parsed


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------

# Building a converter loads models, so reuse it. OCR changes the pipeline, so
# each setting gets its own.
_converter_cache: dict[tuple[bool, tuple[str, ...]], DocumentConverter] = {}


def _converter(settings: Settings) -> DocumentConverter:
    """Return a docling converter configured from the settings."""
    key = (settings.enable_ocr, tuple(settings.ocr_languages))
    if key in _converter_cache:
        return _converter_cache[key]

    options = PdfPipelineOptions()
    options.do_ocr = settings.enable_ocr
    if settings.enable_ocr:
        options.ocr_options.lang = list(settings.ocr_languages)

    # PDFs and images share docling's page-based pipeline, so both need the
    # OCR setting applied.
    page_formats = PdfFormatOption(pipeline_options=options)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: page_formats,
            InputFormat.IMAGE: page_formats,
        }
    )

    _converter_cache[key] = converter
    return converter


def _blocks_of(document: DoclingDocument) -> list[Block]:
    """Flatten a parsed document into text pieces that remember their origin."""
    blocks: list[Block] = []
    heading: str | None = None

    for item, _level in document.iterate_items():
        text = (getattr(item, "text", "") or "").strip()

        # Tables carry no plain text, so render them instead of losing them.
        if not text and hasattr(item, "export_to_markdown"):
            try:
                text = item.export_to_markdown(document).strip()
            except Exception:  # noqa: BLE001 - a table we cannot render is skipped
                text = ""

        if not text:
            continue

        if str(getattr(item, "label", "")) in _HEADING_LABELS:
            heading = text

        pages = [p.page_no for p in getattr(item, "prov", [])]
        blocks.append(Block(text=text, page=pages[0] if pages else None, heading=heading))

    return blocks


def _is_url(source: str | Path) -> bool:
    return str(source).lower().startswith(("http://", "https://"))


def _supported_list() -> str:
    return ", ".join(sorted(SUPPORTED_SUFFIXES))
