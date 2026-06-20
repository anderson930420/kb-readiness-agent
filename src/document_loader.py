"""Structured document loaders for Change Impact Mode."""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import fitz

from .ingest import HEADING_RE, SLUG_IN_TITLE_RE, _chinese_title, _slug


SUPPORTED_EXTENSIONS = (".md", ".markdown", ".txt", ".pdf")
TEXT_CHUNK_CHAR_LIMIT = 4_000
_SPACE_RE = re.compile(r"\s+")
_PAGE_NUMBER_RE = re.compile(r"^(?:page\s+)?\d+(?:\s+(?:of|/)\s*\d+)?$", re.IGNORECASE)


class UnsupportedDocumentTypeError(ValueError):
    """Raised when Change Impact receives an unsupported file extension."""


class DocumentParseError(ValueError):
    """Raised when a supported document cannot be parsed into comparison units."""


def detect_document_type(path: Path | str) -> str:
    """Return the normalized loader type for a supported document name or path."""

    document_path = Path(path)
    extension = document_path.suffix.casefold()
    if extension in {".md", ".markdown"}:
        return "markdown"
    if extension == ".txt":
        return "text"
    if extension == ".pdf":
        return "pdf"
    supported = ", ".join(SUPPORTED_EXTENSIONS)
    shown_extension = extension or "<none>"
    raise UnsupportedDocumentTypeError(
        f"Unsupported document extension '{shown_extension}' for '{document_path.name}'. "
        f"Supported extensions: {supported}."
    )


def _available_slug(heading: str) -> str | None:
    explicit = SLUG_IN_TITLE_RE.search(heading)
    if explicit:
        return explicit.group(1).lower()
    # A generated ASCII slug is useful for an English heading. Generating one
    # from a mixed/Chinese heading drops meaningful characters and can create
    # false exact matches, so those headings use similarity alignment instead.
    if re.search(r"[\u3400-\u9fff]", heading):
        return None
    return _slug(heading)


def _section(
    document_path: Path,
    heading: str,
    heading_level: int,
    text: str,
    *,
    page: int | None,
    page_end: int | None,
    source_type: str,
    analysis_unit: str,
    chunk_index: int,
) -> dict:
    slug = _available_slug(heading)
    page_token = f"p{page}" if page is not None else "section"
    return {
        "doc": document_path.name,
        "section": heading,
        "section_zh": _chinese_title(heading),
        "section_slug": slug,
        "heading_level": heading_level,
        "page": page,
        "page_end": page_end,
        "text": text,
        "source_type": source_type,
        "analysis_unit": analysis_unit,
        "chunk_id": f"{document_path.stem}::{page_token}::{slug or 'section'}::{chunk_index:04d}",
    }


def _plain_text_sections(document_path: Path, content: str, source_type: str) -> list[dict]:
    """Split unstructured text into bounded paragraph-based comparison chunks."""

    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", content) if item.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    def flush() -> None:
        nonlocal current, current_size
        if current:
            chunks.append("\n\n".join(current))
        current = []
        current_size = 0

    for paragraph in paragraphs:
        remaining = paragraph
        while len(remaining) > TEXT_CHUNK_CHAR_LIMIT:
            flush()
            split_at = remaining.rfind(" ", 0, TEXT_CHUNK_CHAR_LIMIT)
            if split_at <= 0:
                split_at = TEXT_CHUNK_CHAR_LIMIT
            chunks.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        added_size = len(remaining) + (2 if current else 0)
        if current and current_size + added_size > TEXT_CHUNK_CHAR_LIMIT:
            flush()
        if remaining:
            current.append(remaining)
            current_size += added_size
    flush()

    return [
        _section(
            document_path,
            f"Text chunk {index:03d} (text_chunk_{index:03d})",
            2,
            text,
            page=None,
            page_end=None,
            source_type=source_type,
            analysis_unit="text_chunk",
            chunk_index=index,
        )
        for index, text in enumerate(chunks, start=1)
    ]


def parse_markdown_sections(path: Path | str) -> list[dict]:
    """Parse Markdown H1/H2 bodies into normalized policy sections."""

    document_path = Path(path)
    try:
        markdown = document_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DocumentParseError(
            f"Unable to read text document '{document_path}': {exc}"
        ) from exc
    sections: list[dict] = []
    heading_text: str | None = None
    heading_level: int | None = None
    body: list[str] = []

    def flush() -> None:
        if heading_text is None or heading_level is None:
            return
        text = "\n".join(body).strip()
        if not text:
            return
        sections.append(
            _section(
                document_path,
                heading_text,
                heading_level,
                text,
                page=None,
                page_end=None,
                source_type="markdown",
                analysis_unit="section",
                chunk_index=len(sections) + 1,
            )
        )

    for line in markdown.splitlines():
        heading = HEADING_RE.match(line)
        if heading and len(heading.group(1)) <= 2:
            flush()
            heading_text = heading.group(2).strip()
            heading_level = len(heading.group(1))
            body = []
        elif heading_text is not None:
            body.append(line.rstrip())
    flush()
    # Headerless Markdown remains usable, but is bounded like plain text rather
    # than becoming one unbounded comparison unit.
    return sections or _plain_text_sections(document_path, markdown, "markdown")


def parse_text_sections(path: Path | str) -> list[dict]:
    """Parse UTF-8 plain text into bounded paragraph-based comparison chunks."""

    document_path = Path(path)
    try:
        content = document_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DocumentParseError(
            f"Unable to read text document '{document_path}': {exc}"
        ) from exc
    sections = _plain_text_sections(document_path, content, "text")
    if not sections:
        raise DocumentParseError(f"Text document is empty: '{document_path}'")
    return sections


def _pdf_page_lines(page: fitz.Page, page_number: int) -> list[dict]:
    """Extract normalized layout lines from exactly one PDF page."""

    lines: list[dict] = []
    page_height = float(page.rect.height)
    payload = page.get_text("dict", sort=True)
    for block in payload.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [
                span
                for span in line.get("spans", [])
                if span.get("text", "").strip()
            ]
            if not spans:
                continue
            text = _SPACE_RE.sub(
                " ", " ".join(span["text"].strip() for span in spans)
            ).strip()
            if not text:
                continue
            bbox = line.get("bbox") or spans[0].get("bbox")
            lines.append(
                {
                    "page": page_number,
                    "page_height": page_height,
                    "text": text,
                    "normalized": text.casefold(),
                    "size": max(float(span.get("size", 0.0)) for span in spans),
                    "bold": any(
                        "bold" in str(span.get("font", "")).casefold()
                        for span in spans
                    ),
                    "y0": float(bbox[1]),
                    "y1": float(bbox[3]),
                }
            )
    return lines


def _in_page_margin(line: dict) -> bool:
    return line["y0"] <= 60 or line["y1"] >= line["page_height"] - 50


def _repeated_marginal_text(
    pages_by_text: dict[str, set[int]], page_count: int
) -> set[str]:
    repeat_threshold = max(2, (page_count + 1) // 2)
    return {
        text
        for text, pages in pages_by_text.items()
        if len(pages) >= repeat_threshold
    }


def _body_font_size(weighted_sizes: list[float]) -> float:
    return statistics.median(weighted_sizes) if weighted_sizes else 10.0


def _is_pdf_heading(line: dict, body_size: float) -> bool:
    explicit_slug = SLUG_IN_TITLE_RE.search(line["text"])
    larger_text = line["size"] >= body_size + 1.5
    bold_text = line["bold"] and line["size"] >= body_size + 0.5
    return bool(explicit_slug or larger_text or bold_text)


def parse_pdf_sections(path: Path | str) -> list[dict]:
    """Parse a PDF page by page into page-bounded, layout-derived sections."""

    document_path = Path(path)
    try:
        document = fitz.open(document_path)
    except (fitz.FileDataError, RuntimeError, OSError) as exc:
        raise DocumentParseError(
            f"Unable to read PDF document '{document_path}': {exc}"
        ) from exc

    try:
        if document.needs_pass:
            raise DocumentParseError(
                f"PDF document requires a password: '{document_path}'"
            )

        # First page-wise pass records only repeated-margin fingerprints and font
        # sizes. Document text is never concatenated into one full-PDF context.
        pages_by_marginal_text: dict[str, set[int]] = {}
        weighted_sizes: list[float] = []
        for page_index, page in enumerate(document):
            for line in _pdf_page_lines(page, page_index + 1):
                if _in_page_margin(line):
                    pages_by_marginal_text.setdefault(
                        line["normalized"], set()
                    ).add(line["page"])
                else:
                    weighted_sizes.extend(
                        [line["size"]] * max(1, min(len(line["text"]), 80))
                    )

        repeated_margins = _repeated_marginal_text(
            pages_by_marginal_text, document.page_count
        )
        body_size = _body_font_size(weighted_sizes)
        sections: list[dict] = []
        heading_text: str | None = None
        heading_level: int | None = None
        chunk_page: int | None = None
        body: list[str] = []

        def flush() -> None:
            nonlocal body
            if (
                heading_text is None
                or heading_level is None
                or chunk_page is None
                or not body
            ):
                body = []
                return
            text = "\n".join(body).strip()
            if text:
                sections.append(
                    _section(
                        document_path,
                        heading_text,
                        heading_level,
                        text,
                        page=chunk_page,
                        page_end=chunk_page,
                        source_type="pdf",
                        analysis_unit="page_section_chunk",
                        chunk_index=len(sections) + 1,
                    )
                )
            body = []

        # The second pass creates page-bounded analysis units. A section may
        # continue on later pages, but no unit can contain the complete PDF.
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            chunk_page = page_number
            content_lines = [
                line
                for line in _pdf_page_lines(page, page_number)
                if line["normalized"] not in repeated_margins
                and not (
                    _in_page_margin(line) and _PAGE_NUMBER_RE.match(line["text"])
                )
            ]
            for line in content_lines:
                if _is_pdf_heading(line, body_size):
                    flush()
                    heading_text = line["text"]
                    heading_level = 1 if line["size"] >= body_size + 6 else 2
                elif heading_text is not None:
                    body.append(line["text"])
            flush()

        if not sections:
            raise DocumentParseError(
                f"No structured sections found in PDF document '{document_path}'. "
                "The PDF must contain extractable text and visually distinct section headings."
            )
        return sections
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(
            f"Unable to parse PDF document '{document_path}' page by page: {exc}"
        ) from exc
    finally:
        document.close()


def load_document(path: Path | str) -> list[dict]:
    """Load one supported document into normalized comparison sections/chunks."""

    document_path = Path(path)
    document_type = detect_document_type(document_path)
    if document_type == "markdown":
        return parse_markdown_sections(document_path)
    if document_type == "text":
        return parse_text_sections(document_path)
    return parse_pdf_sections(document_path)


# Short aliases keep the public API convenient for callers and tests.
parse_markdown = parse_markdown_sections
parse_pdf = parse_pdf_sections
parse_document = load_document
