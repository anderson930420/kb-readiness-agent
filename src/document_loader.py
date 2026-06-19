"""Structured document loaders for Change Impact Mode."""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import fitz

from .ingest import HEADING_RE, SLUG_IN_TITLE_RE, _chinese_title, _slug


SUPPORTED_EXTENSIONS = (".md", ".markdown", ".pdf")
_SPACE_RE = re.compile(r"\s+")
_PAGE_NUMBER_RE = re.compile(r"^(?:page\s+)?\d+(?:\s+(?:of|/)\s*\d+)?$", re.IGNORECASE)


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
) -> dict:
    return {
        "doc": document_path.name,
        "section": heading,
        "section_zh": _chinese_title(heading),
        "section_slug": _available_slug(heading),
        "heading_level": heading_level,
        "page": page,
        "page_end": page_end,
        "text": text,
    }


def parse_markdown_sections(path: Path | str) -> list[dict]:
    """Parse Markdown H1/H2 bodies into normalized policy sections."""

    document_path = Path(path)
    markdown = document_path.read_text(encoding="utf-8")
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
    return sections


def _pdf_lines(document: fitz.Document) -> list[dict]:
    lines: list[dict] = []
    for page_index, page in enumerate(document):
        page_number = page_index + 1
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


def _repeated_marginal_text(lines: list[dict], page_count: int) -> set[str]:
    pages_by_text: dict[str, set[int]] = {}
    for line in lines:
        in_margin = line["y0"] <= 60 or line["y1"] >= line["page_height"] - 50
        if not in_margin:
            continue
        pages_by_text.setdefault(line["normalized"], set()).add(line["page"])
    repeat_threshold = max(2, (page_count + 1) // 2)
    return {
        text
        for text, pages in pages_by_text.items()
        if len(pages) >= repeat_threshold
    }


def _body_font_size(lines: list[dict]) -> float:
    weighted_sizes: list[float] = []
    for line in lines:
        # Cap the weighting so one long paragraph cannot dominate the document.
        weighted_sizes.extend([line["size"]] * max(1, min(len(line["text"]), 80)))
    return statistics.median(weighted_sizes) if weighted_sizes else 10.0


def _is_pdf_heading(line: dict, body_size: float) -> bool:
    explicit_slug = SLUG_IN_TITLE_RE.search(line["text"])
    larger_text = line["size"] >= body_size + 1.5
    bold_text = line["bold"] and line["size"] >= body_size + 0.5
    return bool(explicit_slug or larger_text or bold_text)


def parse_pdf_sections(path: Path | str) -> list[dict]:
    """Parse a PDF into layout-derived sections with 1-based page metadata."""

    document_path = Path(path)
    try:
        document = fitz.open(document_path)
    except (fitz.FileDataError, RuntimeError) as exc:
        raise ValueError(f"Unable to read PDF document '{document_path}': {exc}") from exc

    try:
        if document.needs_pass:
            raise ValueError(f"PDF document requires a password: '{document_path}'")
        lines = _pdf_lines(document)
        repeated_margins = _repeated_marginal_text(lines, document.page_count)
        content_lines = [
            line
            for line in lines
            if line["normalized"] not in repeated_margins
            and not (
                (line["y0"] <= 60 or line["y1"] >= line["page_height"] - 50)
                and _PAGE_NUMBER_RE.match(line["text"])
            )
        ]
        body_size = _body_font_size(content_lines)

        sections: list[dict] = []
        heading_text: str | None = None
        heading_level: int | None = None
        heading_page: int | None = None
        body_end_page: int | None = None
        body: list[str] = []

        def flush() -> None:
            if heading_text is None or heading_level is None or heading_page is None:
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
                    page=heading_page,
                    page_end=body_end_page or heading_page,
                )
            )

        for line in content_lines:
            if _is_pdf_heading(line, body_size):
                flush()
                heading_text = line["text"]
                heading_level = 1 if line["size"] >= body_size + 6 else 2
                heading_page = line["page"]
                body_end_page = None
                body = []
            elif heading_text is not None:
                body.append(line["text"])
                body_end_page = line["page"]
        flush()
        if not sections:
            raise ValueError(
                f"No structured sections found in PDF document '{document_path}'. "
                "The PDF must contain extractable text and visually distinct section headings."
            )
        return sections
    finally:
        document.close()


def load_document(path: Path | str) -> list[dict]:
    """Load one supported document into normalized policy sections."""

    document_path = Path(path)
    extension = document_path.suffix.casefold()
    if extension in {".md", ".markdown"}:
        return parse_markdown_sections(document_path)
    if extension == ".pdf":
        return parse_pdf_sections(document_path)
    supported = ", ".join(SUPPORTED_EXTENSIONS)
    shown_extension = extension or "<none>"
    raise ValueError(
        f"Unsupported document extension '{shown_extension}' for '{document_path}'. "
        f"Supported extensions: {supported}."
    )


# Short aliases keep the public API convenient for callers and tests.
parse_markdown = parse_markdown_sections
parse_pdf = parse_pdf_sections
parse_document = load_document
