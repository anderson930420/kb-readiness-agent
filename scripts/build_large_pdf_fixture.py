"""Build deterministic, synthetic large-PDF Change Impact fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz


DEFAULT_OLD_PATH = Path("compare_docs/large_old_refund_policy.pdf")
DEFAULT_NEW_PATH = Path("compare_docs/large_new_refund_policy.pdf")
MINIMUM_PAGES = 50
PAGE_SIZE = fitz.paper_rect("a4")

OLD_POLICY_SECTIONS = {
    3: (
        "Standard Refund Window (standard_refund_window)",
        "Standard monthly subscribers may request a refund within 14 days of the initial purchase. "
        "Annual subscribers may request a refund within 30 days of the initial purchase.",
    ),
    11: (
        "Renewal Payments (renewal_payments)",
        "Customers may request a refund for renewal payments within 7 days of payment, "
        "limited to one refund per customer each year.",
    ),
    19: (
        "Enterprise Customer Refunds (enterprise_refunds)",
        "Enterprise customers within the refund window may request an automatic refund "
        "submitted by customer support.",
    ),
    27: (
        "Digital Service Fees (digital_services)",
        "Onboarding services and migration services are refundable when customer support "
        "approves the individual request.",
    ),
    35: (
        "Refund Processing Time (refund_processing_time)",
        "Approved refunds are processed within 10 to 15 business days.",
    ),
}

NEW_POLICY_SECTIONS = {
    2: (
        "Unsupported Exception Scenarios (unsupported_exceptions)",
        "This policy does not define refund exceptions for medical reasons, emergencies, "
        "or hardship scenarios.",
    ),
    3: (
        "Standard Refund Window (standard_refund_window)",
        "Standard monthly subscribers may request a refund within 7 days of the initial purchase. "
        "Annual subscribers may request a refund within 14 days of the initial purchase.",
    ),
    11: (
        "Renewal Payments (renewal_payments)",
        "Renewal payments are non-refundable unless applicable law requires otherwise.",
    ),
    19: (
        "Enterprise Customer Refunds (enterprise_refunds)",
        "Enterprise customer refund requests must be sent to account management for manual review. "
        "Customer support must not promise an automatic refund.",
    ),
    27: (
        "Digital Service Fees (digital_services)",
        "Delivered onboarding services, custom migration services, and professional services "
        "are non-refundable.",
    ),
    35: (
        "Refund Processing Time (refund_processing_time)",
        "Approved refunds are processed within 5 to 10 business days.",
    ),
}

BOILERPLATE_TEXT = (
    "This synthetic operational reference page intentionally repeats stable boilerplate. "
    "It verifies that large documents are aligned and compared section by section without "
    "placing the complete document into one context."
)


def _insert_textbox(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    *,
    fontsize: float,
    fontname: str,
    color: tuple[float, float, float],
) -> None:
    remaining = page.insert_textbox(
        rect,
        text,
        fontsize=fontsize,
        fontname=fontname,
        color=color,
        lineheight=1.35,
    )
    if remaining < 0:
        raise RuntimeError(f"Fixture text did not fit on page {page.number + 1}: {text[:60]}")


def _draw_page(
    page: fitz.Page,
    *,
    page_number: int,
    page_count: int,
    version: str,
    section: tuple[str, str] | None,
) -> None:
    page.insert_text(
        (54, 38),
        "KB Readiness Agent - Synthetic Refund Policy",
        fontsize=8,
        fontname="helv",
        color=(0.35, 0.35, 0.35),
    )
    page.insert_text(
        (PAGE_SIZE.width - 105, PAGE_SIZE.height - 28),
        f"Page {page_number} of {page_count}",
        fontsize=8,
        fontname="helv",
        color=(0.35, 0.35, 0.35),
    )

    if page_number == 1:
        _insert_textbox(
            page,
            fitz.Rect(54, 105, PAGE_SIZE.width - 54, 180),
            f"{version.title()} Refund Policy",
            fontsize=24,
            fontname="hebo",
            color=(0.08, 0.16, 0.28),
        )
        return
    if section is None:
        return

    title, body = section
    _insert_textbox(
        page,
        fitz.Rect(54, 78, PAGE_SIZE.width - 54, 130),
        title,
        fontsize=15,
        fontname="hebo",
        color=(0.08, 0.16, 0.28),
    )
    _insert_textbox(
        page,
        fitz.Rect(54, 145, PAGE_SIZE.width - 54, PAGE_SIZE.height - 70),
        body,
        fontsize=10.5,
        fontname="helv",
        color=(0.05, 0.05, 0.05),
    )


def _build_pdf(path: Path, *, page_count: int, version: str) -> None:
    policy_sections = OLD_POLICY_SECTIONS if version == "old" else NEW_POLICY_SECTIONS
    document = fitz.open()
    try:
        document.set_metadata(
            {
                "title": f"Synthetic {version.title()} Refund Policy",
                "author": "kb-readiness-agent",
                "subject": "Deterministic Change Impact fixture",
                "creator": "scripts.build_large_pdf_fixture",
                "producer": "PyMuPDF",
                "creationDate": "D:20260101000000+00'00'",
                "modDate": "D:20260101000000+00'00'",
            }
        )
        for page_number in range(1, page_count + 1):
            page = document.new_page(width=PAGE_SIZE.width, height=PAGE_SIZE.height)
            section = policy_sections.get(page_number)
            if section is None and page_number != 1 and not (
                version == "old" and page_number == 2
            ):
                section = (
                    f"Operational Reference {page_number:03d} "
                    f"(operational_reference_{page_number:03d})",
                    BOILERPLATE_TEXT,
                )
            _draw_page(
                page,
                page_number=page_number,
                page_count=page_count,
                version=version,
                section=section,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        document.save(path, garbage=4, clean=True, deflate=True, no_new_id=True)
    finally:
        document.close()


def build_large_pdf_fixtures(
    old_path: Path, new_path: Path, pages: int = MINIMUM_PAGES
) -> None:
    """Write old/new synthetic PDF policies with stable structure and content."""

    if pages < MINIMUM_PAGES:
        raise ValueError(f"Large PDF fixtures require at least {MINIMUM_PAGES} pages")
    if old_path.resolve() == new_path.resolve():
        raise ValueError("Old and new fixture paths must be different")
    _build_pdf(old_path, page_count=pages, version="old")
    _build_pdf(new_path, page_count=pages, version="new")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD_PATH)
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW_PATH)
    parser.add_argument("--pages", type=int, default=MINIMUM_PAGES)
    args = parser.parse_args()
    try:
        build_large_pdf_fixtures(args.old, args.new, args.pages)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Generated {args.pages}-page synthetic PDFs: {args.old} and {args.new}")


if __name__ == "__main__":
    main()
