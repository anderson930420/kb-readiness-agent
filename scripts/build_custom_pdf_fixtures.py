"""Build synthetic support-contract PDFs for custom-upload Change Impact testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz


DEFAULT_OLD_PATH = Path("compare_docs/custom_old_support_contract.pdf")
DEFAULT_NEW_PATH = Path("compare_docs/custom_new_support_contract.pdf")
DEFAULT_PAGES = 50
MINIMUM_PAGES = 40
MAXIMUM_PAGES = 60
PAGE_SIZE = fitz.paper_rect("a4")

OLD_CONTRACT_SECTIONS = {
    3: (
        "Support SLA (enterprise_support_response_time)",
        "The contracted initial response time for Priority 1 support incidents is 24 hours. "
        "The response clock begins when the support ticket is accepted by the service desk.",
    ),
    11: (
        "Refund Exceptions (unsupported_exceptions)",
        "Customers may request a refund exception for a verified service outage that prevented "
        "use for more than 72 hours. The exception must be documented by the support lead.",
    ),
    19: (
        "Enterprise Manual Review (enterprise_refunds)",
        "Enterprise customers inside the contractual refund window may receive an automatic "
        "refund after support confirms account ownership.",
    ),
    27: (
        "Data Deletion (data_deletion)",
        "Standard account data deletion requests are completed within 30 days after identity "
        "verification succeeds.",
    ),
    35: (
        "Escalation Rules (refund_escalation)",
        "Tier 1 support records the decision and the supporting evidence for disputed refund "
        "requests before closing the ticket.",
    ),
    43: (
        "Onboarding Fees (digital_services)",
        "Completed onboarding workshops, implementation services, and custom data migration "
        "fees are non-refundable after delivery.",
    ),
}

NEW_CONTRACT_SECTIONS = {
    3: (
        "Support SLA (enterprise_support_response_time)",
        "The contracted initial response time for Priority 1 support incidents is 4 hours. "
        "The response clock begins when the support ticket is accepted by the service desk.",
    ),
    11: (
        "Refund Exceptions (unsupported_exceptions)",
        "Refund requests caused by verified service outages are non-refundable. Support must "
        "not approve a refund solely because an outage prevented use of the service.",
    ),
    19: (
        "Enterprise Manual Review (enterprise_refunds)",
        "Enterprise refund requests must be sent to account management for manual review. "
        "Support must not promise or issue an automatic refund.",
    ),
    27: (
        "Data Deletion (data_deletion)",
        "Standard account data deletion requests are completed within 14 days after identity "
        "verification succeeds.",
    ),
    35: (
        "Escalation Rules (refund_escalation)",
        "Any disputed refund request involving a service outage must be escalated to the support "
        "director for manual review before the ticket can be closed.",
    ),
    43: (
        "Onboarding Fees (digital_services)",
        "Completed onboarding workshops, implementation services, and custom data migration "
        "fees are non-refundable after delivery.",
    ),
}

BOILERPLATE_TEXT = (
    "This synthetic contract reference page contains stable operational language used in both "
    "versions. It verifies page-wise extraction, repeated-margin cleanup, and section alignment "
    "without treating the full PDF as one comparison context."
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
        raise RuntimeError(
            f"Fixture text did not fit on page {page.number + 1}: {text[:60]}"
        )


def _draw_page(
    page: fitz.Page,
    *,
    page_number: int,
    page_count: int,
    version: str,
    section: tuple[str, str] | None,
) -> None:
    # The identical header/footer strings intentionally exercise repeated-margin
    # removal. Page numbers remain visible in a separate footer text span.
    page.insert_text(
        (54, 38),
        "Northstar Cloud - Enterprise Support Contract",
        fontsize=8,
        fontname="helv",
        color=(0.35, 0.35, 0.35),
    )
    page.insert_text(
        (54, PAGE_SIZE.height - 28),
        "Confidential - Synthetic Upload Fixture",
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
            fitz.Rect(54, 105, PAGE_SIZE.width - 54, 190),
            f"{version.title()} Enterprise Support Contract",
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
        fitz.Rect(54, 78, PAGE_SIZE.width - 54, 132),
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
    contract_sections = (
        OLD_CONTRACT_SECTIONS if version == "old" else NEW_CONTRACT_SECTIONS
    )
    document = fitz.open()
    try:
        document.set_metadata(
            {
                "title": f"Synthetic {version.title()} Enterprise Support Contract",
                "author": "kb-readiness-agent",
                "subject": "Custom-upload Change Impact fixture",
                "creator": "scripts.build_custom_pdf_fixtures",
                "producer": "PyMuPDF",
                "creationDate": "D:20260101000000+00'00'",
                "modDate": "D:20260101000000+00'00'",
            }
        )
        for page_number in range(1, page_count + 1):
            page = document.new_page(width=PAGE_SIZE.width, height=PAGE_SIZE.height)
            section = contract_sections.get(page_number)
            if section is None and page_number != 1:
                section = (
                    f"Contract Reference {page_number:03d} "
                    f"(contract_reference_{page_number:03d})",
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


def build_custom_pdf_fixtures(
    old_path: Path, new_path: Path, pages: int = DEFAULT_PAGES
) -> None:
    """Write old/new synthetic support contracts with stable page structure."""

    if not MINIMUM_PAGES <= pages <= MAXIMUM_PAGES:
        raise ValueError(
            f"Custom PDF fixtures require {MINIMUM_PAGES} to {MAXIMUM_PAGES} pages"
        )
    if old_path.resolve() == new_path.resolve():
        raise ValueError("Old and new fixture paths must be different")
    _build_pdf(old_path, page_count=pages, version="old")
    _build_pdf(new_path, page_count=pages, version="new")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD_PATH)
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW_PATH)
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES)
    args = parser.parse_args()
    try:
        build_custom_pdf_fixtures(args.old, args.new, args.pages)
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"Generated {args.pages}-page custom-upload PDFs: {args.old} and {args.new}"
    )


if __name__ == "__main__":
    main()
