from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fitz

from scripts.build_large_pdf_fixture import build_large_pdf_fixtures
from src.compare import compare_documents
from src.document_loader import load_document, parse_pdf_sections
from src.ingest import PROJECT_ROOT


OLD_MARKDOWN = PROJECT_ROOT / "compare_docs" / "old_refund_policy.md"
NEW_MARKDOWN = PROJECT_ROOT / "compare_docs" / "new_refund_policy.md"


class PdfChangeImpactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        root = Path(cls.temp_dir.name)
        cls.old_pdf = root / "large_old_refund_policy.pdf"
        cls.new_pdf = root / "large_new_refund_policy.pdf"
        build_large_pdf_fixtures(cls.old_pdf, cls.new_pdf, pages=50)
        cls.old_sections = parse_pdf_sections(cls.old_pdf)
        cls.new_sections = parse_pdf_sections(cls.new_pdf)
        cls.result = compare_documents(cls.old_pdf, cls.new_pdf)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_markdown_compare_baseline_is_unchanged(self) -> None:
        summary = compare_documents(OLD_MARKDOWN, NEW_MARKDOWN)["summary"]

        self.assertEqual(summary["old_sections"], 5)
        self.assertEqual(summary["new_sections"], 6)
        self.assertEqual(summary["changed_sections"], 6)
        self.assertEqual(summary["severity"], {"high": 4, "medium": 2, "low": 0})
        self.assertEqual(summary["impacted_eval_cases"], 13)
        self.assertEqual(summary["required_kb_updates"], 9)

    def test_pdf_loader_extracts_sections_and_page_numbers(self) -> None:
        with fitz.open(self.old_pdf) as document:
            self.assertEqual(document.page_count, 50)

        standard = next(
            section
            for section in self.old_sections
            if section["section_slug"] == "standard_refund_window"
        )
        added = next(
            section
            for section in self.new_sections
            if section["section_slug"] == "unsupported_exceptions"
        )

        self.assertEqual(
            standard["section"],
            "Standard Refund Window (standard_refund_window)",
        )
        self.assertEqual((standard["page"], standard["page_end"]), (3, 3))
        self.assertIn("14 days", standard["text"])
        self.assertEqual((added["page"], added["page_end"]), (2, 2))
        self.assertIn("medical reasons", added["text"])

    def test_large_pdf_compare_detects_intended_changed_sections(self) -> None:
        changes = {change["section_slug"]: change for change in self.result["changes"]}

        self.assertEqual(
            set(changes),
            {
                "standard_refund_window",
                "renewal_payments",
                "enterprise_refunds",
                "digital_services",
                "refund_processing_time",
                "unsupported_exceptions",
            },
        )
        self.assertEqual(self.result["summary"]["changed_sections"], 6)
        self.assertEqual(self.result["summary"]["severity"]["high"], 4)
        self.assertTrue(changes["standard_refund_window"]["refund_window_changed"])
        self.assertEqual(changes["standard_refund_window"]["old_page"], 3)
        self.assertEqual(changes["unsupported_exceptions"]["status"], "added")
        self.assertEqual(changes["unsupported_exceptions"]["new_page"], 2)

    def test_large_pdf_compare_maps_eval_cases_and_kb_updates(self) -> None:
        impacted = {item["case_id"] for item in self.result["impacted_eval_cases"]}
        updates = {
            (item["file"], item["section_slug"])
            for item in self.result["required_kb_updates"]
        }

        self.assertEqual(self.result["summary"]["impacted_eval_cases"], 13)
        self.assertIn("q001", impacted)
        self.assertIn("q021", impacted)
        self.assertEqual(self.result["summary"]["required_kb_updates"], 9)
        self.assertIn(("corpus/refund_policy.md", "standard_refund_window"), updates)
        self.assertIn(("corpus/support_escalation_sop.md", "refund_escalation"), updates)

    def test_unsupported_extension_fails_clearly(self) -> None:
        unsupported = Path(self.temp_dir.name) / "policy.txt"
        unsupported.write_text("Policy text", encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError,
            r"Unsupported document extension '\.txt'.*Supported extensions: \.md, \.markdown, \.pdf",
        ):
            load_document(unsupported)


if __name__ == "__main__":
    unittest.main()
