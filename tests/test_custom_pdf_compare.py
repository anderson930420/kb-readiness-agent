from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fitz

from scripts.build_custom_pdf_fixtures import build_custom_pdf_fixtures
from src.compare import compare_documents, render_markdown
from src.document_loader import parse_pdf_sections


EXPECTED_HEADINGS = {
    "enterprise_support_response_time": "Support SLA",
    "unsupported_exceptions": "Refund Exceptions",
    "enterprise_refunds": "Enterprise Manual Review",
    "data_deletion": "Data Deletion",
    "refund_escalation": "Escalation Rules",
    "digital_services": "Onboarding Fees",
}


class CustomSupportContractPdfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        root = Path(cls.temp_dir.name)
        cls.old_pdf = root / "custom_old_support_contract.pdf"
        cls.new_pdf = root / "custom_new_support_contract.pdf"
        build_custom_pdf_fixtures(cls.old_pdf, cls.new_pdf, pages=50)
        cls.old_sections = parse_pdf_sections(cls.old_pdf)
        cls.new_sections = parse_pdf_sections(cls.new_pdf)
        cls.result = compare_documents(cls.old_pdf, cls.new_pdf)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_fixture_has_50_pages_visible_numbers_and_repeated_margins(self) -> None:
        with fitz.open(self.old_pdf) as document:
            self.assertEqual(document.page_count, 50)
            page_text = document[2].get_text()

        self.assertIn("Northstar Cloud - Enterprise Support Contract", page_text)
        self.assertIn("Confidential - Synthetic Upload Fixture", page_text)
        self.assertIn("Page 3 of 50", page_text)

    def test_parser_preserves_required_headings_and_removes_margins(self) -> None:
        sections = {section["section_slug"]: section for section in self.old_sections}

        for slug, heading in EXPECTED_HEADINGS.items():
            self.assertIn(slug, sections)
            self.assertIn(heading, sections[slug]["section"])
            self.assertEqual(sections[slug]["page"], sections[slug]["page_end"])
        self.assertFalse(
            any(
                "Northstar Cloud - Enterprise Support Contract" in section["text"]
                or "Confidential - Synthetic Upload Fixture" in section["text"]
                or "Page 3 of 50" in section["text"]
                for section in self.old_sections
            )
        )

    def test_comparison_detects_five_high_risk_section_changes(self) -> None:
        changes = {change["section_slug"]: change for change in self.result["changes"]}

        self.assertEqual(
            set(changes),
            {
                "enterprise_support_response_time",
                "unsupported_exceptions",
                "enterprise_refunds",
                "data_deletion",
                "refund_escalation",
            },
        )
        self.assertEqual(self.result["summary"]["changed_sections"], 5)
        self.assertEqual(self.result["summary"]["severity"], {"high": 5, "medium": 0, "low": 0})
        self.assertTrue(changes["enterprise_support_response_time"]["sla_response_time_changed"])
        self.assertEqual(
            changes["enterprise_support_response_time"]["changed_values"],
            {"changed": True, "old": ["24 hours"], "new": ["4 hours"]},
        )
        self.assertTrue(changes["unsupported_exceptions"]["refund_exception_removed"])
        self.assertTrue(changes["enterprise_refunds"]["manual_review_requirement_added"])
        self.assertTrue(changes["data_deletion"]["data_deletion_timeline_changed"])
        self.assertTrue(changes["refund_escalation"]["escalation_rule_added"])
        self.assertNotIn("digital_services", changes)

    def test_report_requires_review_and_maps_eval_and_kb_updates(self) -> None:
        updates = {
            (item["file"], item["section_slug"])
            for item in self.result["required_kb_updates"]
        }
        impacted = {item["case_id"] for item in self.result["impacted_eval_cases"]}
        report = render_markdown(self.result)

        self.assertTrue(self.result["summary"]["human_review_required"])
        self.assertIn("q011", impacted)
        self.assertIn("q017", impacted)
        self.assertIn(("corpus/privacy_policy.md", "data_deletion"), updates)
        self.assertIn(
            ("corpus/enterprise_plan_faq.md", "enterprise_support_response_time"),
            updates,
        )
        self.assertIn(
            ("corpus/support_escalation_sop.md", "refund_escalation"), updates
        )
        self.assertIn("## High-risk changes", report)
        self.assertIn("not loaded as one context", report)


if __name__ == "__main__":
    unittest.main()
