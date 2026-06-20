from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.compare import (
    align_sections,
    compare_documents,
    parse_markdown_sections,
    write_reports,
)
from src.ingest import DEFAULT_CORPUS_DIR, PROJECT_ROOT, build_chunks


OLD_POLICY = PROJECT_ROOT / "compare_docs" / "old_refund_policy.md"
NEW_POLICY = PROJECT_ROOT / "compare_docs" / "new_refund_policy.md"


class Day5ChangeImpactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = compare_documents(OLD_POLICY, NEW_POLICY)

    def test_parser_extracts_h1_h2_policy_sections(self) -> None:
        old_sections = parse_markdown_sections(OLD_POLICY)

        self.assertEqual(len(old_sections), 5)
        self.assertEqual(old_sections[0]["doc"], "old_refund_policy.md")
        self.assertEqual(old_sections[0]["section_zh"], "標準退款期限")
        self.assertEqual(old_sections[0]["section_slug"], "standard_refund_window")
        self.assertIn("14 天", old_sections[0]["text"])

    def test_parser_preserves_h1_body_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "policy.md"
            path.write_text("# Policy\n\nIntro text.\n\n## Terms\n\nTerm text.\n", encoding="utf-8")

            sections = parse_markdown_sections(path)

        self.assertEqual([section["heading_level"] for section in sections], [1, 2])
        self.assertEqual(sections[0]["text"], "Intro text.")

    def test_sections_align_by_exact_slug(self) -> None:
        alignments = align_sections(
            parse_markdown_sections(OLD_POLICY), parse_markdown_sections(NEW_POLICY)
        )
        standard = next(
            item
            for item in alignments
            if item["old"]
            and item["old"]["section_slug"] == "standard_refund_window"
        )

        self.assertEqual(standard["alignment_method"], "exact_slug")
        self.assertEqual(standard["new"]["section_slug"], "standard_refund_window")

    def test_sections_align_by_normalized_heading(self) -> None:
        old = [{"section": "Refund Window", "section_zh": "Refund Window", "section_slug": "old-window", "text": "14 days"}]
        new = [{"section": "Refund window!", "section_zh": "Refund window!", "section_slug": "new-window", "text": "7 days"}]

        alignment = align_sections(old, new)[0]

        self.assertEqual(alignment["alignment_method"], "heading_similarity")
        self.assertEqual(alignment["status"], "changed")

    def test_chinese_heading_without_slug_does_not_create_false_exact_slug(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            old_path = Path(temp_dir) / "old.md"
            new_path = Path(temp_dir) / "new.md"
            old_path.write_text("## 退款資格\n\n舊規則。\n", encoding="utf-8")
            new_path.write_text("## 退款資格\n\n新規則。\n", encoding="utf-8")
            old_sections = parse_markdown_sections(old_path)
            new_sections = parse_markdown_sections(new_path)

        alignment = align_sections(old_sections, new_sections)[0]
        self.assertIsNone(old_sections[0]["section_slug"])
        self.assertEqual(alignment["alignment_method"], "heading_similarity")

    def test_refund_window_change_is_high_severity(self) -> None:
        change = next(
            item
            for item in self.result["changes"]
            if item["section_slug"] == "standard_refund_window"
        )

        self.assertTrue(change["refund_window_changed"])
        self.assertTrue(change["numeric_date_time_changed"])
        self.assertEqual(change["changed_values"]["old"], ["14 天", "30 天"])
        self.assertEqual(change["changed_values"]["new"], ["7 天", "14 天"])
        self.assertEqual(change["severity"], "high")

    def test_impacted_eval_questions_include_refund_cases(self) -> None:
        impacted = {item["case_id"]: item for item in self.result["impacted_eval_cases"]}

        self.assertIn("q001", impacted)
        self.assertIn("q002", impacted)
        self.assertIn("q021", impacted)
        self.assertEqual(impacted["q001"]["expected_section"], "standard_refund_window")
        self.assertEqual(impacted["q001"]["suggested_action"], "update_expected_answer")

    def test_manual_review_and_kb_updates_are_recommended(self) -> None:
        enterprise_change = next(
            item
            for item in self.result["changes"]
            if item["section_slug"] == "enterprise_refunds"
        )
        update_targets = {
            (item["file"], item["section_slug"])
            for item in self.result["required_kb_updates"]
        }

        self.assertTrue(enterprise_change["manual_review_requirement_added"])
        self.assertIn(("corpus/refund_policy.md", "standard_refund_window"), update_targets)
        self.assertIn(("corpus/support_escalation_sop.md", "refund_escalation"), update_targets)
        self.assertTrue(self.result["summary"]["human_review_required"])

    def test_report_files_are_generated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, report_path = write_reports(self.result, Path(temp_dir))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            report = report_path.read_text(encoding="utf-8")

        self.assertEqual(payload["mode"], "change_impact")
        self.assertIn("## High-risk changes", report)
        self.assertIn("## Impacted eval questions / existing answers", report)
        self.assertIn("does not claim full legal or semantic conflict detection", report)

    def test_default_ask_mode_still_excludes_compare_docs(self) -> None:
        chunks = build_chunks(DEFAULT_CORPUS_DIR)

        self.assertEqual(len(chunks), 34)
        self.assertFalse(any("old_refund_policy" in chunk["doc"] for chunk in chunks))
        self.assertFalse(any("new_refund_policy" in chunk["doc"] for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
