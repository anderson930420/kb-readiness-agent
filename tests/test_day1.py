from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.answer import answer_question
from src.ingest import DEFAULT_CORPUS_DIR, ingest


class Day1RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.index_path = Path(cls._temporary_directory.name) / "chunks.jsonl"
        cls.chunk_count = ingest(DEFAULT_CORPUS_DIR, cls.index_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def assert_cites(self, question: str, expected_doc: str, *, refused: bool) -> None:
        result = answer_question(question, index_path=self.index_path)
        self.assertEqual(result.refused, refused, result.text)
        self.assertIn(expected_doc, {item["doc"] for item in result.citations})

    def test_index_is_corpus_only_and_schema_is_normalized(self) -> None:
        rows = [
            json.loads(line)
            for line in self.index_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(self.chunk_count, 34)
        self.assertEqual(len({row["chunk_id"] for row in rows}), len(rows))
        self.assertTrue(all("/" not in row["doc"] for row in rows))
        self.assertTrue(all(row["page"] is None for row in rows))
        self.assertTrue(all(row["text"] == row["content"] for row in rows))
        self.assertFalse(any("old_refund_policy" in row["doc"] for row in rows))
        self.assertFalse(any("new_refund_policy" in row["doc"] for row in rows))

    def test_english_refund_window(self) -> None:
        self.assert_cites(
            "What is the refund window for standard monthly subscribers?",
            "refund_policy.md",
            refused=False,
        )

    def test_chinese_refund_window(self) -> None:
        self.assert_cites(
            "標準月付用戶的退款期限是多久？",
            "refund_policy.md",
            refused=False,
        )

    def test_medical_records_uses_privacy_policy(self) -> None:
        result = answer_question(
            "客戶可以把醫療紀錄上傳到客服工單嗎？",
            index_path=self.index_path,
        )
        self.assertFalse(result.refused, result.text)
        self.assertEqual(result.citations[0]["doc"], "privacy_policy.md")
        self.assertNotIn("refund_policy.md", {item["doc"] for item in result.citations})

    def test_medical_refund_exception_refuses(self) -> None:
        self.assert_cites(
            "Can customers get a refund after 90 days for medical reasons?",
            "refund_policy.md",
            refused=True,
        )

    def test_exact_enterprise_sla_refuses(self) -> None:
        self.assert_cites(
            "Enterprise 方案的精確系統可用率 SLA 是多少？",
            "enterprise_plan_faq.md",
            refused=True,
        )

    def test_enterprise_price_refuses_and_routes_to_sales(self) -> None:
        result = answer_question(
            "Enterprise 方案每月價格是多少台幣？",
            index_path=self.index_path,
        )
        self.assertTrue(result.refused, result.text)
        self.assertIn("業務團隊", result.text)
        self.assertIn(
            result.citations[0]["doc"],
            {"pricing_policy.md", "enterprise_plan_faq.md"},
        )

    def test_eval_categories(self) -> None:
        eval_path = Path(__file__).resolve().parents[1] / "eval" / "eval_set.jsonl"
        rows = [
            json.loads(line)
            for line in eval_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), 25)
        self.assertEqual(
            Counter(row["category"] for row in rows),
            {"answerable": 15, "unanswerable": 5, "conflict": 5},
        )


if __name__ == "__main__":
    unittest.main()
