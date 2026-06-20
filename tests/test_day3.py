from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.answer import AnswerResult, answer_from_retrieved, answer_question, main
from src.ingest import DEFAULT_CORPUS_DIR, ingest


class Day3ReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.index_path = Path(cls._temporary_directory.name) / "chunks.jsonl"
        ingest(DEFAULT_CORPUS_DIR, cls.index_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_answer_result_schema_and_grounded_normal_answer(self) -> None:
        result = answer_question(
            "標準月付用戶的退款期限是多久？",
            index_path=self.index_path,
        )

        self.assertIsInstance(result, AnswerResult)
        self.assertEqual(
            set(result.to_dict()),
            {
                "question",
                "retriever",
                "answer",
                "refused",
                "refusal_reason",
                "requires_human_review",
                "confidence",
                "citations",
                "retrieved_chunks",
                "groundedness",
                "warnings",
                "latency_ms",
                "answer_mode",
                "validator_decision",
                "generation_trace",
                "blocked_generated_answer",
                "response_type",
            },
        )
        self.assertEqual(result.response_type, "kb_answer")
        self.assertEqual(result.answer_mode, "extractive")
        self.assertEqual(result.validator_decision, "not_run")
        self.assertIsNone(result.generation_trace)
        self.assertIsNone(result.blocked_generated_answer)
        self.assertFalse(result.refused)
        self.assertEqual(result.confidence, "high")
        self.assertEqual(result.groundedness["status"], "supported")
        self.assertTrue(result.groundedness["checks"]["numeric_claims_supported"])
        self.assertTrue(
            {
                "chunk_id",
                "doc",
                "section",
                "section_zh",
                "section_slug",
                "page",
                "text",
            }.issubset(result.citations[0])
        )

    def test_json_cli_prints_full_answer_result(self) -> None:
        output = io.StringIO()
        argv = [
            "src.answer",
            "標準月付用戶的退款期限是多久？",
            "--retriever",
            "lexical",
            "--index",
            str(self.index_path),
            "--json",
        ]
        with patch.object(sys, "argv", argv), redirect_stdout(output):
            main()

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["question"], argv[1])
        self.assertEqual(payload["retriever"], "lexical")
        self.assertEqual(payload["groundedness"]["status"], "supported")
        self.assertEqual(payload["citations"][0]["doc"], "refund_policy.md")
        self.assertTrue(payload["retrieved_chunks"])

    def test_evidence_based_refusal_is_grounded(self) -> None:
        result = answer_question(
            "Can customers get a refund after 90 days for medical reasons?",
            index_path=self.index_path,
        )

        self.assertTrue(result.refused)
        self.assertEqual(result.refusal_reason, "refund_exception")
        self.assertTrue(result.requires_human_review)
        self.assertEqual(result.confidence, "medium")
        self.assertEqual(result.groundedness["status"], "supported")
        self.assertEqual(result.citations[0]["doc"], "refund_policy.md")
        self.assertEqual(
            result.citations[0]["section_slug"], "unsupported_exceptions"
        )

    def test_citations_are_members_of_retrieved_chunks(self) -> None:
        result = answer_question(
            "已核准的退款需要多久才會完成處理？",
            index_path=self.index_path,
        )
        retrieved_ids = {chunk["chunk_id"] for chunk in result.retrieved_chunks}

        self.assertTrue(result.citations)
        self.assertTrue(
            all(citation["chunk_id"] in retrieved_ids for citation in result.citations)
        )
        self.assertTrue(
            result.groundedness["checks"]["citations_from_retrieved_chunks"]
        )

    def test_q013_cites_privacy_policy(self) -> None:
        result = answer_question(
            "客戶是否應該把醫療紀錄上傳到客服工單？",
            index_path=self.index_path,
        )

        self.assertFalse(result.refused)
        self.assertEqual(result.citations[0]["doc"], "privacy_policy.md")
        self.assertEqual(result.citations[0]["section_slug"], "sensitive_data")

    def test_query_scoped_conflicting_values_require_review(self) -> None:
        def result(chunk_id: str, doc: str, days: int, score: float) -> dict:
            return {
                "chunk_id": chunk_id,
                "doc": doc,
                "section": "Standard refund window",
                "section_zh": "標準退款期限",
                "section_slug": "standard_refund_window",
                "page": None,
                "score": score,
                "retrieval_method": "lexical",
                "text": f"Standard monthly subscribers may request a refund within {days} days.",
            }

        retrieved = [
            result("current", "current.md", 7, 10.0),
            result("duplicate", "duplicate.md", 14, 9.0),
        ]
        answer = answer_from_retrieved(
            "What is the refund window for standard monthly subscribers?",
            retrieved,
            retriever="lexical",
        )

        self.assertTrue(answer.requires_human_review)
        self.assertIn("Potential conflicting evidence detected", answer.warnings)


if __name__ == "__main__":
    unittest.main()
