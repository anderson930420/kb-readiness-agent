from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.answer import answer_question
from src.generation import GeneratedAnswer, build_generation_prompt
from src.ingest import DEFAULT_CORPUS_DIR, ingest
from src.retrieve import load_chunks


class GenerativeAnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.index_path = Path(cls._temporary_directory.name) / "chunks.jsonl"
        ingest(DEFAULT_CORPUS_DIR, cls.index_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_extractive_default_is_unchanged(self) -> None:
        with patch("src.answer.generate_answer") as generate:
            result = answer_question(
                "標準月付用戶的退款期限是多久？",
                index_path=self.index_path,
            )

        generate.assert_not_called()
        self.assertEqual(result.answer_mode, "extractive")
        self.assertEqual(result.validator_decision, "not_run")
        self.assertFalse(result.refused)
        self.assertEqual(result.answer, result.retrieved_chunks[0]["text"])

    def test_fake_supported_generation_is_allowed(self) -> None:
        result = answer_question(
            "標準月付用戶的退款期限是多久？",
            index_path=self.index_path,
            mode="generative",
            llm_provider="fake_supported",
        )

        self.assertEqual(result.answer_mode, "generative")
        self.assertEqual(result.validator_decision, "allowed")
        self.assertFalse(result.refused)
        self.assertEqual(result.confidence, "high")
        self.assertEqual(result.groundedness["status"], "supported")
        self.assertIsNone(result.blocked_generated_answer)
        self.assertEqual(result.generation_trace["provider"], "fake_supported")

    def test_fake_hallucinated_numeric_claim_is_blocked(self) -> None:
        result = answer_question(
            "客戶如果因為醫療因素，90 天後還可以退款嗎？",
            index_path=self.index_path,
            mode="generative",
            llm_provider="fake_hallucination",
        )

        self.assertEqual(result.validator_decision, "blocked")
        self.assertTrue(result.refused)
        self.assertNotEqual(result.answer, result.blocked_generated_answer)
        self.assertIn("90 天", result.blocked_generated_answer)
        self.assertTrue(result.requires_human_review)
        self.assertEqual(result.confidence, "low")
        generated_groundedness = result.generation_trace["generated_groundedness"]
        self.assertEqual(generated_groundedness["status"], "unsupported")
        self.assertTrue(
            any(
                "Unsupported numeric/date/time claim" in error
                for error in result.generation_trace["validation_errors"]
            )
        )

    def test_fake_invalid_chunk_id_is_blocked(self) -> None:
        generated = GeneratedAnswer(
            refused=False,
            refusal_reason=None,
            answer="標準月付訂閱用戶可於首次購買後 7 天內申請退款。",
            used_chunk_ids=["not-a-retrieved-chunk"],
            claims=[
                {
                    "text": "標準月付訂閱用戶可於首次購買後 7 天內申請退款。",
                    "chunk_ids": ["not-a-retrieved-chunk"],
                }
            ],
            requires_human_review=False,
        )
        with patch(
            "src.answer.generate_answer",
            return_value=(generated, "fake-invalid-v1", "context-only prompt"),
        ):
            result = answer_question(
                "標準月付用戶的退款期限是多久？",
                index_path=self.index_path,
                mode="generative",
                llm_provider="fake_supported",
            )

        self.assertEqual(result.validator_decision, "blocked")
        self.assertEqual(result.blocked_generated_answer, generated.answer)
        self.assertTrue(result.requires_human_review)
        self.assertEqual(result.confidence, "low")
        self.assertFalse(result.citations[0]["chunk_id"] == "not-a-retrieved-chunk")
        self.assertTrue(
            any(
                "not retrieved chunks" in error
                for error in result.generation_trace["validation_errors"]
            )
        )

    def test_missing_api_keys_do_not_affect_extractive_mode(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""},
        ), patch("src.answer.generate_answer") as generate:
            result = answer_question(
                "標準月付用戶的退款期限是多久？",
                index_path=self.index_path,
            )

        generate.assert_not_called()
        self.assertEqual(result.answer_mode, "extractive")
        self.assertFalse(result.refused)

    def test_prompt_contains_only_question_and_retrieved_context(self) -> None:
        chunks = load_chunks(self.index_path)[:2]
        retrieved = [
            {
                "chunk_id": chunk["chunk_id"],
                "doc": chunk["doc"],
                "section": chunk["section"],
                "section_zh": chunk["section_zh"],
                "section_slug": chunk["section_slug"],
                "page": chunk["page"],
                "score": 1.0,
                "retrieval_method": "lexical",
                "text": chunk["text"],
            }
            for chunk in chunks
        ]
        prompt = build_generation_prompt("test question", retrieved)

        self.assertIn("test question", prompt)
        for chunk in retrieved:
            self.assertIn(chunk["chunk_id"], prompt)
            self.assertIn(chunk["text"], prompt)
        self.assertIn("using only the retrieved CONTEXT", prompt)


if __name__ == "__main__":
    unittest.main()
