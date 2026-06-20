from __future__ import annotations

import unittest
from unittest.mock import patch

from src.answer import answer_question
from src.generation import GeneratedAnswer


class NonKBQueryGuardTests(unittest.TestCase):
    def test_canned_queries_skip_retrieval_and_generation(self) -> None:
        queries = (
            "",
            "   ",
            "你好",
            "hi",
            "Thanks!",
            "謝謝。",
            "你可以做什麼？",
            "What can you do?",
            "這個 App 是做什麼的？",
            "What does this app do?",
        )

        with patch("src.answer.retrieve") as retrieve, patch(
            "src.answer.generate_answer"
        ) as generate:
            for query in queries:
                with self.subTest(query=query):
                    result = answer_question(
                        query,
                        mode="generative",
                        llm_provider="fake_supported",
                    )

                    self.assertEqual(result.response_type, "non_kb_chitchat")
                    self.assertEqual(result.validator_decision, "not_run")
                    self.assertEqual(result.groundedness["status"], "not_applicable")
                    self.assertFalse(result.refused)
                    self.assertFalse(result.requires_human_review)
                    self.assertEqual(result.citations, [])
                    self.assertEqual(result.retrieved_chunks, [])

        retrieve.assert_not_called()
        generate.assert_not_called()

    def test_app_intro_message_describes_supported_capabilities(self) -> None:
        result = answer_question("你可以做什麼？")

        self.assertIn("知識庫", result.answer)
        self.assertIn("就緒度", result.answer)
        self.assertIn("政策變更", result.answer)

    def test_policy_question_uses_retrieval_and_generation_validator(self) -> None:
        question = "Can customers get a refund after 90 days for medical reasons?"
        retrieved = [
            {
                "chunk_id": "refund-policy-unsupported-exceptions",
                "doc": "refund_policy.md",
                "section": "Unsupported exceptions",
                "section_zh": "不支援的例外",
                "section_slug": "unsupported_exceptions",
                "page": None,
                "score": 10.0,
                "retrieval_method": "lexical",
                "text": "The policy does not define medical refund exceptions.",
            }
        ]
        generated = GeneratedAnswer(
            refused=False,
            refusal_reason=None,
            answer="Medical circumstances allow a refund after 90 days.",
            used_chunk_ids=["invented-medical-policy"],
            claims=[
                {
                    "text": "Medical circumstances allow a refund after 90 days.",
                    "chunk_ids": ["invented-medical-policy"],
                }
            ],
            requires_human_review=False,
            contract_status="answered",
        )

        with patch("src.answer.retrieve", return_value=retrieved) as retrieve, patch(
            "src.answer.generate_answer",
            return_value=(generated, "fake-model", "closed-book prompt"),
        ) as generate:
            result = answer_question(
                question,
                mode="generative",
                llm_provider="fake_hallucination",
            )

        retrieve.assert_called_once()
        generate.assert_called_once()
        self.assertEqual(result.response_type, "kb_answer")
        self.assertEqual(result.validator_decision, "blocked")
        self.assertTrue(result.refused)
        self.assertNotEqual(result.answer, generated.answer)
        self.assertEqual(result.blocked_generated_answer, generated.answer)


if __name__ == "__main__":
    unittest.main()
