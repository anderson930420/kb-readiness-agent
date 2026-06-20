from __future__ import annotations

import unittest
from unittest.mock import patch

from eval.run_eval import split_non_kb_eval_cases
from src.answer import (
    answer_question,
    classify_query,
    has_sufficient_query_content,
)
from src.generation import GeneratedAnswer


class NonKBQueryGuardTests(unittest.TestCase):
    def test_canned_queries_skip_retrieval_and_generation(self) -> None:
        queries = (
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

    def test_low_information_queries_use_task_boundary_without_pipeline_work(self) -> None:
        queries = (
            "",
            "   ",
            "我",
            "你",
            "他",
            "那個",
            "可以嗎",
            "幫我",
            "我想問",
            "?",
            "？？",
            "what about",
            "tell me more",
            "can you",
            "help me",
            "I",
            "me",
        )

        with patch("src.answer.retrieve") as retrieve, patch(
            "src.answer.generate_answer"
        ) as generate, patch("src.answer._generate_and_validate") as validate:
            for query in queries:
                with self.subTest(query=query):
                    self.assertFalse(has_sufficient_query_content(query))
                    result = answer_question(query, mode="generative")
                    self.assertEqual(result.response_type, "out_of_scope_general")
                    self.assertEqual(result.retrieved_chunks, [])
                    self.assertEqual(result.validator_decision, "not_run")

        retrieve.assert_not_called()
        generate.assert_not_called()
        validate.assert_not_called()

    def test_app_intro_message_describes_supported_capabilities(self) -> None:
        result = answer_question("你可以做什麼？")

        self.assertIn("知識庫", result.answer)
        self.assertIn("就緒度", result.answer)
        self.assertIn("政策變更", result.answer)

    def test_general_queries_return_canned_out_of_scope_refusal(self) -> None:
        queries = (
            "What is the capital of France?",
            "What's the weather today?",
            "Tell me a joke.",
            "How do I bake a cake?",
            "今天天氣如何？",
            "法國的首都是哪裡？",
            "世界最高的山是什麼？",
        )

        with patch("src.answer.retrieve") as retrieve, patch(
            "src.answer.generate_answer"
        ) as generate:
            for query in queries:
                with self.subTest(query=query):
                    result = answer_question(
                        query,
                        mode="generative",
                    )

                    self.assertEqual(result.response_type, "out_of_scope_general")
                    self.assertEqual(result.refusal_reason, "out_of_scope_general")
                    self.assertEqual(result.validator_decision, "not_run")
                    self.assertEqual(result.groundedness["status"], "not_applicable")
                    self.assertTrue(result.refused)
                    self.assertFalse(result.requires_human_review)
                    self.assertEqual(result.citations, [])
                    self.assertEqual(result.retrieved_chunks, [])

        retrieve.assert_not_called()
        generate.assert_not_called()

    def test_identity_queries_use_existing_out_of_scope_response(self) -> None:
        expected_answer = (
            "This Ask Mode is scoped to the indexed support knowledge base. I can "
            "help with grounded KB questions about refunds, pricing, Enterprise "
            "plans, privacy, onboarding, SLA, support escalation, readiness audit, "
            "and policy change impact. Your question is outside this demo's task "
            "boundary."
        )
        queries = (
            "我是誰",
            "我叫什麼",
            "你知道我是誰嗎",
            "你認識我嗎",
            "who am I",
            "what is my name",
            "do you know me",
            "what is my identity",
        )

        with patch("src.answer.retrieve") as retrieve, patch(
            "src.answer.generate_answer"
        ) as generate, patch("src.answer._generate_and_validate") as validate:
            for query in queries:
                with self.subTest(query=query):
                    result = answer_question(query, mode="generative")

                    self.assertEqual(result.response_type, "out_of_scope_general")
                    self.assertEqual(result.answer, expected_answer)
                    self.assertEqual(result.citations, [])
                    self.assertEqual(result.retrieved_chunks, [])
                    self.assertEqual(result.validator_decision, "not_run")
                    self.assertEqual(result.groundedness["status"], "not_applicable")
                    self.assertFalse(result.requires_human_review)
                    self.assertIsNone(result.blocked_generated_answer)

        retrieve.assert_not_called()
        generate.assert_not_called()
        validate.assert_not_called()

    def test_business_scope_signals_always_use_kb_pipeline(self) -> None:
        queries = (
            "What is your refund policy?",
            "How much does the Enterprise plan cost?",
            "How can I contact support?",
            "What services does the company offer?",
            "今天天氣會影響客服服務嗎？",
        )

        retrieved = [
            {
                "chunk_id": "company-kb-evidence",
                "doc": "company_kb.md",
                "section": "Company information",
                "section_zh": "公司資訊",
                "section_slug": "company_information",
                "page": None,
                "score": 10.0,
                "retrieval_method": "lexical",
                "text": "Company knowledge-base evidence.",
            }
        ]

        with patch("src.answer.retrieve", return_value=retrieved) as retrieve:
            for query in queries:
                with self.subTest(query=query):
                    result = answer_question(query)

                    self.assertEqual(classify_query(query), "kb_answer")
                    self.assertEqual(result.response_type, "kb_answer")
                    self.assertTrue(result.citations)
                    self.assertEqual(
                        result.citations[0]["chunk_id"], "company-kb-evidence"
                    )

        self.assertEqual(retrieve.call_count, len(queries))

    def test_domain_signaled_short_queries_enter_kb_pipeline(self) -> None:
        retrieved = [
            {
                "chunk_id": "refund-policy",
                "doc": "refund_policy.md",
                "section": "Refunds",
                "section_zh": "退款",
                "section_slug": "refunds",
                "page": None,
                "score": 4.0,
                "retrieval_method": "lexical",
                "text": "Refund policy evidence.",
            }
        ]

        with patch("src.answer.retrieve", return_value=retrieved) as retrieve:
            for query in ("退款？", "SLA?", "pricing", "KB"):
                with self.subTest(query=query):
                    result = answer_question(query)
                    self.assertEqual(result.response_type, "kb_answer")

        self.assertEqual(retrieve.call_count, 4)

    def test_semantic_kb_query_is_admitted_by_high_relevance_probe(self) -> None:
        query = "購買後多久可以拿回款項？"
        retrieved = [
            {
                "chunk_id": "refund-window",
                "doc": "refund_policy.md",
                "section": "Refund window",
                "section_zh": "退款期限",
                "section_slug": "standard_refund_window",
                "page": None,
                "score": 8.0,
                "retrieval_method": "lexical",
                "text": "Customers can receive their money back after an approved request.",
            }
        ]

        with patch("src.answer.retrieve", return_value=retrieved) as retrieve:
            result = answer_question(query)

        retrieve.assert_called_once()
        self.assertTrue(has_sufficient_query_content(query))
        self.assertEqual(result.response_type, "kb_answer")
        self.assertEqual(result.citations[0]["chunk_id"], "refund-window")

    def test_low_relevance_probe_stops_before_generation_and_validator(self) -> None:
        query = "Explain mountain climbing equipment"
        retrieved = [
            {
                "chunk_id": "weak-match",
                "doc": "support.md",
                "section": "General",
                "section_zh": "一般",
                "section_slug": "general",
                "page": None,
                "score": 2.99,
                "retrieval_method": "lexical",
                "text": "Weakly matching support evidence.",
            }
        ]

        with patch("src.answer.retrieve", return_value=retrieved) as retrieve, patch(
            "src.answer.generate_answer"
        ) as generate, patch("src.answer._generate_and_validate") as validate:
            result = answer_question(
                query,
                mode="generative",
                llm_provider="fake_hallucination",
            )

        retrieve.assert_called_once()
        generate.assert_not_called()
        validate.assert_not_called()
        self.assertEqual(result.response_type, "out_of_scope_general")
        self.assertEqual(result.retrieved_chunks, [])

    def test_readiness_partition_excludes_general_queries(self) -> None:
        kb_row = {
            "id": "kb",
            "question": "退款政策是什麼？",
            "question_en": "What is the refund policy?",
        }
        general_row = {
            "id": "general",
            "question": "今天天氣如何？",
            "question_en": "What's the weather today?",
        }

        included, excluded = split_non_kb_eval_cases([kb_row, general_row])

        self.assertEqual(included, [kb_row])
        self.assertEqual(excluded, [general_row])

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
