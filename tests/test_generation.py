from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.answer import answer_question
from src.generation import (
    GENERATION_CONTRACT,
    GENERATION_JSON_SCHEMA,
    GenerationError,
    GeneratedAnswer,
    build_generation_prompt,
    generate_answer,
)
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
            {
                "OPENAI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
                "MINIMAX_API_KEY": "",
            },
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
        self.assertIn(GENERATION_CONTRACT, prompt)
        self.assertEqual(
            GENERATION_JSON_SCHEMA["required"],
            ["status", "answer", "claims", "missing_evidence"],
        )


@unittest.skipUnless(
    os.environ.get("MINIMAX_API_KEY"),
    "MINIMAX_API_KEY is not configured; skipping optional live MiniMax tests",
)
class MiniMaxLiveGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.index_path = Path(cls._temporary_directory.name) / "chunks.jsonl"
        ingest(DEFAULT_CORPUS_DIR, cls.index_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_minimax_provider_smoke_returns_parseable_proposal(self) -> None:
        chunk = {
            "chunk_id": "smoke-refund-window",
            "doc": "smoke_policy.md",
            "section": "Refund window",
            "section_zh": "退款期限",
            "section_slug": "refund_window",
            "page": None,
            "score": 1.0,
            "retrieval_method": "lexical",
            "text": "Customers may request a refund within 7 days of purchase.",
        }

        generated, model, _ = generate_answer(
            "What is the refund request window?",
            [chunk],
            provider="minimax",
        )

        self.assertEqual(model, os.environ.get("MINIMAX_MODEL", "MiniMax-M3"))
        self.assertEqual(generated.contract_status, "answered")
        self.assertFalse(generated.refused)
        self.assertTrue(generated.answer)
        self.assertTrue(generated.claims)
        self.assertIn("smoke-refund-window", generated.used_chunk_ids)
        self.assertIsNone(generated.parse_error)

    def test_minimax_answerable_case_passes_validator(self) -> None:
        result = answer_question(
            "標準月付用戶的退款期限是多久？",
            index_path=self.index_path,
            retriever="lexical",
            top_k=3,
            mode="generative",
            llm_provider="minimax",
        )

        retrieved_ids = {chunk["chunk_id"] for chunk in result.retrieved_chunks}
        self.assertEqual(result.validator_decision, "allowed")
        self.assertFalse(result.refused)
        self.assertEqual(result.groundedness["status"], "supported")
        self.assertTrue(result.citations)
        self.assertTrue(
            all(citation["chunk_id"] in retrieved_ids for citation in result.citations)
        )

    def test_minimax_unsupported_medical_exception_is_blocked(self) -> None:
        generated = GeneratedAnswer(
            refused=False,
            refusal_reason=None,
            answer="Medical circumstances allow a refund 90 days after purchase.",
            used_chunk_ids=["invented-medical-policy"],
            claims=[
                {
                    "text": "Medical circumstances allow a refund 90 days after purchase.",
                    "chunk_ids": ["invented-medical-policy"],
                }
            ],
            requires_human_review=False,
            contract_status="answered",
        )
        with patch(
            "src.answer.generate_answer",
            return_value=(generated, "MiniMax-M3", "closed-book prompt"),
        ):
            result = answer_question(
                "Can customers get a refund after 90 days for medical reasons?",
                index_path=self.index_path,
                retriever="lexical",
                mode="generative",
                llm_provider="minimax",
            )

        self.assertEqual(result.validator_decision, "blocked")
        self.assertTrue(result.refused)
        self.assertTrue(result.requires_human_review)
        self.assertNotEqual(result.answer, generated.answer)
        self.assertEqual(result.blocked_generated_answer, generated.answer)
        self.assertEqual(result.generation_trace["provider"], "minimax")

    def test_minimax_malformed_json_becomes_blocked_proposal(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: response)
            )
        )
        with patch("openai.OpenAI", return_value=client):
            generated, _, _ = generate_answer(
                "What is supported?",
                [],
                provider="minimax",
            )

        self.assertEqual(generated.answer, "not-json")
        self.assertIsNotNone(generated.parse_error)
        self.assertTrue(generated.requires_human_review)

        with patch(
            "src.answer.generate_answer",
            return_value=(generated, "MiniMax-M3", "closed-book prompt"),
        ):
            result = answer_question(
                "標準月付用戶的退款期限是多久？",
                index_path=self.index_path,
                mode="generative",
                llm_provider="minimax",
            )
        self.assertEqual(result.validator_decision, "blocked")
        self.assertEqual(result.blocked_generated_answer, "not-json")
        self.assertNotEqual(result.answer, "not-json")

        with patch("openai.OpenAI", return_value=client):
            with self.assertRaises(GenerationError):
                generate_answer(
                    "What is supported?",
                    [],
                    provider="minimax",
                    fail_fast=True,
                )

    def test_minimax_retries_one_rate_limit_then_succeeds(self) -> None:
        rate_limit = RuntimeError("rate limited")
        rate_limit.status_code = 429
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"status":"insufficient_evidence",'
                            '"answer":"The provided evidence is insufficient.",'
                            '"claims":[],"missing_evidence":["policy"]}'
                        )
                    )
                )
            ]
        )
        calls = iter((rate_limit, response))

        def create(**_: object) -> object:
            outcome = next(calls)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with patch.dict(
            os.environ,
            {"MINIMAX_MAX_RETRIES": "1", "MINIMAX_RETRY_BASE_SECONDS": "0"},
        ), patch("openai.OpenAI", return_value=client), patch(
            "src.generation.time.sleep"
        ) as sleep:
            generated, _, _ = generate_answer(
                "What is supported?",
                [],
                provider="minimax",
            )

        self.assertTrue(generated.refused)
        sleep.assert_called_once_with(0.0)


if __name__ == "__main__":
    unittest.main()
