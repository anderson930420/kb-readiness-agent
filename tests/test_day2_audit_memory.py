from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eval.run_eval import run_evaluation
from src.degraded import generate_degraded_fixture
from src.ingest import DEFAULT_CORPUS_DIR, ingest
from src.session import AnswerSession


class Day2DegradedAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        root = Path(cls._temporary_directory.name)
        cls.healthy_index = root / "healthy" / "chunks.jsonl"
        cls.degraded_corpus = root / "degraded" / "corpus"
        cls.degraded_index = root / "degraded" / "index" / "chunks.jsonl"

        before = {
            path.name: path.read_bytes()
            for path in sorted(DEFAULT_CORPUS_DIR.glob("*.md"))
        }
        ingest(DEFAULT_CORPUS_DIR, cls.healthy_index)
        cls.fixture = generate_degraded_fixture(
            corpus_dir=cls.degraded_corpus,
            index_path=cls.degraded_index,
        )
        after = {
            path.name: path.read_bytes()
            for path in sorted(DEFAULT_CORPUS_DIR.glob("*.md"))
        }
        cls.primary_corpus_unchanged = before == after

        cls.healthy = run_evaluation(
            retrievers="lexical",
            index_path=cls.healthy_index,
            write_report=True,
            report_dir=root / "reports" / "healthy",
        )
        cls.degraded = run_evaluation(
            retrievers="lexical",
            index_path=cls.degraded_index,
            write_report=True,
            report_dir=root / "reports" / "degraded",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_generator_does_not_mutate_primary_corpus(self) -> None:
        self.assertTrue(self.primary_corpus_unchanged)
        self.assertFalse((self.degraded_corpus / "refund_policy.md").exists())
        enterprise = (self.degraded_corpus / "enterprise_plan_faq.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("enterprise_support_response_time", enterprise)
        self.assertNotIn("enterprise_pricing_quote", enterprise)
        self.assertEqual(self.fixture.indexed_chunks, 26)

    def test_healthy_corpus_remains_internal_pilot_ready(self) -> None:
        self.assertEqual(self.healthy["gate_status"], "PASS")
        self.assertEqual(
            self.healthy["metrics"]["gate"]["recommendation"],
            "Internal Pilot Ready",
        )

    def test_degraded_corpus_is_not_ready_with_concrete_gaps(self) -> None:
        self.assertEqual(self.degraded["gate_status"], "FAIL")
        self.assertEqual(
            self.degraded["metrics"]["gate"]["recommendation"], "Not Ready"
        )
        topics = {
            gap["topic"] for gap in self.degraded["metrics"]["knowledge_gaps"]
        }
        self.assertTrue(
            {
                "Standard and annual refund windows",
                "Renewal payment refund policy",
                "Refund processing timeline",
                "Refund exception / hardship policy",
                "Signed Enterprise SLA",
                "Enterprise quote handling",
            }.issubset(topics)
        )
        report = self.degraded["report_path"].read_text(encoding="utf-8")
        self.assertIn("**Not Ready**", report)
        self.assertIn("Standard and annual refund windows", report)


class Day2SessionMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.index_path = Path(cls._temporary_directory.name) / "chunks.jsonl"
        ingest(DEFAULT_CORPUS_DIR, cls.index_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_enterprise_follow_up_is_resolved_before_grounded_pipeline(self) -> None:
        session = AnswerSession(index_path=self.index_path, retriever="lexical")

        first = session.ask("What is the standard refund window?")
        second = session.ask("What about enterprise customers?")

        self.assertEqual(first.original_question, first.resolved_question)
        self.assertIn("Enterprise customer refund policy", second.resolved_question)
        self.assertEqual(second.answer.question, second.resolved_question)
        slugs = {
            chunk["section_slug"] for chunk in second.answer.retrieved_chunks
        }
        self.assertTrue(
            {
                "enterprise_automatic_refunds",
                "enterprise_refunds",
                "refund_escalation",
            }.issubset(slugs)
        )
        self.assertFalse(second.answer.refused)
        self.assertEqual(second.answer.answer_mode, "extractive")
        self.assertEqual(second.answer.validator_decision, "not_run")
        self.assertEqual(second.answer.groundedness["status"], "supported")
        self.assertTrue(second.answer.citations)
        retrieved_ids = {
            chunk["chunk_id"] for chunk in second.answer.retrieved_chunks
        }
        self.assertTrue(
            all(
                citation["chunk_id"] in retrieved_ids
                for citation in second.answer.citations
            )
        )

    def test_session_memory_can_be_cleared_and_is_not_persistent(self) -> None:
        session = AnswerSession(index_path=self.index_path, retriever="lexical")
        session.ask("What is the standard refund window?")
        session.clear()

        turn = session.ask("What about enterprise customers?")

        self.assertEqual(turn.resolved_question, turn.original_question)
        self.assertEqual(len(session.turns), 1)


if __name__ == "__main__":
    unittest.main()
