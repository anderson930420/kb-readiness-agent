from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.ingest import DEFAULT_CORPUS_DIR, ingest
from src.answer import answer_question
from src.retrieve import (
    DenseRetriever,
    HybridRetriever,
    LexicalRetriever,
    load_chunks,
)


class FakeMultilingualModel:
    """Small deterministic encoder used to test dense plumbing without downloads."""

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        self.encode_calls += 1
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.array(
                [
                    float(
                        "standard_refund_window" in lowered
                        or "refund window" in lowered
                        or "退款期限" in text
                    ),
                    float(
                        "sensitive_data" in lowered
                        or "medical records" in lowered
                        or "醫療紀錄" in text
                    ),
                    float(
                        "unsupported_exceptions" in lowered
                        or "90 days" in lowered
                        or "90 天" in text
                    ),
                ],
                dtype=np.float32,
            )
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.asarray(vectors, dtype=np.float32)


class Day2RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        root = Path(cls._temporary_directory.name)
        cls.index_path = root / "chunks.jsonl"
        cls.cache_dir = root / "embeddings"
        ingest(DEFAULT_CORPUS_DIR, cls.index_path)
        cls.chunks = load_chunks(cls.index_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_result_schema_and_dense_bilingual_queries(self) -> None:
        dense = DenseRetriever(
            self.chunks,
            model_name="fake-multilingual-model",
            cache_dir=self.cache_dir,
            model=FakeMultilingualModel(),
        )
        required = {
            "chunk_id",
            "doc",
            "section",
            "section_zh",
            "page",
            "score",
            "retrieval_method",
            "text",
        }
        for question in (
            "What is the refund window for standard monthly subscribers?",
            "標準月付用戶的退款期限是多久？",
        ):
            result = dense.search(question, top_k=1)[0]
            self.assertTrue(required.issubset(result))
            self.assertEqual(result["doc"], "refund_policy.md")
            self.assertEqual(result["retrieval_method"], "dense")

    def test_embeddings_are_reused_from_local_cache(self) -> None:
        first_model = FakeMultilingualModel()
        first = DenseRetriever(
            self.chunks,
            model_name="cache-test-model",
            cache_dir=self.cache_dir,
            model=first_model,
        )
        first.search("醫療紀錄", top_k=1)
        self.assertEqual(first_model.encode_calls, 2)

        second_model = FakeMultilingualModel()
        second = DenseRetriever(
            self.chunks,
            model_name="cache-test-model",
            cache_dir=self.cache_dir,
            model=second_model,
        )
        second.search("醫療紀錄", top_k=1)
        self.assertEqual(second_model.encode_calls, 1)
        self.assertTrue(second.cache_path.exists())

    def test_hybrid_is_deterministic_and_preserves_core_matches(self) -> None:
        dense = DenseRetriever(
            self.chunks,
            model_name="fake-hybrid-model",
            cache_dir=self.cache_dir,
            model=FakeMultilingualModel(),
        )
        hybrid = HybridRetriever(
            self.chunks,
            lexical=LexicalRetriever(self.chunks),
            dense=dense,
        )
        cases = {
            "標準月付用戶的退款期限是多久？": "refund_policy.md",
            "客戶可以把醫療紀錄上傳到客服工單嗎？": "privacy_policy.md",
            "Can customers get a refund after 90 days for medical reasons?": "refund_policy.md",
        }
        for question, expected_doc in cases.items():
            first = hybrid.search(question, top_k=3)
            second = hybrid.search(question, top_k=3)
            self.assertEqual(first, second)
            self.assertEqual(first[0]["doc"], expected_doc)
            self.assertTrue(all(item["retrieval_method"] == "hybrid" for item in first))

    def test_dense_answer_uses_relevant_refusal_evidence_below_rank_one(self) -> None:
        def result(slug: str, score: float) -> dict:
            return {
                "chunk_id": f"refund-policy-{slug}",
                "doc": "refund_policy.md",
                "section": slug,
                "section_zh": slug,
                "section_slug": slug,
                "page": None,
                "score": score,
                "retrieval_method": "dense",
                "text": slug,
            }

        retrieved = [
            result("refund_processing_time", 0.71),
            result("standard_refund_window", 0.70),
            result("unsupported_exceptions", 0.55),
        ]
        with patch("src.answer.retrieve", return_value=retrieved):
            answer = answer_question(
                "Can customers get a refund after 90 days for medical reasons?",
                retriever="dense",
            )

        self.assertTrue(answer.refused)
        self.assertEqual(answer.citations[0]["section_slug"], "unsupported_exceptions")


if __name__ == "__main__":
    unittest.main()
