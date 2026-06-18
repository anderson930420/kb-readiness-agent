"""Standard-library lexical retrieval for Chinese and English support queries."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import TypedDict

from .ingest import DEFAULT_INDEX_PATH


LATIN_RE = re.compile(r"[a-zA-Z0-9]+")
CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
ENGLISH_STOP_WORDS = {
    "a",
    "after",
    "an",
    "and",
    "are",
    "be",
    "can",
    "do",
    "does",
    "for",
    "from",
    "get",
    "how",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "the",
    "their",
    "to",
    "what",
    "when",
    "who",
    "with",
}


class SearchResult(TypedDict, total=False):
    chunk_id: str
    doc: str
    section: str
    section_zh: str
    section_slug: str
    page: None
    text: str
    content: str
    aliases: str
    score: float


def tokenize(text: str) -> list[str]:
    tokens = [
        token.lower()
        for token in LATIN_RE.findall(text)
        if token.lower() not in ENGLISH_STOP_WORDS
    ]
    for run in CJK_RUN_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def load_chunks(index_path: Path = DEFAULT_INDEX_PATH) -> list[dict]:
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}. Run `python -m src.ingest` first."
        )
    return [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class LexicalRetriever:
    """Small BM25-style retriever over bilingual chunk metadata and evidence."""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.documents = [
            tokenize(
                " ".join(
                    [
                        chunk.get("doc", ""),
                        chunk.get("section", ""),
                        chunk.get("section_slug", ""),
                        chunk.get("aliases", ""),
                        chunk.get("text", chunk.get("content", "")),
                    ]
                )
            )
            for chunk in chunks
        ]
        self.term_frequencies = [Counter(document) for document in self.documents]
        self.document_frequencies = Counter(
            token for document in self.documents for token in set(document)
        )
        self.average_length = (
            sum(map(len, self.documents)) / len(self.documents) if self.documents else 0.0
        )

    def _idf(self, token: str) -> float:
        count = len(self.documents)
        frequency = self.document_frequencies.get(token, 0)
        return math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))

    def search(self, question: str, top_k: int = 5) -> list[SearchResult]:
        query_tokens = set(tokenize(question))
        if not query_tokens or top_k <= 0:
            return []

        scored: list[SearchResult] = []
        k1 = 1.5
        b = 0.75
        for chunk, document, frequencies in zip(
            self.chunks, self.documents, self.term_frequencies
        ):
            length_ratio = len(document) / self.average_length if self.average_length else 0
            score = 0.0
            for token in query_tokens:
                term_frequency = frequencies.get(token, 0)
                if not term_frequency:
                    continue
                denominator = term_frequency + k1 * (1 - b + b * length_ratio)
                score += self._idf(token) * (term_frequency * (k1 + 1)) / denominator

            if score > 0:
                result = dict(chunk)
                result["score"] = round(score, 4)
                scored.append(result)

        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]


def retrieve(
    question: str,
    *,
    top_k: int = 5,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> list[SearchResult]:
    return LexicalRetriever(load_chunks(index_path)).search(question, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    args = parser.parse_args()

    for result in retrieve(args.question, top_k=args.top_k, index_path=args.index):
        print(
            f"{result['score']:>7.3f} | {result['doc']} | "
            f"{result['section']} | {result['chunk_id']}"
        )
        print(result["text"])
        print()


if __name__ == "__main__":
    main()
