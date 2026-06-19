"""Lexical, multilingual dense, and hybrid retrieval for Ask Mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal, Protocol, TypedDict

from .ingest import DEFAULT_INDEX_PATH, PROJECT_ROOT


DEFAULT_DENSE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_CACHE_DIR = PROJECT_ROOT / "data" / "index" / "embeddings"
RETRIEVAL_METHODS = ("lexical", "dense", "hybrid")
RetrieverName = Literal["lexical", "dense", "hybrid"]

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


class RetrievalResult(TypedDict):
    """Normalized result returned by every retrieval backend."""

    chunk_id: str
    doc: str
    section: str
    section_zh: str
    section_slug: str
    page: int | None
    score: float
    retrieval_method: RetrieverName
    text: str


# Backward-compatible name for Day 1 imports.
SearchResult = RetrievalResult


class Retriever(Protocol):
    def search(self, question: str, top_k: int = 5) -> list[RetrievalResult]: ...


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


def _result(
    chunk: dict, score: float, retrieval_method: RetrieverName
) -> RetrievalResult:
    return {
        "chunk_id": chunk["chunk_id"],
        "doc": chunk["doc"],
        "section": chunk["section"],
        "section_zh": chunk.get("section_zh", chunk["section"]),
        "section_slug": chunk.get("section_slug", ""),
        "page": chunk.get("page"),
        "score": round(float(score), 6),
        "retrieval_method": retrieval_method,
        "text": chunk.get("text", chunk.get("content", "")),
    }


def _retrieval_text(chunk: dict) -> str:
    return "\n".join(
        value
        for value in (
            chunk.get("doc", ""),
            chunk.get("section", ""),
            chunk.get("section_slug", ""),
            chunk.get("aliases", ""),
            chunk.get("text", chunk.get("content", "")),
        )
        if value
    )


class LexicalRetriever:
    """Small BM25-style retriever over bilingual chunk metadata and evidence."""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.documents = [tokenize(_retrieval_text(chunk)) for chunk in chunks]
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

    def search(self, question: str, top_k: int = 5) -> list[RetrievalResult]:
        query_tokens = set(tokenize(question))
        if not query_tokens or top_k <= 0:
            return []

        scored: list[RetrievalResult] = []
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
                scored.append(_result(chunk, score, "lexical"))

        return sorted(
            scored, key=lambda item: (-item["score"], item["chunk_id"])
        )[:top_k]


class DenseRetriever:
    """Cosine-similarity retrieval with locally cached corpus embeddings."""

    def __init__(
        self,
        chunks: list[dict],
        *,
        model_name: str = DEFAULT_DENSE_MODEL,
        cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
        model: object | None = None,
    ) -> None:
        self.chunks = chunks
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = model
        self._embeddings = None

    def _load_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "Dense retrieval requires sentence-transformers. "
                    "Install dependencies with `python -m pip install -r requirements.txt`."
                ) from error
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as online_error:
                # Recent Hugging Face clients may perform metadata requests even
                # when all model files are cached. Restricted/offline runtimes
                # should still be able to use that complete local cache.
                try:
                    self._model = SentenceTransformer(
                        self.model_name, local_files_only=True
                    )
                except Exception:
                    raise online_error
        return self._model

    def _fingerprint(self) -> str:
        payload = {
            "model": self.model_name,
            "chunks": [
                [chunk["chunk_id"], _retrieval_text(chunk)] for chunk in self.chunks
            ],
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def cache_path(self) -> Path:
        safe_model_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", self.model_name)
        return self.cache_dir / f"{safe_model_name}-{self._fingerprint()[:16]}.npz"

    def _load_or_encode_embeddings(self):
        if self._embeddings is not None:
            return self._embeddings

        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("Dense retrieval requires numpy.") from error

        cache_path = self.cache_path
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                embeddings = cached["embeddings"]
            if len(embeddings) == len(self.chunks):
                self._embeddings = embeddings
                return embeddings

        model = self._load_model()
        embeddings = model.encode(
            [_retrieval_text(chunk) for chunk in self.chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=cache_path.parent, suffix=".npz", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            np.savez_compressed(temporary_path, embeddings=embeddings)
            os.replace(temporary_path, cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        self._embeddings = embeddings
        return embeddings

    def search(self, question: str, top_k: int = 5) -> list[RetrievalResult]:
        if not question.strip() or top_k <= 0 or not self.chunks:
            return []

        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("Dense retrieval requires numpy.") from error

        model = self._load_model()
        embeddings = self._load_or_encode_embeddings()
        query_embedding = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = embeddings @ np.asarray(query_embedding, dtype=np.float32)
        ranked_indices = sorted(
            range(len(self.chunks)),
            key=lambda index: (-float(scores[index]), self.chunks[index]["chunk_id"]),
        )
        return [
            _result(self.chunks[index], scores[index], "dense")
            for index in ranked_indices[:top_k]
        ]


def _min_max_scores(results: list[RetrievalResult]) -> dict[str, float]:
    if not results:
        return {}
    values = [result["score"] for result in results]
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return {result["chunk_id"]: 1.0 for result in results}
    scale = maximum - minimum
    return {
        result["chunk_id"]: (result["score"] - minimum) / scale
        for result in results
    }


class HybridRetriever:
    """Equal-weight fusion of min-max normalized lexical and dense scores."""

    def __init__(
        self,
        chunks: list[dict],
        *,
        lexical: LexicalRetriever | None = None,
        dense: DenseRetriever | None = None,
    ) -> None:
        self.chunks = chunks
        self.lexical = lexical or LexicalRetriever(chunks)
        self.dense = dense or DenseRetriever(chunks)

    def search(self, question: str, top_k: int = 5) -> list[RetrievalResult]:
        if not question.strip() or top_k <= 0:
            return []

        lexical_results = self.lexical.search(question, top_k=len(self.chunks))
        dense_results = self.dense.search(question, top_k=len(self.chunks))
        lexical_scores = _min_max_scores(lexical_results)
        dense_scores = _min_max_scores(dense_results)

        scored = [
            _result(
                chunk,
                0.5 * lexical_scores.get(chunk["chunk_id"], 0.0)
                + 0.5 * dense_scores.get(chunk["chunk_id"], 0.0),
                "hybrid",
            )
            for chunk in self.chunks
        ]
        return sorted(
            scored, key=lambda item: (-item["score"], item["chunk_id"])
        )[:top_k]


def build_retriever(
    chunks: list[dict],
    *,
    retriever: RetrieverName = "lexical",
    model_name: str = DEFAULT_DENSE_MODEL,
    cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
) -> Retriever:
    if retriever == "lexical":
        return LexicalRetriever(chunks)

    dense = DenseRetriever(chunks, model_name=model_name, cache_dir=cache_dir)
    if retriever == "dense":
        return dense
    if retriever == "hybrid":
        return HybridRetriever(chunks, dense=dense)
    raise ValueError(f"Unknown retriever: {retriever}")


def retrieve(
    question: str,
    *,
    top_k: int = 5,
    index_path: Path = DEFAULT_INDEX_PATH,
    retriever: RetrieverName = "lexical",
    model_name: str = DEFAULT_DENSE_MODEL,
    cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
) -> list[RetrievalResult]:
    backend = build_retriever(
        load_chunks(index_path),
        retriever=retriever,
        model_name=model_name,
        cache_dir=cache_dir,
    )
    return backend.search(question, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--retriever", choices=RETRIEVAL_METHODS, default="lexical")
    parser.add_argument("--model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument(
        "--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE_DIR
    )
    args = parser.parse_args()

    for result in retrieve(
        args.question,
        top_k=args.top_k,
        index_path=args.index,
        retriever=args.retriever,
        model_name=args.model,
        cache_dir=args.embedding_cache,
    ):
        print(
            f"{result['score']:>8.4f} | {result['retrieval_method']:<7} | "
            f"{result['doc']} | {result['section']} | {result['chunk_id']}"
        )
        print(result["text"])
        print()


if __name__ == "__main__":
    main()
