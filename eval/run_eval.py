"""Compare lexical, dense, and hybrid retrieval on bilingual eval questions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.ingest import DEFAULT_INDEX_PATH, PROJECT_ROOT
from src.retrieve import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_EMBEDDING_CACHE_DIR,
    RETRIEVAL_METHODS,
    DenseRetriever,
    HybridRetriever,
    LexicalRetriever,
    load_chunks,
)


DEFAULT_EVAL_PATH = PROJECT_ROOT / "eval" / "eval_set.jsonl"


def _load_eval(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument(
        "--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE_DIR
    )
    parser.add_argument(
        "--retrievers",
        default=",".join(RETRIEVAL_METHODS),
        help="Comma-separated subset of lexical,dense,hybrid",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    selected = [name.strip() for name in args.retrievers.split(",") if name.strip()]
    invalid = sorted(set(selected) - set(RETRIEVAL_METHODS))
    if invalid:
        parser.error(f"unknown retriever(s): {', '.join(invalid)}")

    chunks = load_chunks(args.index)
    lexical = LexicalRetriever(chunks)
    dense = DenseRetriever(
        chunks, model_name=args.model, cache_dir=args.embedding_cache
    )
    backends = {
        "lexical": lexical,
        "dense": dense,
        "hybrid": HybridRetriever(chunks, lexical=lexical, dense=dense),
    }

    rows = [
        row
        for row in _load_eval(args.eval_set)
        if row.get("evaluation_scope") != "p2_change_impact"
    ]
    if args.limit is not None:
        rows = rows[: args.limit]

    doc_totals: dict[tuple[str, str], list[bool]] = defaultdict(list)
    section_totals: dict[tuple[str, str], list[bool]] = defaultdict(list)
    print(
        "method\tlang\tid\tdoc_hit\tsection_hit\tscore\t"
        "top_doc\ttop_section\texpected_doc\texpected_section"
    )
    for row in rows:
        expected_docs = set(row.get("source_docs", [row["source_doc"]]))
        for language, question in (("zh", row["question"]), ("en", row["question_en"])):
            for method in selected:
                results = backends[method].search(question, top_k=args.top_k)
                doc_hit = any(result["doc"] in expected_docs for result in results)
                section_hit = any(
                    result["section_slug"] == row["source_section"]
                    for result in results
                )
                doc_totals[(method, language)].append(doc_hit)
                section_totals[(method, language)].append(section_hit)
                top = results[0] if results else None
                print(
                    "\t".join(
                        (
                            method,
                            language,
                            row["id"],
                            "yes" if doc_hit else "no",
                            "yes" if section_hit else "no",
                            f"{top['score']:.4f}" if top else "-",
                            top["doc"] if top else "-",
                            top["section_slug"] if top else "-",
                            row["source_doc"],
                            row["source_section"],
                        )
                    )
                )

    print("\nRecall@k summary (document / section)")
    for method in selected:
        for language in ("zh", "en"):
            doc_hits = doc_totals[(method, language)]
            section_hits = section_totals[(method, language)]
            if not doc_hits:
                print(f"{method:<7} {language}: no rows")
                continue
            doc_correct = sum(doc_hits)
            section_correct = sum(section_hits)
            print(
                f"{method:<7} {language}: "
                f"{doc_correct}/{len(doc_hits)} ({doc_correct / len(doc_hits):.1%}) / "
                f"{section_correct}/{len(section_hits)} "
                f"({section_correct / len(section_hits):.1%})"
            )


if __name__ == "__main__":
    main()
