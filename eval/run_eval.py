"""Run the official bilingual Ask Mode retrieval and answer gate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from src.answer import answer_from_retrieved
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


def _ratio(values: list[bool]) -> str:
    correct = sum(values)
    total = len(values)
    return f"{correct}/{total} ({correct / total:.1%})" if total else "no rows"


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
        "--retriever",
        dest="retrievers",
        default="hybrid",
        help="Comma-separated subset of lexical,dense,hybrid (default: hybrid)",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    selected = [name.strip() for name in args.retrievers.split(",") if name.strip()]
    invalid = sorted(set(selected) - set(RETRIEVAL_METHODS))
    if invalid:
        parser.error(f"unknown retriever(s): {', '.join(invalid)}")

    all_rows = _load_eval(args.eval_set)
    excluded_rows = [
        row for row in all_rows if row.get("evaluation_scope") == "p2_change_impact"
    ]
    active_rows = [
        row for row in all_rows if row.get("evaluation_scope") != "p2_change_impact"
    ]
    rows = active_rows[: args.limit] if args.limit is not None else active_rows

    print(f"Total eval cases: {len(all_rows)}")
    print(f"Included active Ask Mode cases: {len(active_rows)}")
    print(f"Excluded P2 conflict/change cases: {len(excluded_rows)}")
    if args.limit is not None:
        print(f"Cases executed due to --limit: {len(rows)}")

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

    metric_names = (
        "source_hit",
        "section_hit",
        "correct_refusal",
        "citation_coverage",
        "groundedness_pass",
    )
    totals: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    failures: list[dict] = []

    print(
        f"\nmethod\tlang\tid\tsource_hit@{args.top_k}\tsection_hit@{args.top_k}\t"
        "correct_refusal\tcitation\tgrounded\ttop1_score\ttop1_doc\t"
        "top1_section\texpected_doc\texpected_section"
    )
    for row in rows:
        expected_docs = set(row.get("source_docs", [row["source_doc"]]))
        for language, question in (("zh", row["question"]), ("en", row["question_en"])):
            for method in selected:
                started = perf_counter()
                results = backends[method].search(question, top_k=args.top_k)
                latency_ms = (perf_counter() - started) * 1000
                answer = answer_from_retrieved(
                    question,
                    results,
                    retriever=method,
                    latency_ms=latency_ms,
                )
                checks = {
                    "source_hit": any(
                        result["doc"] in expected_docs for result in results
                    ),
                    "section_hit": any(
                        result["section_slug"] == row["source_section"]
                        for result in results
                    ),
                    "correct_refusal": answer.refused == (not row["answerable"]),
                    "citation_coverage": bool(answer.citations),
                    "groundedness_pass": answer.groundedness["status"] == "supported",
                }
                for metric in metric_names:
                    totals[(method, language, metric)].append(checks[metric])

                failed_metrics = [
                    metric for metric in metric_names if not checks[metric]
                ]
                if failed_metrics:
                    failures.append(
                        {
                            "method": method,
                            "lang": language,
                            "id": row["id"],
                            "failed": failed_metrics,
                            "warnings": answer.warnings,
                        }
                    )

                top = results[0] if results else None
                print(
                    "\t".join(
                        (
                            method,
                            language,
                            row["id"],
                            "yes" if checks["source_hit"] else "no",
                            "yes" if checks["section_hit"] else "no",
                            "yes" if checks["correct_refusal"] else "no",
                            "yes" if checks["citation_coverage"] else "no",
                            "yes" if checks["groundedness_pass"] else "no",
                            f"{top['score']:.4f}" if top else "-",
                            top["doc"] if top else "-",
                            top["section_slug"] if top else "-",
                            row["source_doc"],
                            row["source_section"],
                        )
                    )
                )

    print("\nAsk Mode gate summary")
    for method in selected:
        print(f"{method} (k={args.top_k})")
        for language in ("zh", "en"):
            values = {
                metric: totals[(method, language, metric)] for metric in metric_names
            }
            print(
                f"  {language}: source_hit@{args.top_k} {_ratio(values['source_hit'])}; "
                f"section_hit@{args.top_k} {_ratio(values['section_hit'])}; "
                f"correct refusal {_ratio(values['correct_refusal'])}; "
                f"citation coverage {_ratio(values['citation_coverage'])}; "
                f"groundedness pass {_ratio(values['groundedness_pass'])}"
            )

    print("\nPer-case failures")
    if failures:
        for failure in failures:
            warning_text = "; ".join(failure["warnings"]) or "none"
            print(
                f"- {failure['method']} {failure['lang']} {failure['id']}: "
                f"{', '.join(failure['failed'])}; warnings: {warning_text}"
            )
    else:
        print("- None")
    print(f"\nGate: {'FAIL' if failures else 'PASS'}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
