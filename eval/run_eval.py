"""Run the official bilingual Ask Mode retrieval and answer gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Iterable

from src.answer import answer_from_retrieved, uses_kb_pipeline
from src.audit import (
    DEFAULT_REPORT_DIR,
    build_case_evaluation,
    build_metrics,
    load_eval_cases,
    split_eval_cases,
    write_reports,
)
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


def split_non_kb_eval_cases(rows: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Exclude rows whose populated questions are all outside the KB pipeline."""

    included: list[dict] = []
    excluded: list[dict] = []
    for row in rows:
        questions = [
            row.get(field) for field in ("question", "question_en") if row.get(field)
        ]
        if questions and all(not uses_kb_pipeline(question) for question in questions):
            excluded.append(row)
        else:
            included.append(row)
    return included, excluded


def _ratio(values: list[bool]) -> str:
    correct = sum(values)
    total = len(values)
    return f"{correct}/{total} ({correct / total:.1%})" if total else "no rows"


def run_evaluation(
    *,
    retrievers: str | Iterable[str] = ("hybrid",),
    eval_path: Path = DEFAULT_EVAL_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    top_k: int = 3,
    model_name: str = DEFAULT_DENSE_MODEL,
    cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
    limit: int | None = None,
    write_report: bool = False,
    report_dir: Path = DEFAULT_REPORT_DIR,
    emit_output: bool = False,
) -> dict:
    """Run the official gate and return its results for CLI or UI callers."""

    names = retrievers.split(",") if isinstance(retrievers, str) else retrievers
    selected = [name.strip() for name in names if name.strip()]
    invalid = sorted(set(selected) - set(RETRIEVAL_METHODS))
    if invalid:
        raise ValueError(f"unknown retriever(s): {', '.join(invalid)}")
    if not selected:
        raise ValueError("at least one retriever is required")
    if write_report and len(selected) != 1:
        raise ValueError("--write-report requires exactly one retriever")

    all_rows = load_eval_cases(eval_path)
    active_rows, excluded_rows = split_eval_cases(all_rows)
    metric_rows, excluded_non_kb_rows = split_non_kb_eval_cases(active_rows)
    rows = metric_rows[:limit] if limit is not None else metric_rows

    if emit_output:
        print(f"Total eval cases: {len(all_rows)}")
        print(f"Included active KB Ask Mode cases: {len(metric_rows)}")
        print(f"Excluded P2 conflict/change cases: {len(excluded_rows)}")
        print(f"Excluded non-KB cases: {len(excluded_non_kb_rows)}")
        if limit is not None:
            print(f"Cases executed due to --limit: {len(rows)}")

    chunks = load_chunks(index_path)
    lexical = LexicalRetriever(chunks)
    dense = DenseRetriever(
        chunks, model_name=model_name, cache_dir=cache_dir
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
    report_records: list[dict] = []

    if emit_output:
        print(
            f"\nmethod\tlang\tid\tsource_hit@{top_k}\tsection_hit@{top_k}\t"
            "correct_refusal\tcitation\tgrounded\ttop1_score\ttop1_doc\t"
            "top1_section\texpected_doc\texpected_section"
        )
    for row in rows:
        questions = (
            (language, row.get(field))
            for language, field in (("zh", "question"), ("en", "question_en"))
        )
        for language, question in questions:
            if not question:
                continue
            if not uses_kb_pipeline(question):
                continue
            for method in selected:
                started = perf_counter()
                results = backends[method].search(question, top_k=top_k)
                latency_ms = (perf_counter() - started) * 1000
                answer = answer_from_retrieved(
                    question,
                    results,
                    retriever=method,
                    latency_ms=latency_ms,
                )
                case_evaluation = build_case_evaluation(
                    row,
                    language=language,
                    question=question,
                    retrieved=results,
                    answer=answer,
                )
                checks = case_evaluation["checks"]
                if write_report:
                    report_records.append(case_evaluation)
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
                if emit_output:
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

    if emit_output:
        print("\nAsk Mode gate summary")
        for method in selected:
            print(f"{method} (k={top_k})")
            for language in ("zh", "en"):
                values = {
                    metric: totals[(method, language, metric)]
                    for metric in metric_names
                }
                print(
                    f"  {language}: source_hit@{top_k} {_ratio(values['source_hit'])}; "
                    f"section_hit@{top_k} {_ratio(values['section_hit'])}; "
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

    gate_failed = bool(failures)
    metrics = None
    metrics_path = None
    report_path = None
    if write_report:
        metrics = build_metrics(
            retriever=selected[0],
            top_k=top_k,
            all_cases=[*metric_rows, *excluded_rows],
            records=report_records,
        )
        metrics_path, report_path = write_reports(metrics, report_dir)
        gate_failed = metrics["gate"]["status"] == "FAIL"
        if emit_output:
            print(f"\nMetrics JSON: {metrics_path}")
            print(f"Readiness report: {report_path}")
            print(f"Launch recommendation: {metrics['gate']['recommendation']}")

    if emit_output:
        print(f"\nGate: {'FAIL' if gate_failed else 'PASS'}")

    return {
        "selected_retrievers": selected,
        "failures": failures,
        "gate_status": "FAIL" if gate_failed else "PASS",
        "metrics": metrics,
        "metrics_path": metrics_path,
        "report_path": report_path,
        "excluded_non_kb_cases": excluded_non_kb_rows,
    }


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
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write data/reports/metrics.json and readiness_report.md",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Report output directory (default: data/reports)",
    )
    args = parser.parse_args()

    selected = [name.strip() for name in args.retrievers.split(",") if name.strip()]
    try:
        result = run_evaluation(
            retrievers=selected,
            eval_path=args.eval_set,
            index_path=args.index,
            top_k=args.top_k,
            model_name=args.model,
            cache_dir=args.embedding_cache,
            limit=args.limit,
            write_report=args.write_report,
            report_dir=args.report_dir,
            emit_output=True,
        )
    except ValueError as error:
        parser.error(str(error))

    if result["gate_status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
