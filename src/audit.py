"""Build machine-readable and human-readable Knowledge Base readiness reports."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .answer import AnswerResult
from .ingest import PROJECT_ROOT
from .retrieve import SearchResult


DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "reports"
METRICS_FILENAME = "metrics.json"
REPORT_FILENAME = "readiness_report.md"
P2_EVALUATION_SCOPE = "p2_change_impact"

CORE_METRICS = (
    "source_hit_at_k",
    "section_hit_at_k",
    "correct_refusal",
    "citation_coverage",
    "groundedness",
)
CHECK_TO_METRIC = {
    "source_hit": "source_hit_at_k",
    "section_hit": "section_hit_at_k",
    "correct_refusal": "correct_refusal",
    "citation_coverage": "citation_coverage",
    "groundedness_pass": "groundedness",
}
METRIC_LABELS = {
    "source_hit_at_k": "source_hit@k",
    "section_hit_at_k": "section_hit@k",
    "correct_refusal": "correct refusal",
    "citation_coverage": "citation coverage",
    "groundedness": "groundedness",
}
GAP_TOPICS_BY_REASON = {
    "refund_exception": "Refund exception / hardship policy",
    "sla": "Signed Enterprise SLA",
    "privacy": "Regional privacy legal advice",
    "pricing": "Enterprise pricing",
    "insufficient_relevance": "Missing or undiscoverable policy coverage",
}
GAP_TOPICS_BY_SECTION = {
    "unsupported_exceptions": "Refund exception / hardship policy",
    "enterprise_support_response_time": "Signed Enterprise SLA",
    "unsupported_privacy_questions": "Regional privacy legal advice",
    "enterprise_pricing": "Enterprise pricing",
}
KNOWN_LIMITATIONS = (
    "The corpus is synthetic and small: 6 Markdown documents and 34 chunks.",
    "The eval set is small and curated, so passing results are not statistically representative of production traffic.",
    "Groundedness is checked with deterministic citation, provenance, numeric-claim, and refusal rules; it is not semantic answer evaluation.",
    "Citations are chunk-level, not sentence-level, and Markdown sources do not provide page numbers.",
    "Retrieval thresholds and hybrid fusion weights are calibrated only against the current local corpus and eval set.",
)


def load_eval_cases(path: Path) -> list[dict]:
    """Load JSONL evaluation cases."""

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def split_eval_cases(rows: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Separate active Ask Mode cases from reserved P2 change-impact cases."""

    active: list[dict] = []
    excluded: list[dict] = []
    for row in rows:
        if row.get("evaluation_scope") == P2_EVALUATION_SCOPE:
            excluded.append(row)
        else:
            active.append(row)
    return active, excluded


def build_case_evaluation(
    row: dict,
    *,
    language: str,
    question: str,
    retrieved: list[SearchResult],
    answer: AnswerResult,
) -> dict:
    """Normalize one eval execution for aggregation and failure reporting."""

    expected_sources = sorted(set(row.get("source_docs", [row["source_doc"]])))
    checks = {
        "source_hit": any(item["doc"] in expected_sources for item in retrieved),
        "section_hit": any(
            item["section_slug"] == row["source_section"] for item in retrieved
        ),
        "correct_refusal": answer.refused == (not row["answerable"]),
        "citation_coverage": bool(answer.citations),
        "groundedness_pass": answer.groundedness["status"] == "supported",
    }
    top = retrieved[0] if retrieved else None
    return {
        "case_id": row["id"],
        "language": language,
        "question": question,
        "answerable": row["answerable"],
        "expected_sources": expected_sources,
        "expected_section": row["source_section"],
        "top_retrieved": (
            {
                "doc": top["doc"],
                "section": top["section"],
                "section_slug": top["section_slug"],
                "score": round(float(top["score"]), 6),
            }
            if top
            else None
        ),
        "refused": answer.refused,
        "refusal_reason": answer.refusal_reason,
        "requires_human_review": answer.requires_human_review,
        "citations": [
            {
                "doc": citation["doc"],
                "section": citation["section"],
                "section_slug": citation["section_slug"],
                "chunk_id": citation["chunk_id"],
            }
            for citation in answer.citations
        ],
        "groundedness_status": answer.groundedness["status"],
        "warnings": list(answer.warnings),
        "checks": checks,
    }


def _failure(record: dict) -> dict | None:
    failed = [name for name, passed in record["checks"].items() if not passed]
    if not failed:
        return None
    labels = [METRIC_LABELS[CHECK_TO_METRIC[name]] for name in failed]
    return {
        "case_id": record["case_id"],
        "language": record["language"],
        "question": record["question"],
        "expected_sources": record["expected_sources"],
        "expected_section": record["expected_section"],
        "top_retrieved": record["top_retrieved"],
        "refused": record["refused"],
        "citations": record["citations"],
        "groundedness_status": record["groundedness_status"],
        "failed_metrics": [CHECK_TO_METRIC[name] for name in failed],
        "reason": "Failed " + ", ".join(labels) + ".",
    }


def _gap_topic(record: dict) -> str:
    reason = record.get("refusal_reason")
    if reason in GAP_TOPICS_BY_REASON:
        return GAP_TOPICS_BY_REASON[reason]
    section = record.get("expected_section")
    if section in GAP_TOPICS_BY_SECTION:
        return GAP_TOPICS_BY_SECTION[section]
    return f"Policy coverage for: {record['question']}"


def extract_knowledge_gaps(records: Iterable[dict]) -> list[dict]:
    """Derive deterministic, de-duplicated gaps from refusals and unanswerable cases."""

    grouped: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {
            "case_ids": set(),
            "questions": set(),
            "refusal_reasons": set(),
            "evidence_sections": set(),
        }
    )
    for record in records:
        if record["answerable"] and not record["refused"]:
            continue
        topic = _gap_topic(record)
        gap = grouped[topic]
        gap["case_ids"].add(record["case_id"])
        gap["questions"].add(record["question"])
        if record.get("refusal_reason"):
            gap["refusal_reasons"].add(record["refusal_reason"])
        gap["evidence_sections"].update(
            citation["section_slug"] for citation in record["citations"]
        )

    return [
        {
            "topic": topic,
            "case_ids": sorted(values["case_ids"]),
            "questions": sorted(values["questions"]),
            "refusal_reasons": sorted(values["refusal_reasons"]),
            "evidence_sections": sorted(values["evidence_sections"]),
        }
        for topic, values in sorted(grouped.items())
    ]


def build_metrics(
    *,
    retriever: str,
    top_k: int,
    all_cases: list[dict],
    records: list[dict],
    small_synthetic_corpus: bool = True,
) -> dict:
    """Aggregate case executions into the stable Day 4 metrics schema."""

    active, excluded = split_eval_cases(all_cases)
    languages: dict[str, dict] = {}
    for language in sorted({record["language"] for record in records}):
        language_records = [
            record for record in records if record["language"] == language
        ]
        language_metrics: dict[str, dict[str, int]] = {}
        for check_name, metric_name in CHECK_TO_METRIC.items():
            values = [record["checks"][check_name] for record in language_records]
            language_metrics[metric_name] = {
                "passed": sum(values),
                "total": len(values),
            }
        languages[language] = language_metrics

    failures = [failure for record in records if (failure := _failure(record))]
    executed_case_ids = {record["case_id"] for record in records}
    expected_languages = {
        language
        for row in active
        for language, field in (("zh", "question"), ("en", "question_en"))
        if row.get(field)
    }
    expected_counts = {
        language: sum(
            bool(row.get(field))
            for row in active
        )
        for language, field in (("zh", "question"), ("en", "question_en"))
        if any(row.get(field) for row in active)
    }
    evaluation_complete = (
        executed_case_ids == {row["id"] for row in active}
        and set(languages) == expected_languages
        and all(
            metric["total"] == expected_counts[language]
            for language, language_metrics in languages.items()
            for metric in language_metrics.values()
        )
    )
    gate_passed = evaluation_complete and not failures

    human_review_warnings = [
        {
            "case_id": record["case_id"],
            "language": record["language"],
            "warnings": record["warnings"],
        }
        for record in records
        if record["warnings"]
        or (record["requires_human_review"] and not record["refused"])
    ]
    if not gate_passed:
        recommendation = "Not Ready"
    elif small_synthetic_corpus or human_review_warnings:
        recommendation = "Internal Pilot Ready"
    else:
        recommendation = "External Ready"

    return {
        "schema_version": 1,
        "retriever": retriever,
        "top_k": top_k,
        "total_cases": len(all_cases),
        "active_cases": len(active),
        "executed_active_cases": len(executed_case_ids),
        "excluded_p2_cases": len(excluded),
        "evaluation_complete": evaluation_complete,
        "languages": languages,
        "gate": {
            "status": "PASS" if gate_passed else "FAIL",
            "recommendation": recommendation,
        },
        "human_review_warnings": human_review_warnings,
        "failures": failures,
        "knowledge_gaps": extract_knowledge_gaps(records),
        "known_limitations": list(KNOWN_LIMITATIONS),
    }


def _ratio(metric: dict[str, int]) -> str:
    passed = metric["passed"]
    total = metric["total"]
    percentage = passed / total if total else 0.0
    return f"{passed}/{total} ({percentage:.1%})"


def render_markdown(metrics: dict) -> str:
    """Render a readiness report from a metrics payload."""

    gate = metrics["gate"]
    lines = [
        "# Knowledge Base Readiness Report",
        "",
        "## Executive summary",
        "",
        f"The active Ask Mode evaluation gate is **{gate['status']}** using the "
        f"`{metrics['retriever']}` retriever at top-k={metrics['top_k']}. "
        f"Launch recommendation: **{gate['recommendation']}**.",
        "",
        "## Scope",
        "",
        f"- Total eval cases: {metrics['total_cases']}",
        f"- Active Ask Mode cases: {metrics['active_cases']}",
        f"- Executed active cases: {metrics['executed_active_cases']}",
        f"- Excluded P2 conflict/change cases: {metrics['excluded_p2_cases']}",
        "- Languages: " + ", ".join(sorted(metrics["languages"])),
        "",
        "P2 conflict/change cases are reserved for future Change Impact Mode and are not part of this Ask Mode gate.",
        "",
        "## Gate status",
        "",
        f"**{gate['status']}**",
        "",
        "The gate passes only when every core metric passes for every available language and the full active eval set is executed.",
        "",
        "## Launch recommendation",
        "",
        f"**{gate['recommendation']}**",
        "",
    ]
    if gate["recommendation"] == "Internal Pilot Ready":
        lines.append(
            "Core checks pass, but the small synthetic corpus and current validation limits do not support an external-readiness claim."
        )
    elif gate["recommendation"] == "Not Ready":
        lines.append(
            "At least one core gate requirement failed or the active evaluation was incomplete."
        )
    else:
        lines.append("All core checks pass without outstanding human-review warnings.")

    lines.extend(("", "## Metrics", ""))
    report_labels = dict(METRIC_LABELS)
    report_labels["source_hit_at_k"] = f"source_hit@{metrics['top_k']}"
    report_labels["section_hit_at_k"] = f"section_hit@{metrics['top_k']}"
    header = ["Language"] + [report_labels[name] for name in CORE_METRICS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for language, values in sorted(metrics["languages"].items()):
        row = [language] + [_ratio(values[name]) for name in CORE_METRICS]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(("", "## Refusal quality", ""))
    for language, values in sorted(metrics["languages"].items()):
        lines.append(
            f"- {language}: correct refusal {_ratio(values['correct_refusal'])}"
        )

    lines.extend(("", "## Citation and groundedness quality", ""))
    for language, values in sorted(metrics["languages"].items()):
        lines.append(
            f"- {language}: citation coverage {_ratio(values['citation_coverage'])}; "
            f"groundedness {_ratio(values['groundedness'])}"
        )

    lines.extend(("", "## Knowledge gaps", ""))
    if metrics["knowledge_gaps"]:
        for gap in metrics["knowledge_gaps"]:
            case_ids = ", ".join(gap["case_ids"])
            sections = ", ".join(gap["evidence_sections"]) or "none"
            lines.append(
                f"- **{gap['topic']}** — cases: {case_ids}; evidence sections: {sections}"
            )
    else:
        lines.append("- None identified by the active eval set.")

    lines.extend(("", "## Failure cases", ""))
    if metrics["failures"]:
        for failure in metrics["failures"]:
            top = failure["top_retrieved"] or {}
            top_label = (
                f"{top.get('doc', 'none')} / {top.get('section_slug', 'none')}"
            )
            citations = ", ".join(
                f"{item['doc']} / {item['section_slug']}"
                for item in failure["citations"]
            ) or "none"
            lines.extend(
                (
                    f"### {failure['case_id']} ({failure['language']})",
                    "",
                    f"- Question: {failure['question']}",
                    f"- Expected: {', '.join(failure['expected_sources'])} / {failure['expected_section']}",
                    f"- Top retrieved: {top_label}",
                    f"- Refused: {'yes' if failure['refused'] else 'no'}",
                    f"- Citations: {citations}",
                    f"- Groundedness: {failure['groundedness_status']}",
                    f"- Reason: {failure['reason']}",
                    "",
                )
            )
    else:
        lines.append("- None.")

    lines.extend(("", "## Known limitations", ""))
    lines.extend(f"- {item}" for item in metrics["known_limitations"])
    lines.extend(
        (
            "",
            "## Next steps",
            "",
            "- Review and prioritize the deterministic knowledge gaps with policy owners.",
            "- Add representative real-world corpus content and eval cases before external use.",
            "- Keep P2 conflict/change cases isolated until Change Impact Mode is implemented.",
            "- Re-run this gate after corpus, retrieval, or answer-contract changes.",
            "",
        )
    )
    return "\n".join(lines)


def write_reports(metrics: dict, output_dir: Path = DEFAULT_REPORT_DIR) -> tuple[Path, Path]:
    """Write metrics JSON and Markdown report, returning both paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / METRICS_FILENAME
    report_path = output_dir / REPORT_FILENAME
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(metrics), encoding="utf-8")
    return metrics_path, report_path
