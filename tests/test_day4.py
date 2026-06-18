from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.audit import (
    CORE_METRICS,
    build_metrics,
    load_eval_cases,
    render_markdown,
    split_eval_cases,
    write_reports,
)


EVAL_PATH = Path(__file__).resolve().parents[1] / "eval" / "eval_set.jsonl"
REFUSAL_REASONS = {
    "unsupported_exceptions": "refund_exception",
    "enterprise_support_response_time": "sla",
    "unsupported_privacy_questions": "privacy",
    "enterprise_pricing": "pricing",
}


def passing_records(rows: list[dict]) -> list[dict]:
    records: list[dict] = []
    for row in rows:
        for language, field in (("zh", "question"), ("en", "question_en")):
            question = row.get(field)
            if not question:
                continue
            refused = not row["answerable"]
            records.append(
                {
                    "case_id": row["id"],
                    "language": language,
                    "question": question,
                    "answerable": row["answerable"],
                    "expected_sources": row.get(
                        "source_docs", [row["source_doc"]]
                    ),
                    "expected_section": row["source_section"],
                    "top_retrieved": {
                        "doc": row["source_doc"],
                        "section": row["source_section_zh"],
                        "section_slug": row["source_section"],
                        "score": 1.0,
                    },
                    "refused": refused,
                    "refusal_reason": (
                        REFUSAL_REASONS[row["source_section"]] if refused else None
                    ),
                    "requires_human_review": refused,
                    "citations": [
                        {
                            "doc": row["source_doc"],
                            "section": row["source_section_zh"],
                            "section_slug": row["source_section"],
                            "chunk_id": f"fixture-{row['id']}",
                        }
                    ],
                    "groundedness_status": "supported",
                    "warnings": [],
                    "checks": {
                        "source_hit": True,
                        "section_hit": True,
                        "correct_refusal": True,
                        "citation_coverage": True,
                        "groundedness_pass": True,
                    },
                }
            )
    return records


class Day4ReadinessReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.all_cases = load_eval_cases(EVAL_PATH)
        cls.active_cases, cls.excluded_cases = split_eval_cases(cls.all_cases)
        cls.metrics = build_metrics(
            retriever="hybrid",
            top_k=3,
            all_cases=cls.all_cases,
            records=passing_records(cls.active_cases),
        )

    def test_p2_cases_are_excluded_from_active_ask_mode_gate(self) -> None:
        self.assertEqual(len(self.all_cases), 25)
        self.assertEqual(len(self.active_cases), 20)
        self.assertEqual(len(self.excluded_cases), 5)
        self.assertTrue(
            all(
                row["evaluation_scope"] == "p2_change_impact"
                for row in self.excluded_cases
            )
        )

    def test_metrics_schema_and_internal_pilot_recommendation(self) -> None:
        self.assertEqual(self.metrics["schema_version"], 1)
        self.assertEqual(self.metrics["retriever"], "hybrid")
        self.assertEqual(self.metrics["top_k"], 3)
        self.assertEqual(self.metrics["total_cases"], 25)
        self.assertEqual(self.metrics["active_cases"], 20)
        self.assertEqual(self.metrics["excluded_p2_cases"], 5)
        self.assertTrue(self.metrics["evaluation_complete"])
        self.assertEqual(set(self.metrics["languages"]), {"zh", "en"})
        for language in ("zh", "en"):
            self.assertEqual(
                set(self.metrics["languages"][language]), set(CORE_METRICS)
            )
            for metric in self.metrics["languages"][language].values():
                self.assertEqual(metric, {"passed": 20, "total": 20})
        self.assertEqual(self.metrics["gate"]["status"], "PASS")
        self.assertEqual(
            self.metrics["gate"]["recommendation"], "Internal Pilot Ready"
        )
        self.assertEqual(self.metrics["failures"], [])
        self.assertEqual(
            {gap["topic"] for gap in self.metrics["knowledge_gaps"]},
            {
                "Refund exception / hardship policy",
                "Signed Enterprise SLA",
                "Regional privacy legal advice",
                "Enterprise pricing",
            },
        )

    def test_markdown_report_has_required_readiness_sections_and_metrics(self) -> None:
        report = render_markdown(self.metrics)
        for expected in (
            "# Knowledge Base Readiness Report",
            "Active Ask Mode cases: 20",
            "Excluded P2 conflict/change cases: 5",
            "source_hit@3",
            "section_hit@3",
            "citation coverage",
            "groundedness",
            "## Refusal quality",
            "## Knowledge gaps",
            "## Known limitations",
            "Internal Pilot Ready",
        ):
            self.assertIn(expected, report)
        self.assertNotIn("**External Ready**", report)

    def test_report_files_are_written_as_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            metrics_path, report_path = write_reports(
                self.metrics, Path(temporary_directory)
            )
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            report = report_path.read_text(encoding="utf-8")

        self.assertEqual(metrics_path.name, "metrics.json")
        self.assertEqual(report_path.name, "readiness_report.md")
        self.assertEqual(payload["gate"]["recommendation"], "Internal Pilot Ready")
        self.assertIn("## Failure cases", report)

    def test_failure_payload_contains_diagnostic_context(self) -> None:
        records = passing_records(self.active_cases)
        records[0]["checks"]["source_hit"] = False
        metrics = build_metrics(
            retriever="hybrid",
            top_k=3,
            all_cases=self.all_cases,
            records=records,
        )
        failure = metrics["failures"][0]

        self.assertEqual(metrics["gate"]["status"], "FAIL")
        self.assertEqual(metrics["gate"]["recommendation"], "Not Ready")
        self.assertTrue(
            {
                "case_id",
                "language",
                "question",
                "expected_sources",
                "expected_section",
                "top_retrieved",
                "refused",
                "citations",
                "groundedness_status",
                "reason",
            }.issubset(failure)
        )


if __name__ == "__main__":
    unittest.main()
