"""Optional Streamlit demo for the three implemented product modes."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from eval.run_eval import run_evaluation
from src.answer import AnswerResult, answer_question
from src.audit import CORE_METRICS, METRIC_LABELS
from src.compare import compare_documents, write_reports as write_change_reports
from src.ingest import DEFAULT_INDEX_PATH, PROJECT_ROOT


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _ratio(metric: dict[str, int]) -> str:
    passed = metric["passed"]
    total = metric["total"]
    return f"{passed}/{total} ({passed / total:.1%})" if total else "0/0"


def _render_answer(result: AnswerResult) -> None:
    st.subheader("Answer")
    st.write(result.answer)

    status_columns = st.columns(4)
    status_columns[0].metric("Refused", "Yes" if result.refused else "No")
    status_columns[1].metric("Confidence", result.confidence.title())
    status_columns[2].metric(
        "Human review", "Required" if result.requires_human_review else "Not required"
    )
    status_columns[3].metric("Latency", f"{result.latency_ms:.1f} ms")

    groundedness = result.groundedness["status"]
    if groundedness == "supported":
        st.success("Groundedness: supported")
    else:
        st.error("Groundedness: unsupported")
    with st.expander("Groundedness checks"):
        st.json(result.groundedness)

    st.subheader("Warnings")
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)
    else:
        st.caption("None")

    st.subheader("Citations")
    if not result.citations:
        st.caption("None")
    for citation in result.citations:
        st.markdown(
            f"**{citation['doc']} / {citation['section']} / "
            f"`{citation['chunk_id']}`**"
        )
        st.caption(f"Section slug: {citation['section_slug']}")
        st.write(citation["text"])

    st.subheader("Retrieved chunks")
    for rank, chunk in enumerate(result.retrieved_chunks, start=1):
        label = (
            f"{rank}. {chunk['doc']} / {chunk['section_slug']} "
            f"(score {chunk['score']:.4f})"
        )
        with st.expander(label):
            st.caption(
                f"chunk_id: {chunk['chunk_id']} · retriever: "
                f"{chunk['retrieval_method']}"
            )
            st.write(chunk["text"])


def _render_readiness(run: dict) -> None:
    metrics = run["metrics"]
    if metrics is None:
        st.error("The audit did not produce metrics.")
        return

    gate = metrics["gate"]
    status_columns = st.columns(2)
    status_columns[0].metric("Gate status", gate["status"])
    status_columns[1].metric("Launch recommendation", gate["recommendation"])

    rows = []
    labels = dict(METRIC_LABELS)
    labels["source_hit_at_k"] = f"source_hit@{metrics['top_k']}"
    labels["section_hit_at_k"] = f"section_hit@{metrics['top_k']}"
    for language, language_metrics in sorted(metrics["languages"].items()):
        row = {"Language": language}
        row.update(
            {
                labels[name]: _ratio(language_metrics[name])
                for name in CORE_METRICS
            }
        )
        rows.append(row)
    st.subheader("Metrics")
    st.table(rows)

    st.subheader("Knowledge gaps")
    if metrics["knowledge_gaps"]:
        gap_rows = [
            {
                "Topic": gap["topic"],
                "Cases": ", ".join(gap["case_ids"]),
                "Evidence sections": ", ".join(gap["evidence_sections"]) or "None",
            }
            for gap in metrics["knowledge_gaps"]
        ]
        st.table(gap_rows)
    else:
        st.caption("None identified by the active eval set.")

    metrics_path = run["metrics_path"]
    report_path = run["report_path"]
    st.subheader("Generated reports")
    st.code(f"Metrics: {metrics_path}\nMarkdown: {report_path}")
    if report_path and report_path.exists():
        report = report_path.read_text(encoding="utf-8")
        with st.expander("Rendered readiness report"):
            st.markdown(report)


def _render_change_impact(run: dict) -> None:
    result = run["result"]
    summary = result["summary"]
    severity = summary["severity"]

    st.subheader("Executive summary")
    st.write(
        f"Detected {summary['changed_sections']} changed sections: "
        f"{severity['high']} high, {severity['medium']} medium, and "
        f"{severity['low']} low risk. {summary['impacted_eval_cases']} eval cases "
        f"and {summary['required_kb_updates']} KB sections require review."
    )

    st.subheader("High-risk changes")
    high_risk = [change for change in result["changes"] if change["severity"] == "high"]
    if high_risk:
        for change in high_risk:
            with st.expander(f"{change['id']}: {change['section']}"):
                st.write("; ".join(change["severity_reasons"]))
                st.caption("Signals: " + ", ".join(change["change_types"]))
                if change["changed_values"]["changed"]:
                    st.code(
                        f"{', '.join(change['changed_values']['old']) or 'none'} -> "
                        f"{', '.join(change['changed_values']['new']) or 'none'}"
                    )
    else:
        st.caption("None detected by the configured rules.")

    st.subheader("Impacted eval cases")
    if result["impacted_eval_cases"]:
        st.dataframe(
            [
                {
                    "Case": case["case_id"],
                    "Question": case["question"],
                    "Expected section": case["expected_section"],
                    "Action": case["suggested_action"],
                }
                for case in result["impacted_eval_cases"]
            ],
            hide_index=True,
        )
    else:
        st.caption("None identified.")

    st.subheader("Required KB updates")
    if result["required_kb_updates"]:
        st.dataframe(
            [
                {
                    "Priority": update["priority"],
                    "File": update["file"],
                    "Section": update["section_slug"],
                    "Reason": update["reason"],
                }
                for update in result["required_kb_updates"]
            ],
            hide_index=True,
        )
    else:
        st.caption("No existing corpus target was identified.")

    st.subheader("Human review recommendations")
    for recommendation in result["human_review_recommendations"]:
        st.markdown(f"- {recommendation}")

    st.subheader("Generated reports")
    st.code(f"JSON: {run['json_path']}\nMarkdown: {run['report_path']}")
    report_path = run["report_path"]
    if report_path.exists():
        with st.expander("Rendered change impact report"):
            st.markdown(report_path.read_text(encoding="utf-8"))


st.set_page_config(page_title="KB Readiness Agent", layout="wide")
st.title("KB Readiness Agent")
st.caption(
    "Local deterministic demo: extractive Ask Mode, readiness evaluation, and "
    "Markdown policy change impact."
)

ask_tab, readiness_tab, change_tab = st.tabs(
    ["Ask", "Readiness Audit", "Change Impact"]
)

with ask_tab:
    question = st.text_input(
        "Support policy question",
        value="標準月付用戶的退款期限是多久？",
    )
    retriever = st.selectbox(
        "Retriever", ("lexical", "dense", "hybrid"), index=2
    )
    if st.button("Ask", type="primary", key="ask_button"):
        if not question.strip():
            st.warning("Enter a question.")
        elif not DEFAULT_INDEX_PATH.exists():
            st.error("Index not found. Run `python -m src.ingest` first.")
        else:
            try:
                with st.spinner("Retrieving evidence and checking groundedness..."):
                    st.session_state["answer_result"] = answer_question(
                        question.strip(), retriever=retriever
                    )
            except Exception as error:
                st.error(f"Ask Mode failed: {error}")
    if "answer_result" in st.session_state:
        _render_answer(st.session_state["answer_result"])

with readiness_tab:
    st.write("Runs the official hybrid Ask Mode gate and writes readiness artifacts.")
    if st.button(
        "Run readiness audit", type="primary", key="readiness_button"
    ):
        if not DEFAULT_INDEX_PATH.exists():
            st.error("Index not found. Run `python -m src.ingest` first.")
        else:
            try:
                with st.spinner("Running 40 bilingual eval queries..."):
                    st.session_state["readiness_run"] = run_evaluation(
                        retrievers=("hybrid",),
                        write_report=True,
                    )
            except Exception as error:
                st.error(f"Readiness audit failed: {error}")
    if "readiness_run" in st.session_state:
        _render_readiness(st.session_state["readiness_run"])

with change_tab:
    old_policy = st.text_input(
        "Old policy path",
        value="compare_docs/old_refund_policy.md",
        key="old_policy",
    )
    new_policy = st.text_input(
        "New policy path",
        value="compare_docs/new_refund_policy.md",
        key="new_policy",
    )
    if st.button(
        "Run change impact analysis", type="primary", key="change_button"
    ):
        old_path = _project_path(old_policy)
        new_path = _project_path(new_policy)
        if not old_path.is_file() or not new_path.is_file():
            st.error("Both policy paths must point to readable Markdown files.")
        else:
            try:
                with st.spinner("Comparing policy sections and mapping impact..."):
                    change_result = compare_documents(old_path, new_path)
                    json_path, report_path = write_change_reports(change_result)
                    st.session_state["change_run"] = {
                        "result": change_result,
                        "json_path": json_path,
                        "report_path": report_path,
                    }
            except Exception as error:
                st.error(f"Change impact analysis failed: {error}")
    if "change_run" in st.session_state:
        _render_change_impact(st.session_state["change_run"])
