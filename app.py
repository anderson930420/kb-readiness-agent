"""Optional Streamlit demo for the three implemented product modes."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from eval.run_eval import run_evaluation
from src.answer import AnswerResult, classify_query
from src.audit import CORE_METRICS, METRIC_LABELS
from src.compare import compare_documents, write_reports as write_change_reports
from src.ingest import DEFAULT_INDEX_PATH, PROJECT_ROOT
from src.session import AnswerSession


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _ratio(metric: dict[str, int]) -> str:
    passed = metric["passed"]
    total = metric["total"]
    return f"{passed}/{total} ({passed / total:.1%})" if total else "0/0"


def _render_answer(result: AnswerResult) -> None:
    st.subheader("Validator verdict")
    if result.response_type == "out_of_scope_general":
        st.markdown(
            '<span style="display:inline-block;padding:0.2rem 0.55rem;'
            'border-radius:0.5rem;background:#e5e7eb;color:#374151;'
            'font-size:0.85rem;font-weight:600">Out of scope</span>',
            unsafe_allow_html=True,
        )
    elif result.response_type == "non_kb_chitchat":
        st.info("Not run — deterministic non-KB response")
    elif result.answer_mode == "extractive":
        st.info("Not run — extractive answer mode")
    elif result.validator_decision == "allowed":
        st.success("✓ Generated answer supported and safe")
        if result.requires_human_review:
            st.warning("Requires human review")
    elif result.validator_decision == "blocked":
        st.error("⚠ Generated proposal blocked by validator")
        st.warning("Requires human review")
    else:
        st.warning(f"Unexpected validator decision: {result.validator_decision}")

    st.subheader("Answer")
    st.write(result.answer)

    if result.blocked_generated_answer is not None:
        st.error(
            "**Blocked generated answer — not shown to end users**\n\n"
            f"{result.blocked_generated_answer}"
        )

    if result.generation_trace is not None:
        with st.expander("Generation trace"):
            st.json(result.generation_trace)

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
    elif groundedness == "not_applicable":
        st.info("Groundedness: not applicable to non-KB response")
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
    if not result.retrieved_chunks:
        st.caption("None")
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
    "Local deterministic demo: validator-gated Ask Mode, readiness evaluation, "
    "and Markdown policy change impact."
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
    answer_mode = st.selectbox(
        "Answer mode", ("extractive", "generative"), index=0
    )
    llm_provider = None
    minimax_key_missing = False
    if answer_mode == "generative":
        llm_provider = st.selectbox(
            "LLM provider", ("fake_hallucination", "minimax"), index=0
        )
        st.caption(
            "`fake_hallucination` is the no-key reproducible validator demo.  \n"
            "`minimax` requires `MINIMAX_API_KEY`."
        )
        minimax_key_missing = (
            llm_provider == "minimax"
            and not os.environ.get("MINIMAX_API_KEY", "").strip()
        )
        if minimax_key_missing:
            st.warning(
                "MiniMax requires `MINIMAX_API_KEY` for KB questions. Set it in "
                "the Streamlit environment or select `fake_hallucination`; no "
                "request was made."
            )
    response_type = classify_query(question)
    canned_response = response_type != "kb_answer"
    ask_column, clear_column = st.columns([1, 1])
    ask_clicked = ask_column.button(
        "Ask",
        type="primary",
        key="ask_button",
        disabled=minimax_key_missing and not canned_response,
    )
    clear_clicked = clear_column.button("Clear session", key="clear_session_button")
    if clear_clicked:
        st.session_state.pop("answer_session", None)
        st.session_state.pop("answer_result", None)
        st.session_state.pop("question_resolution", None)
    if ask_clicked:
        if not canned_response and not DEFAULT_INDEX_PATH.exists():
            st.error("Index not found. Run `python -m src.ingest` first.")
        else:
            try:
                spinner_text = (
                    "Preparing deterministic response..."
                    if canned_response
                    else "Retrieving evidence and checking groundedness..."
                )
                with st.spinner(spinner_text):
                    session = st.session_state.get("answer_session")
                    if (
                        session is None
                        or session.retriever != retriever
                        or session.mode != answer_mode
                        or session.llm_provider != llm_provider
                    ):
                        session = AnswerSession(
                            retriever=retriever,
                            mode=answer_mode,
                            llm_provider=llm_provider,
                        )
                        st.session_state["answer_session"] = session
                    turn = session.ask(question.strip())
                    st.session_state["answer_result"] = turn.answer
                    st.session_state["question_resolution"] = turn
            except Exception as error:
                st.error(f"Ask Mode failed: {error}")
    if "answer_result" in st.session_state:
        turn = st.session_state.get("question_resolution")
        if turn and turn.original_question != turn.resolved_question:
            st.caption(f"Resolved follow-up: {turn.resolved_question}")
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
            st.error("Both policy paths must point to readable Markdown or PDF files.")
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
