"""Optional Streamlit demo for the three implemented product modes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from eval.run_eval import run_evaluation
from src.answer import AnswerResult, classify_query
from src.audit import CORE_METRICS, METRIC_LABELS
from src.compare import compare_documents, write_reports as write_change_reports
from src.document_loader import (
    DocumentParseError,
    UnsupportedDocumentTypeError,
    detect_document_type,
)
from src.ingest import DEFAULT_INDEX_PATH, PROJECT_ROOT
from src.session import AnswerSession


LARGE_UPLOAD_BYTES = 10 * 1024 * 1024
LARGE_PDF_DEMO = (
    PROJECT_ROOT / "compare_docs" / "large_old_refund_policy.pdf",
    PROJECT_ROOT / "compare_docs" / "large_new_refund_policy.pdf",
)
MARKDOWN_DEMO = (
    PROJECT_ROOT / "compare_docs" / "old_refund_policy.md",
    PROJECT_ROOT / "compare_docs" / "new_refund_policy.md",
)


def _ratio(metric: dict[str, int]) -> str:
    passed = metric["passed"]
    total = metric["total"]
    return f"{passed}/{total} ({passed / total:.1%})" if total else "0/0"


def _render_answer(result: AnswerResult) -> None:
    if result.response_type == "out_of_scope_general":
        st.info(result.answer)
        return

    if result.response_type == "non_kb_chitchat":
        st.write(result.answer)
        return

    st.subheader("Validator verdict")
    if result.answer_mode == "extractive":
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

    compared = result["compared_documents"]
    old_name = compared.get("old_name") or Path(compared["old"]).name
    new_name = compared.get("new_name") or Path(compared["new"]).name

    st.subheader("Compared documents")
    st.write(f"**Old:** `{old_name}`  \n**New:** `{new_name}`")

    status_columns = st.columns(5)
    status_columns[0].metric("Changed sections", summary["changed_sections"])
    status_columns[1].metric("High risk", severity["high"])
    status_columns[2].metric("Medium risk", severity["medium"])
    status_columns[3].metric("Low risk", severity["low"])
    status_columns[4].metric(
        "Human review",
        "Required" if summary["human_review_required"] else "Not required",
    )

    st.subheader("Executive summary")
    st.write(
        f"Detected {summary['changed_sections']} changed sections: "
        f"{severity['high']} high, {severity['medium']} medium, and "
        f"{severity['low']} low risk. {summary['impacted_eval_cases']} eval cases "
        f"and {summary['required_kb_updates']} KB sections require review."
    )

    st.subheader("Changes by risk")
    for risk in ("high", "medium", "low"):
        risk_changes = [
            change for change in result["changes"] if change["severity"] == risk
        ]
        with st.expander(
            f"{risk.title()} risk ({len(risk_changes)})", expanded=risk == "high"
        ):
            if not risk_changes:
                st.caption("None detected by the configured rules.")
            for change in risk_changes:
                page_note = ""
                if change.get("old_page") is not None or change.get("new_page") is not None:
                    page_note = (
                        f" · pages {change.get('old_page') or 'none'} → "
                        f"{change.get('new_page') or 'none'}"
                    )
                st.markdown(f"**{change['id']}: {change['section']}**{page_note}")
                st.write("; ".join(change["severity_reasons"]))
                st.caption("Signals: " + ", ".join(change["change_types"]))
                if change["changed_values"]["changed"]:
                    st.code(
                        f"{', '.join(change['changed_values']['old']) or 'none'} -> "
                        f"{', '.join(change['changed_values']['new']) or 'none'}"
                    )

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
    if summary["human_review_required"]:
        st.warning("Human review is required before affected answers are released.")
    else:
        st.info("No high-risk rule triggered mandatory human review.")
    for recommendation in result["human_review_recommendations"]:
        st.markdown(f"- {recommendation}")

    processing = result.get("document_processing", {})
    if any(item.get("document_type") == "pdf" for item in processing.values()):
        st.info(
            "PDFs were parsed page by page into page-bounded section chunks. "
            "The full PDF was not loaded as one context."
        )

    st.subheader("Generated reports")
    st.code(f"JSON: {run['json_path']}\nMarkdown: {run['report_path']}")
    report_path = run["report_path"]
    if report_path.exists():
        with st.expander("Rendered change impact report"):
            st.markdown(report_path.read_text(encoding="utf-8"))
    download_columns = st.columns(2)
    json_path = run["json_path"]
    if json_path.exists():
        download_columns[0].download_button(
            "Download JSON report",
            data=json_path.read_bytes(),
            file_name=json_path.name,
            mime="application/json",
            key=f"download_change_json_{run['run_id']}",
        )
    if report_path.exists():
        download_columns[1].download_button(
            "Download Markdown report",
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime="text/markdown",
            key=f"download_change_markdown_{run['run_id']}",
        )


st.set_page_config(page_title="KB Readiness Agent", layout="wide")
st.title("KB Readiness Agent")
st.caption(
    "Local deterministic demo: validator-gated Ask Mode, readiness evaluation, "
    "and section-based Markdown, text, or PDF policy change impact."
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
    source_mode = st.radio(
        "Document source",
        ("Built-in demo files", "Upload custom documents"),
        horizontal=True,
        key="change_source_mode",
    )
    old_path: Path | None = None
    new_path: Path | None = None
    old_upload = None
    new_upload = None
    selection_id = source_mode

    if source_mode == "Built-in demo files":
        demo_choice = st.selectbox(
            "Demo document pair",
            (
                "Large PDF refund policy (50 pages)",
                "Markdown refund policy",
            ),
            key="change_demo_choice",
        )
        selection_id = f"{source_mode}:{demo_choice}"
        if demo_choice.startswith("Large PDF"):
            old_path, new_path = LARGE_PDF_DEMO
            st.info(
                "Recommended large-document demo. PDFs are parsed page by page and "
                "compared as page-bounded section chunks; processing may take time."
            )
        else:
            old_path, new_path = MARKDOWN_DEMO
        st.caption(f"Old: `{old_path.name}` · New: `{new_path.name}`")
    else:
        upload_columns = st.columns(2)
        old_upload = upload_columns[0].file_uploader(
            "Old document",
            type=["pdf", "md", "txt"],
            key="old_change_upload",
        )
        new_upload = upload_columns[1].file_uploader(
            "New document",
            type=["pdf", "md", "txt"],
            key="new_change_upload",
        )
        if (old_upload is None) != (new_upload is None):
            st.warning(
                "Upload both the old document and the new document before running analysis."
            )
        uploads = [item for item in (old_upload, new_upload) if item is not None]
        for upload in uploads:
            try:
                detect_document_type(upload.name)
            except UnsupportedDocumentTypeError as error:
                st.error(str(error))
        if any(upload.size >= LARGE_UPLOAD_BYTES for upload in uploads):
            st.info("One or both documents are very large. Processing may take time.")
        if any(Path(upload.name).suffix.casefold() == ".pdf" for upload in uploads):
            st.caption(
                "PDF text is extracted page by page. Repeated headers and footers are "
                "removed when detected, and page numbers are preserved on comparison chunks."
            )

    run_clicked = st.button(
        "Run change impact analysis", type="primary", key="change_button"
    )
    if run_clicked:
        selected_types: set[str] = set()
        try:
            if source_mode == "Built-in demo files":
                assert old_path is not None and new_path is not None
                if old_path == LARGE_PDF_DEMO[0] and (
                    not old_path.is_file() or not new_path.is_file()
                ):
                    from scripts.build_large_pdf_fixture import build_large_pdf_fixtures

                    with st.spinner("Generating the built-in 50-page PDF demo..."):
                        build_large_pdf_fixtures(old_path, new_path, pages=50)
                if not old_path.is_file() or not new_path.is_file():
                    raise DocumentParseError(
                        "The selected built-in demo files are missing or unreadable."
                    )
                selected_types = {
                    detect_document_type(old_path),
                    detect_document_type(new_path),
                }
                with st.spinner("Comparing policy sections and mapping impact..."):
                    change_result = compare_documents(old_path, new_path)
            else:
                if old_upload is None or new_upload is None:
                    st.warning(
                        "Upload both the old document and the new document before running analysis."
                    )
                    change_result = None
                else:
                    old_type = detect_document_type(old_upload.name)
                    new_type = detect_document_type(new_upload.name)
                    selected_types = {old_type, new_type}
                    # Uploaded bytes exist only under the operating system's temporary
                    # directory and are deleted immediately after comparison.
                    with tempfile.TemporaryDirectory(
                        prefix="kb-change-impact-"
                    ) as runtime_dir:
                        runtime_root = Path(runtime_dir)
                        old_path = runtime_root / "old" / Path(old_upload.name).name
                        new_path = runtime_root / "new" / Path(new_upload.name).name
                        old_path.parent.mkdir(parents=True)
                        new_path.parent.mkdir(parents=True)
                        old_path.write_bytes(old_upload.getvalue())
                        new_path.write_bytes(new_upload.getvalue())
                        with st.spinner(
                            "Parsing documents and comparing section/chunk impact..."
                        ):
                            change_result = compare_documents(old_path, new_path)
                        # Reports should expose user-facing names, not deleted temp paths.
                        change_result["compared_documents"].update(
                            {
                                "old": Path(old_upload.name).name,
                                "new": Path(new_upload.name).name,
                                "old_name": Path(old_upload.name).name,
                                "new_name": Path(new_upload.name).name,
                            }
                        )

            if change_result is not None:
                json_path, report_path = write_change_reports(change_result)
                run_id = st.session_state.get("change_run_counter", 0) + 1
                st.session_state["change_run_counter"] = run_id
                st.session_state["change_run"] = {
                    "result": change_result,
                    "json_path": json_path,
                    "report_path": report_path,
                    "run_id": run_id,
                    "selection_id": selection_id,
                }
        except UnsupportedDocumentTypeError as error:
            st.error(f"Unsupported file type: {error}")
        except DocumentParseError as error:
            prefix = (
                "PDF parse error" if "pdf" in selected_types else "Document parse error"
            )
            st.error(f"{prefix}: {error}")
        except Exception as error:
            st.error(f"Change impact analysis failed: {error}")

    change_run = st.session_state.get("change_run")
    if change_run and change_run.get("selection_id") == selection_id:
        _render_change_impact(st.session_state["change_run"])
