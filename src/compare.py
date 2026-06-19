"""Deterministic Change Impact Mode for Markdown and PDF policy documents."""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Iterable

from .audit import load_eval_cases
from .document_loader import (
    _available_slug,
    load_document,
    parse_markdown_sections,
    parse_pdf_sections,
)
from .ingest import PROJECT_ROOT


DEFAULT_EVAL_PATH = PROJECT_ROOT / "eval" / "eval_set.jsonl"
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "corpus"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "reports"
JSON_FILENAME = "change_impact.json"
REPORT_FILENAME = "change_impact_report.md"

DISCLAIMER = (
    "This report identifies possible answer invalidation caused by policy changes; "
    "it does not claim full legal or semantic conflict detection."
)
KNOWN_LIMITATIONS = (
    "Markdown H1/H2 and visually distinct PDF headings are analyzed; tables, scans/OCR, attachments, and ambiguous heading layouts are not interpreted.",
    "Section alignment uses slugs, normalized heading similarity, and lexical overlap rather than semantic embeddings or an LLM.",
    "Change detection is based on explicit numeric and policy-language signals and can miss paraphrases or legal nuance.",
    "Impacted eval cases and KB updates are conservative deterministic suggestions and require policy-owner validation.",
    "The analysis compares only the two supplied documents and does not perform full-corpus conflict scanning.",
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s*|\n+")
VALUE_RE = re.compile(
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
    r"|\d+(?:\.\d+)?\s*(?:至|到|[-–—]|to)?\s*\d*(?:\.\d+)?\s*"
    r"(?:個?工作天|營業日|天|日|週|周|月|年|小時|分鐘|%|business\s+days?|days?|hours?|minutes?|weeks?|months?|years?)?",
    re.IGNORECASE,
)
TIME_UNIT_RE = re.compile(
    r"工作天|營業日|天|日|週|周|月|年|小時|分鐘|business\s+days?|days?|hours?|minutes?|weeks?|months?|years?",
    re.IGNORECASE,
)

REFUND_TERMS = ("退款", "refund", "refundable")
NON_REFUNDABLE_TERMS = (
    "不予退款",
    "不得退款",
    "不可退款",
    "non-refundable",
    "nonrefundable",
    "not refundable",
)
MANUAL_REVIEW_TERMS = ("人工審查", "人工複核", "manual review", "human review")
AUTOMATIC_TERMS = ("自動退款", "automatic refund", "automated refund")
EXCEPTION_TERMS = (
    "例外",
    "除非",
    "另有要求",
    "exception",
    "unless",
    "except",
)
ELIGIBILITY_POSITIVE_TERMS = ("可於", "可以退款", "可由", "may request", "refundable")
ELIGIBILITY_RESTRICTIVE_TERMS = NON_REFUNDABLE_TERMS + (
    "不得承諾",
    "必須",
    "原則上",
    "must not",
    "must be",
    "required",
    "only if",
)
PROCESSING_TERMS = ("處理時間", "完成處理", "processing time", "processed within")
ESCALATION_TERMS = (
    "人工審查",
    "人工複核",
    "轉交",
    "客戶成功",
    "帳戶管理",
    "escalat",
    "manual review",
    "account management",
    "customer success",
)

# Related sections are intentionally explicit. They model known duplicate/support
# surfaces without searching the full corpus for conflicts.
RELATED_EVAL_SECTIONS: dict[str, set[str]] = {
    "standard_refund_window": {"standard_refund_window"},
    "renewal_payments": {"renewal_payments"},
    "enterprise_refunds": {"enterprise_refunds", "enterprise_automatic_refunds"},
    "digital_services": {"digital_services", "onboarding_fees"},
    "refund_processing_time": {"refund_processing_time"},
    "unsupported_exceptions": {"unsupported_exceptions", "refund_escalation"},
}
SECTION_IMPACT_TERMS: dict[str, tuple[str, ...]] = {
    "standard_refund_window": (
        "月付退款",
        "月付用戶",
        "monthly refunds",
        "monthly subscribers",
        "annual subscribers",
    ),
    "renewal_payments": ("續約", "renewal"),
    "enterprise_refunds": ("enterprise", "自動退款", "automatic refund"),
    "digital_services": ("導入服務", "資料移轉", "onboarding services", "migration services"),
    "refund_processing_time": ("退款處理時間", "完成處理", "processing takes", "business days"),
    "unsupported_exceptions": ("醫療因素", "困難情境", "medical reasons", "hardship"),
}
RELATED_KB_SECTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "enterprise_refunds": (
        ("enterprise_plan_faq.md", "enterprise_automatic_refunds"),
        ("support_escalation_sop.md", "refund_escalation"),
    ),
    "digital_services": (("onboarding_guide.md", "onboarding_fees"),),
    "unsupported_exceptions": (("support_escalation_sop.md", "refund_escalation"),),
}


# Backward-compatible parser aliases remain available from src.compare.
parse_markdown = parse_markdown_sections
parse_pdf = parse_pdf_sections
parse_document = load_document


def _normalized_heading(section: dict) -> str:
    heading = section.get("section_zh") or section.get("section") or ""
    heading = re.sub(r"[（(].*?[）)]", " ", heading)
    return re.sub(r"[^\w\u3400-\u9fff]+", "", heading, flags=re.UNICODE).lower()


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    latin = set(re.findall(r"[a-z0-9_]+", lowered))
    cjk_sequences = re.findall(r"[\u3400-\u9fff]+", lowered)
    cjk: set[str] = set()
    for sequence in cjk_sequences:
        if len(sequence) == 1:
            cjk.add(sequence)
        else:
            cjk.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return latin | cjk


def _heading_similarity(old: dict, new: dict) -> float:
    left = _normalized_heading(old)
    right = _normalized_heading(new)
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def _lexical_overlap(old: dict, new: dict) -> float:
    left = _tokens(f"{old.get('section', '')} {old.get('text', '')}")
    right = _tokens(f"{new.get('section', '')} {new.get('text', '')}")
    return len(left & right) / len(left | right) if left and right else 0.0


def align_sections(old_sections: list[dict], new_sections: list[dict]) -> list[dict]:
    """Align sections one-to-one using deterministic descending-confidence rules."""

    matches: dict[int, tuple[int, str, float]] = {}
    used_new: set[int] = set()

    # Exact non-empty slug identity is the strongest signal.
    for old_index, old in enumerate(old_sections):
        slug = old.get("section_slug")
        if not slug:
            continue
        for new_index, new in enumerate(new_sections):
            if new_index not in used_new and slug == new.get("section_slug"):
                matches[old_index] = (new_index, "exact_slug", 1.0)
                used_new.add(new_index)
                break

    def assign_candidates(method: str, scorer, threshold: float) -> None:
        candidates: list[tuple[float, int, int]] = []
        for old_index, old in enumerate(old_sections):
            if old_index in matches:
                continue
            for new_index, new in enumerate(new_sections):
                if new_index in used_new:
                    continue
                score = scorer(old, new)
                if score >= threshold:
                    candidates.append((score, old_index, new_index))
        for score, old_index, new_index in sorted(
            candidates, key=lambda item: (-item[0], item[1], item[2])
        ):
            if old_index not in matches and new_index not in used_new:
                matches[old_index] = (new_index, method, round(score, 4))
                used_new.add(new_index)

    assign_candidates("heading_similarity", _heading_similarity, 0.72)
    assign_candidates("lexical_overlap", _lexical_overlap, 0.35)

    alignments: list[dict] = []
    for old_index, old in enumerate(old_sections):
        if old_index in matches:
            new_index, method, score = matches[old_index]
            new = new_sections[new_index]
            status = "unchanged" if old["text"] == new["text"] else "changed"
            alignments.append(
                {
                    "status": status,
                    "alignment_method": method,
                    "alignment_score": score,
                    "old": old,
                    "new": new,
                }
            )
        else:
            alignments.append(
                {
                    "status": "removed",
                    "alignment_method": "unmatched",
                    "alignment_score": 0.0,
                    "old": old,
                    "new": None,
                }
            )
    for new_index, new in enumerate(new_sections):
        if new_index not in used_new:
            alignments.append(
                {
                    "status": "added",
                    "alignment_method": "unmatched",
                    "alignment_score": 0.0,
                    "old": None,
                    "new": new,
                }
            )
    return alignments


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in SENTENCE_SPLIT_RE.split(text) if item.strip()]


def _text_delta(old_text: str, new_text: str) -> tuple[list[str], list[str]]:
    old_sentences = _sentences(old_text)
    new_sentences = _sentences(new_text)
    matcher = difflib.SequenceMatcher(None, old_sentences, new_sentences)
    removed: list[str] = []
    added: list[str] = []
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            removed.extend(old_sentences[old_start:old_end])
        if tag in {"insert", "replace"}:
            added.extend(new_sentences[new_start:new_end])
    return added, removed


def _values(text: str) -> list[str]:
    values: list[str] = []
    for match in VALUE_RE.finditer(text):
        value = re.sub(r"\s+", " ", match.group(0)).strip()
        if value and (TIME_UNIT_RE.search(value) or re.search(r"[-/%]", value)):
            values.append(value)
    return list(dict.fromkeys(values))


def _contains(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _section_identity(alignment: dict) -> tuple[str, str]:
    section = alignment.get("new") or alignment.get("old") or {}
    heading = section.get("section", "Section")
    return section.get("section_slug") or _normalized_heading(section) or "section", heading


def classify_severity(change: dict) -> tuple[str, list[str]]:
    """Classify one detected change and return transparent severity reasons."""

    high_signals = {
        "refund_window_changed": "refund window changed",
        "eligibility_changed": "refund eligibility language changed",
        "automatic_manual_review_changed": "automatic handling changed to manual review",
        "manual_review_requirement_added": "a new manual-review requirement was added",
        "non_refundable_category_added": "non-refundable scope was added",
    }
    medium_signals = {
        "processing_time_changed": "processing time changed",
        "escalation_path_changed": "escalation or operational ownership changed",
        "exception_language_added": "exception language was added",
        "exception_language_removed": "exception language was removed",
    }
    reasons = [label for key, label in high_signals.items() if change.get(key)]
    if reasons:
        return "high", reasons
    reasons = [label for key, label in medium_signals.items() if change.get(key)]
    if change.get("status") in {"added", "removed"}:
        reasons.append(f"policy section was {change['status']}")
    if reasons:
        return "medium", list(dict.fromkeys(reasons))
    return "low", ["wording or clarification changed without a recognized policy-risk signal"]


def detect_policy_changes(alignments: list[dict]) -> list[dict]:
    """Detect explicit policy changes for changed, added, and removed alignments."""

    changes: list[dict] = []
    for alignment in alignments:
        if alignment["status"] == "unchanged":
            continue
        old = alignment.get("old") or {}
        new = alignment.get("new") or {}
        old_text = old.get("text", "")
        new_text = new.get("text", "")
        slug, heading = _section_identity(alignment)
        context = f"{slug} {heading} {old_text} {new_text}".lower()
        added_text, removed_text = _text_delta(old_text, new_text)
        old_values = _values(old_text)
        new_values = _values(new_text)
        numeric_changed = old_values != new_values and bool(old_values or new_values)

        refund_context = _contains(context, REFUND_TERMS)
        processing_context = _contains(context, PROCESSING_TERMS) or "processing_time" in slug
        refund_window_context = (
            "refund_window" in slug
            or "renewal_payments" in slug
            or _contains(heading, ("退款期限", "refund window"))
        )
        old_positive = _contains(old_text, ELIGIBILITY_POSITIVE_TERMS)
        new_positive = _contains(new_text, ELIGIBILITY_POSITIVE_TERMS)
        old_restrictive = _contains(old_text, ELIGIBILITY_RESTRICTIVE_TERMS)
        new_restrictive = _contains(new_text, ELIGIBILITY_RESTRICTIVE_TERMS)
        eligibility_changed = refund_context and (
            old_positive != new_positive or old_restrictive != new_restrictive
        )
        old_manual = _contains(old_text, MANUAL_REVIEW_TERMS)
        new_manual = _contains(new_text, MANUAL_REVIEW_TERMS)
        old_automatic = _contains(old_text, AUTOMATIC_TERMS)
        new_automatic = _contains(new_text, AUTOMATIC_TERMS)
        old_non_refundable = _contains(old_text, NON_REFUNDABLE_TERMS)
        new_non_refundable = _contains(new_text, NON_REFUNDABLE_TERMS)
        old_exception = _contains(old_text, EXCEPTION_TERMS)
        new_exception = _contains(new_text, EXCEPTION_TERMS)
        old_escalation = _contains(old_text, ESCALATION_TERMS)
        new_escalation = _contains(new_text, ESCALATION_TERMS)

        change = {
            "id": f"change-{len(changes) + 1:03d}",
            "status": alignment["status"],
            "alignment_method": alignment["alignment_method"],
            "alignment_score": alignment["alignment_score"],
            "section": heading,
            "section_slug": slug,
            "old_section": old.get("section"),
            "new_section": new.get("section"),
            "old_doc": old.get("doc"),
            "new_doc": new.get("doc"),
            "old_page": old.get("page"),
            "old_page_end": old.get("page_end"),
            "new_page": new.get("page"),
            "new_page_end": new.get("page_end"),
            "old_text": old_text or None,
            "new_text": new_text or None,
            "added_text": added_text,
            "removed_text": removed_text,
            "changed_values": {
                "changed": numeric_changed,
                "old": old_values,
                "new": new_values,
            },
            "numeric_date_time_changed": numeric_changed,
            "refund_window_changed": refund_context
            and refund_window_context
            and numeric_changed,
            "eligibility_changed": eligibility_changed,
            "manual_review_requirement_added": new_manual and not old_manual,
            "automatic_manual_review_changed": old_automatic and new_manual,
            "non_refundable_category_added": new_non_refundable
            and not old_non_refundable,
            "exception_language_added": new_exception and not old_exception,
            "exception_language_removed": old_exception and not new_exception,
            "processing_time_changed": processing_context and numeric_changed,
            "escalation_path_changed": old_escalation != new_escalation,
        }
        severity, severity_reasons = classify_severity(change)
        change["severity"] = severity
        change["severity_reasons"] = severity_reasons
        change["change_types"] = [
            key
            for key in (
                "refund_window_changed",
                "eligibility_changed",
                "manual_review_requirement_added",
                "automatic_manual_review_changed",
                "non_refundable_category_added",
                "exception_language_added",
                "exception_language_removed",
                "processing_time_changed",
                "escalation_path_changed",
            )
            if change[key]
        ] or ["wording_changed"]
        changes.append(change)
    return changes


def _eval_text(row: dict) -> str:
    fields = (
        "question",
        "question_en",
        "expected_answer",
        "expected_answer_en",
        "source_section",
        "source_section_zh",
    )
    values = [str(row.get(field, "")) for field in fields]
    values.extend(str(item) for item in row.get("must_include", []))
    values.extend(str(item) for item in row.get("must_include_en", []))
    return " ".join(values)


def _impact_action(row: dict, matched_changes: list[dict]) -> str:
    if row.get("category") == "conflict":
        return "review_answer"
    if any(
        change["manual_review_requirement_added"]
        or change["automatic_manual_review_changed"]
        for change in matched_changes
    ):
        return "manual_review_required"
    if any(
        change["refund_window_changed"]
        or change["eligibility_changed"]
        or change["processing_time_changed"]
        or change["non_refundable_category_added"]
        for change in matched_changes
    ):
        return "update_expected_answer"
    if any(change["status"] in {"added", "removed"} for change in matched_changes):
        return "update_corpus_section"
    return "review_answer"


def identify_impacted_eval_cases(
    changes: list[dict], eval_path: Path | str = DEFAULT_EVAL_PATH
) -> list[dict]:
    """Map policy changes to eval questions and expected AI answers."""

    rows = load_eval_cases(Path(eval_path))
    impacted: list[dict] = []
    for row in rows:
        row_tokens = _tokens(_eval_text(row))
        source_sections = {row.get("source_section", "")}
        source_docs = set(row.get("source_docs", [row.get("source_doc", "")]))
        matched: list[tuple[dict, str]] = []
        for change in changes:
            slug = change["section_slug"]
            related_sections = RELATED_EVAL_SECTIONS.get(slug, {slug})
            change_tokens = _tokens(
                f"{change['section']} {change.get('old_text') or ''} {change.get('new_text') or ''}"
            )
            overlap = len(row_tokens & change_tokens) / max(1, len(row_tokens))
            reason: str | None = None
            if source_sections & related_sections:
                reason = f"Expected source section is directly related to changed section `{slug}`."
            elif source_docs & {change.get("old_doc"), change.get("new_doc")}:
                reason = f"Expected source document is one of the changed policy documents for `{slug}`."
            elif row.get("comparison_docs") and _contains(
                _eval_text(row), SECTION_IMPACT_TERMS.get(slug, ())
            ):
                reason = f"Conflict case references the compared documents and explicitly matches changed section `{slug}`."
            elif row.get("comparison_docs") and overlap >= 0.08:
                reason = f"Conflict case references the compared documents and overlaps changed section `{slug}`."
            elif (
                "refund_policy.md" in source_docs
                and _contains(_eval_text(row), REFUND_TERMS)
                and overlap >= 0.16
            ):
                reason = f"Refund-related expected answer overlaps changed section `{slug}`."
            elif row.get("category") == "conflict" and not row.get("comparison_docs"):
                reason = "Generic policy-conflict handling should be reviewed against this detected change set."
            if reason:
                matched.append((change, reason))

        if not matched:
            continue
        matched_changes = [item[0] for item in matched]
        reasons = list(dict.fromkeys(item[1] for item in matched))
        impacted.append(
            {
                "case_id": row["id"],
                "language": "zh/en" if row.get("question_en") else "zh",
                "languages": [
                    language
                    for language, field in (("zh", "question"), ("en", "question_en"))
                    if row.get(field)
                ],
                "question": row.get("question"),
                "question_en": row.get("question_en"),
                "expected_source": row.get("source_doc"),
                "expected_sources": sorted(source_docs),
                "expected_section": row.get("source_section"),
                "reason_impacted": " ".join(reasons),
                "change_ids": [change["id"] for change in matched_changes],
                "suggested_action": _impact_action(row, matched_changes),
            }
        )
    return impacted


def _corpus_section_index(corpus_dir: Path) -> dict[tuple[str, str], dict]:
    index: dict[tuple[str, str], dict] = {}
    if not corpus_dir.is_dir():
        return index
    for path in sorted(corpus_dir.glob("*.md")):
        for section in parse_markdown_sections(path):
            index[(path.name, section["section_slug"])] = section
    return index


def identify_required_kb_updates(
    changes: list[dict], corpus_dir: Path | str = DEFAULT_CORPUS_DIR
) -> list[dict]:
    """Suggest existing corpus sections that should be verified or updated."""

    corpus_path = Path(corpus_dir)
    existing = _corpus_section_index(corpus_path)
    recommendations: dict[tuple[str, str], dict] = {}
    for change in changes:
        slug = change["section_slug"]
        candidates = [("refund_policy.md", slug)]
        candidates.extend(RELATED_KB_SECTIONS.get(slug, ()))
        for doc, section_slug in candidates:
            key = (doc, section_slug)
            current = existing.get(key)
            if not current:
                continue
            recommendation = recommendations.setdefault(
                key,
                {
                    "file": f"corpus/{doc}",
                    "section": current["section"],
                    "section_slug": section_slug,
                    "exists": True,
                    "suggested_new_file": False,
                    "priority": change["severity"],
                    "action": "verify_or_update_corpus_section",
                    "reason": [],
                    "change_ids": [],
                },
            )
            if change["severity"] == "high":
                recommendation["priority"] = "high"
            elif change["severity"] == "medium" and recommendation["priority"] == "low":
                recommendation["priority"] = "medium"
            recommendation["reason"].append(
                f"Policy section `{slug}` changed ({', '.join(change['change_types'])})."
            )
            recommendation["change_ids"].append(change["id"])

    output: list[dict] = []
    for recommendation in recommendations.values():
        recommendation["reason"] = " ".join(dict.fromkeys(recommendation["reason"]))
        recommendation["change_ids"] = list(dict.fromkeys(recommendation["change_ids"]))
        output.append(recommendation)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        output,
        key=lambda item: (priority_order[item["priority"]], item["file"], item["section_slug"]),
    )


def _human_review_recommendations(changes: list[dict], impacted: list[dict]) -> list[str]:
    recommendations = [
        "Have the policy owner validate every high-risk change before affected AI answers are released.",
        "Review the listed eval answers and corpus sections against the approved effective policy version, then rerun the Ask Mode gate.",
    ]
    high_slugs = [change["section_slug"] for change in changes if change["severity"] == "high"]
    if high_slugs:
        recommendations.insert(
            0,
            "Require human review for answers using these high-risk sections until updates are verified: "
            + ", ".join(high_slugs)
            + ".",
        )
    if any(item["suggested_action"] == "manual_review_required" for item in impacted):
        recommendations.append(
            "Confirm that Enterprise/manual-review routing and support-agent instructions are operationally enforced."
        )
    return recommendations


def compare_documents(
    old_path: Path | str,
    new_path: Path | str,
    *,
    eval_path: Path | str = DEFAULT_EVAL_PATH,
    corpus_dir: Path | str = DEFAULT_CORPUS_DIR,
) -> dict:
    """Build a complete structured policy change-impact result."""

    old_document = Path(old_path)
    new_document = Path(new_path)
    old_sections = load_document(old_document)
    new_sections = load_document(new_document)
    alignments = align_sections(old_sections, new_sections)
    changes = detect_policy_changes(alignments)
    impacted = identify_impacted_eval_cases(changes, eval_path)
    updates = identify_required_kb_updates(changes, corpus_dir)
    severity_counts = {
        severity: sum(change["severity"] == severity for change in changes)
        for severity in ("high", "medium", "low")
    }
    return {
        "schema_version": 1,
        "mode": "change_impact",
        "disclaimer": DISCLAIMER,
        "compared_documents": {
            "old": str(old_document),
            "new": str(new_document),
        },
        "summary": {
            "old_sections": len(old_sections),
            "new_sections": len(new_sections),
            "aligned_sections": sum(
                alignment["old"] is not None and alignment["new"] is not None
                for alignment in alignments
            ),
            "changed_sections": len(changes),
            "added_sections": sum(change["status"] == "added" for change in changes),
            "removed_sections": sum(change["status"] == "removed" for change in changes),
            "severity": severity_counts,
            "impacted_eval_cases": len(impacted),
            "required_kb_updates": len(updates),
            "human_review_required": bool(severity_counts["high"]),
        },
        "parsed_sections": {"old": old_sections, "new": new_sections},
        "alignments": alignments,
        "changes": changes,
        "high_risk_changes": [
            change["id"] for change in changes if change["severity"] == "high"
        ],
        "impacted_eval_cases": impacted,
        "required_kb_updates": updates,
        "human_review_recommendations": _human_review_recommendations(changes, impacted),
        "known_limitations": list(KNOWN_LIMITATIONS),
    }


compare_policies = compare_documents


def render_markdown(result: dict) -> str:
    """Render a human-readable Change Impact report."""

    summary = result["summary"]
    old_doc = result["compared_documents"]["old"]
    new_doc = result["compared_documents"]["new"]
    lines = [
        "# Policy Change Impact Report",
        "",
        "## Executive summary",
        "",
        f"Detected **{summary['changed_sections']}** changed sections: "
        f"**{summary['severity']['high']} high**, **{summary['severity']['medium']} medium**, "
        f"and **{summary['severity']['low']} low** risk. "
        f"The analysis flagged **{summary['impacted_eval_cases']}** eval questions or existing "
        f"answer expectations and **{summary['required_kb_updates']}** KB sections for review.",
        "",
        f"> {result['disclaimer']}",
        "",
        "## Compared documents",
        "",
        f"- Old: `{old_doc}` ({summary['old_sections']} parsed sections)",
        f"- New: `{new_doc}` ({summary['new_sections']} parsed sections)",
        "",
        "## Changed sections",
        "",
    ]
    for change in result["changes"]:
        types = ", ".join(change["change_types"])
        values = change["changed_values"]
        lines.extend(
            (
                f"### {change['section']} (`{change['section_slug']}`)",
                "",
                f"- Status: {change['status']}",
                f"- Severity: **{change['severity']}**",
                f"- Alignment: {change['alignment_method']} ({change['alignment_score']:.2f})",
                f"- Detected signals: {types}",
                f"- Severity basis: {'; '.join(change['severity_reasons'])}",
            )
        )
        if values["changed"]:
            lines.append(
                f"- Values: `{', '.join(values['old']) or 'none'}` → `{', '.join(values['new']) or 'none'}`"
            )
        if change.get("old_page") is not None or change.get("new_page") is not None:
            lines.append(
                f"- Pages: old `{change.get('old_page') or 'none'}` → "
                f"new `{change.get('new_page') or 'none'}`"
            )
        if change["removed_text"]:
            lines.append(f"- Removed: {' '.join(change['removed_text'])}")
        if change["added_text"]:
            lines.append(f"- Added: {' '.join(change['added_text'])}")
        lines.append("")

    lines.extend(("## High-risk changes", ""))
    high_changes = [change for change in result["changes"] if change["severity"] == "high"]
    if high_changes:
        for change in high_changes:
            lines.append(
                f"- **{change['section']}** — {'; '.join(change['severity_reasons'])}."
            )
    else:
        lines.append("- None detected by the configured rules.")

    lines.extend(("", "## Impacted eval questions / existing answers", ""))
    if result["impacted_eval_cases"]:
        lines.append("| Case | Language | Question | Expected source / section | Reason | Suggested action |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for case in result["impacted_eval_cases"]:
            question = (case["question"] or "").replace("|", "\\|")
            reason = case["reason_impacted"].replace("|", "\\|")
            expected = f"{case['expected_source']} / {case['expected_section']}"
            lines.append(
                f"| {case['case_id']} | {case['language']} | {question} | {expected} | {reason} | `{case['suggested_action']}` |"
            )
    else:
        lines.append("- None identified.")

    lines.extend(("", "## Required KB updates", ""))
    if result["required_kb_updates"]:
        for update in result["required_kb_updates"]:
            lines.append(
                f"- **{update['priority']}** — `{update['file']}` / `{update['section_slug']}`: {update['reason']}"
            )
    else:
        lines.append("- No existing corpus target was identified; a policy owner must map the changed section manually.")

    lines.extend(("", "## Human review recommendations", ""))
    lines.extend(f"- {item}" for item in result["human_review_recommendations"])
    lines.extend(("", "## Known limitations", ""))
    lines.extend(f"- {item}" for item in result["known_limitations"])
    lines.append("")
    return "\n".join(lines)


def write_reports(
    result: dict, output_dir: Path | str = DEFAULT_OUTPUT_DIR
) -> tuple[Path, Path]:
    """Write the JSON and Markdown Change Impact artifacts."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / JSON_FILENAME
    report_path = target / REPORT_FILENAME
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, report_path


write_change_reports = write_reports


def _print_summary(result: dict, json_path: Path, report_path: Path) -> None:
    summary = result["summary"]
    print("Change Impact summary")
    print(f"Compared: {result['compared_documents']['old']} -> {result['compared_documents']['new']}")
    print(
        f"Changed sections: {summary['changed_sections']} "
        f"(high={summary['severity']['high']}, medium={summary['severity']['medium']}, low={summary['severity']['low']})"
    )
    print(f"Impacted eval cases: {summary['impacted_eval_cases']}")
    print(f"Required KB updates: {summary['required_kb_updates']}")
    print(f"Human review required: {'yes' if summary['human_review_required'] else 'no'}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write report artifacts (retained as the default for backward compatibility)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full JSON result instead of the summary"
    )
    args = parser.parse_args()

    try:
        result = compare_documents(args.old, args.new)
    except ValueError as exc:
        parser.error(str(exc))
    json_path, report_path = write_reports(result, args.output_dir)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_summary(result, json_path, report_path)


if __name__ == "__main__":
    main()
