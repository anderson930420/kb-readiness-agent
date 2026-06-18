"""Answer support questions from retrieved Ask Mode corpus evidence."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Literal, TypedDict

from .ingest import DEFAULT_INDEX_PATH
from .retrieve import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_EMBEDDING_CACHE_DIR,
    RETRIEVAL_METHODS,
    RetrieverName,
    SearchResult,
    retrieve,
)


MIN_RELEVANCE_SCORES: dict[RetrieverName, float] = {
    "lexical": 1.0,
    "dense": 0.2,
    "hybrid": 0.2,
}
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
NUMBER_RE = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?(?:\s*(?:%|％|days?|business\s+days?|"
    r"hours?|minutes?|years?|months?|天|日|個工作天|小時|分鐘|年|月))?",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s*")
Confidence = Literal["high", "medium", "low"]


class Citation(TypedDict):
    chunk_id: str
    doc: str
    section: str
    section_zh: str
    section_slug: str
    page: int | None
    text: str


class GroundednessChecks(TypedDict):
    has_citation: bool
    citations_from_retrieved_chunks: bool
    numeric_claims_supported: bool
    refusal_supported: bool


class GroundednessResult(TypedDict):
    status: Literal["supported", "unsupported"]
    checks: GroundednessChecks
    unsupported_claims: list[str]


@dataclass(frozen=True)
class AnswerResult:
    question: str
    retriever: RetrieverName
    answer: str
    refused: bool
    refusal_reason: str | None
    requires_human_review: bool
    confidence: Confidence
    citations: list[Citation]
    retrieved_chunks: list[SearchResult]
    groundedness: GroundednessResult
    warnings: list[str]
    latency_ms: float

    @property
    def text(self) -> str:
        """Compatibility alias for the Day 1 answer contract."""

        return self.answer

    def to_dict(self) -> dict:
        return asdict(self)


# Backward-compatible import name.
Answer = AnswerResult


def _is_chinese(question: str) -> bool:
    return bool(CJK_RE.search(question))


def _generic_refusal(question: str) -> str:
    if _is_chinese(question):
        return "目前知識庫沒有足夠依據回答此問題。請補充相關政策或轉交人工複核。"
    return (
        "The current knowledge base does not contain enough evidence to answer "
        "this question. Please add the relevant policy or escalate for manual review."
    )


def _supported_refusal(question: str, reason: str) -> str:
    if _is_chinese(question):
        reasons = {
            "refund_exception": "現有退款政策尚未定義醫療、緊急事故或困難情境下的退款例外規則，應轉交人工審查。",
            "sla": "Enterprise SLA 的精確內容以客戶簽署的服務水準協議為準，沒有已簽署的 SLA 時應轉交帳戶經理。",
            "privacy": "現有隱私權政策不提供地區性隱私法規的個別法律意見，應轉交法務或合規團隊。",
            "pricing": "Enterprise 方案不公開列價，應由業務團隊提供客製報價。",
        }
        return f"目前知識庫沒有足夠依據回答此問題。{reasons[reason]}"

    reasons = {
        "refund_exception": "The refund policy does not define medical, emergency, or hardship-based refund exceptions; escalate for manual review.",
        "sla": "Exact Enterprise SLA terms depend on the customer's signed service-level agreement; escalate to the account manager when it is unavailable.",
        "privacy": "The privacy policy does not provide regional legal advice; escalate to legal or compliance.",
        "pricing": "Enterprise pricing is not publicly listed and requires a custom quote from the sales team.",
    }
    return f"The current knowledge base does not contain enough evidence. {reasons[reason]}"


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in phrases)


def _result_with_slug(
    results: list[SearchResult], slugs: set[str]
) -> SearchResult | None:
    return next(
        (result for result in results if result.get("section_slug") in slugs), None
    )


def _refusal_evidence(
    question: str, results: list[SearchResult]
) -> tuple[str, SearchResult] | None:
    top = results[0]
    slug = top.get("section_slug", "")

    if slug == "unsupported_exceptions":
        return "refund_exception", top
    if slug == "unsupported_privacy_questions":
        return "privacy", top

    refund_exception = _contains_any(question, ("refund", "退款")) and _contains_any(
        question,
        (
            "medical",
            "emergency",
            "hardship",
            "90 days",
            "醫療",
            "緊急",
            "困難",
            "90 天",
        ),
    )
    if refund_exception:
        evidence = _result_with_slug(results, {"unsupported_exceptions"})
        if evidence:
            return "refund_exception", evidence

    regional_privacy = _contains_any(
        question, ("gdpr", "ccpa", "article 17", "privacy law", "隱私法規", "法律意見")
    )
    if regional_privacy:
        evidence = _result_with_slug(results, {"unsupported_privacy_questions"})
        if evidence:
            return "privacy", evidence

    exact_sla = _contains_any(question, ("exact", "uptime", "精確", "可用率"))
    if exact_sla:
        evidence = _result_with_slug(
            results, {"enterprise_support_response_time", "sla_escalation"}
        )
        if evidence:
            return "sla", evidence

    exact_price = _contains_any(
        question,
        (
            "how much",
            "price in",
            "monthly price",
            "exact price",
            "價格是多少",
            "多少台幣",
            "精確價格",
        ),
    )
    if exact_price:
        evidence = _result_with_slug(
            results,
            {"enterprise_pricing", "enterprise_pricing_quote", "sales_escalation"},
        )
        if evidence:
            return "pricing", evidence

    return None


def _related_citations(
    results: list[SearchResult], primary: SearchResult, reason: str
) -> list[SearchResult]:
    companion_slugs = {
        "refund_exception": {"refund_escalation"},
        "sla": {"sla_escalation", "enterprise_support_response_time"},
        "privacy": {"privacy_escalation", "unsupported_privacy_questions"},
        "pricing": {"sales_escalation", "enterprise_pricing", "enterprise_pricing_quote"},
    }[reason]
    citations = [primary]
    for result in results:
        if result["chunk_id"] == primary["chunk_id"]:
            continue
        if result.get("section_slug") in companion_slugs:
            citations.append(result)
            break
    return citations


def _citation(result: SearchResult) -> Citation:
    return {
        "chunk_id": result["chunk_id"],
        "doc": result["doc"],
        "section": result["section"],
        "section_zh": result["section_zh"],
        "section_slug": result["section_slug"],
        "page": result["page"],
        "text": result["text"],
    }


def _normalized_numbers(text: str) -> set[str]:
    return {
        re.sub(r"\s+", "", match.group(0).lower()).replace("％", "%")
        for match in NUMBER_RE.finditer(text)
    }


def _states_insufficient_evidence(text: str) -> bool:
    return _contains_any(
        text,
        (
            "沒有足夠依據",
            "不足以回答",
            "does not contain enough evidence",
            "insufficient evidence",
            "not enough evidence",
        ),
    )


def check_groundedness(
    answer: str,
    *,
    refused: bool,
    citations: list[Citation],
    retrieved_chunks: list[SearchResult],
) -> tuple[GroundednessResult, list[str]]:
    """Validate mechanical answer-to-evidence invariants without an LLM judge."""

    has_citation = bool(citations)
    retrieved_pairs = {
        (chunk["chunk_id"], chunk["doc"]) for chunk in retrieved_chunks
    }
    citations_from_retrieved = all(
        (citation["chunk_id"], citation["doc"]) in retrieved_pairs
        for citation in citations
    )
    citation_numbers = _normalized_numbers(
        "\n".join(citation["text"] for citation in citations)
    )
    unsupported_numbers = sorted(_normalized_numbers(answer) - citation_numbers)
    numeric_claims_supported = not unsupported_numbers
    refusal_supported = (
        not refused or has_citation or _states_insufficient_evidence(answer)
    )

    unsupported_claims: list[str] = []
    warnings: list[str] = []
    if not refused and not has_citation:
        unsupported_claims.append("Non-refused answer has no citation")
        warnings.append("Answer has no citation")
    if not citations_from_retrieved:
        unsupported_claims.append("Citation is not present in retrieved evidence")
        warnings.append("Citation provenance check failed")
    if unsupported_numbers:
        unsupported_claims.extend(
            f"Unsupported numeric/date/time claim: {value}"
            for value in unsupported_numbers
        )
        warnings.append("Numeric/date/time claims are not fully supported by citations")
    if not refusal_supported:
        unsupported_claims.append(
            "Refusal has neither relevant evidence nor an insufficient-evidence statement"
        )
        warnings.append("Refusal basis is unclear")

    required_checks = (
        citations_from_retrieved,
        numeric_claims_supported,
        refusal_supported,
        has_citation or refused,
    )
    groundedness: GroundednessResult = {
        "status": "supported" if all(required_checks) else "unsupported",
        "checks": {
            "has_citation": has_citation,
            "citations_from_retrieved_chunks": citations_from_retrieved,
            "numeric_claims_supported": numeric_claims_supported,
            "refusal_supported": refusal_supported,
        },
        "unsupported_claims": unsupported_claims,
    }
    return groundedness, warnings


def _scoped_sentences(question: str, text: str) -> list[str]:
    qualifier_groups = (
        (("monthly", "month-to-month", "月付"), ("monthly", "month-to-month", "月付")),
        (("annual", "yearly", "年度", "年付"), ("annual", "yearly", "年度", "年付")),
    )
    sentences = [sentence for sentence in SENTENCE_SPLIT_RE.split(text) if sentence]
    for question_terms, evidence_terms in qualifier_groups:
        if _contains_any(question, question_terms):
            scoped = [
                sentence
                for sentence in sentences
                if _contains_any(sentence, evidence_terms)
            ]
            return scoped
    return sentences


def _has_query_scoped_conflict(
    question: str, retrieved_chunks: list[SearchResult]
) -> bool:
    """Flag only clear value mismatches within duplicate policy sections."""

    section_query_terms = {
        "standard_refund_window": (
            "refund window",
            "refund deadline",
            "monthly",
            "annual",
            "退款期限",
            "月付",
            "年付",
            "年度訂閱",
        ),
        "refund_processing_time": (
            "refund process",
            "approved refund",
            "退款處理",
            "核准的退款",
        ),
        "billing_cycle": (
            "billing cycle",
            "annual billing",
            "annual discount",
            "計費週期",
            "年付方案",
            "折扣",
        ),
        "data_deletion": (
            "data deletion",
            "deletion request",
            "資料刪除",
            "刪除請求",
        ),
    }
    by_section: dict[str, list[SearchResult]] = {}
    for chunk in retrieved_chunks:
        slug = chunk.get("section_slug", "")
        if slug in section_query_terms and _contains_any(
            question, section_query_terms[slug]
        ):
            by_section.setdefault(slug, []).append(chunk)

    for chunks in by_section.values():
        if len(chunks) < 2:
            continue
        value_sets = []
        for chunk in chunks:
            scoped_text = " ".join(_scoped_sentences(question, chunk["text"]))
            values = frozenset(_normalized_numbers(scoped_text))
            if values:
                value_sets.append(values)
        if len(value_sets) >= 2 and len(set(value_sets)) > 1:
            return True
    return False


def answer_from_retrieved(
    question: str,
    retrieved_chunks: list[SearchResult],
    *,
    retriever: RetrieverName,
    latency_ms: float = 0.0,
) -> AnswerResult:
    """Build and validate an AnswerResult from an existing retrieval result list."""

    refused = False
    refusal_reason: str | None = None
    citation_results: list[SearchResult] = []

    if (
        not retrieved_chunks
        or retrieved_chunks[0]["score"] < MIN_RELEVANCE_SCORES[retriever]
    ):
        answer = _generic_refusal(question)
        refused = True
        refusal_reason = "insufficient_relevance"
    else:
        refusal = _refusal_evidence(question, retrieved_chunks)
        if refusal:
            refusal_reason, primary = refusal
            answer = _supported_refusal(question, refusal_reason)
            citation_results = _related_citations(
                retrieved_chunks, primary, refusal_reason
            )
            refused = True
        else:
            top = retrieved_chunks[0]
            answer = top["text"]
            citation_results = [top]

    citations = [_citation(result) for result in citation_results]
    groundedness, warnings = check_groundedness(
        answer,
        refused=refused,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
    )
    requires_human_review = refused or groundedness["status"] != "supported"
    if groundedness["status"] != "supported":
        confidence: Confidence = "low"
    elif refused:
        confidence = "medium" if citations else "low"
    else:
        confidence = "high"

    if _has_query_scoped_conflict(question, retrieved_chunks):
        requires_human_review = True
        warnings.append("Potential conflicting evidence detected")

    return AnswerResult(
        question=question,
        retriever=retriever,
        answer=answer,
        refused=refused,
        refusal_reason=refusal_reason,
        requires_human_review=requires_human_review,
        confidence=confidence,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
        groundedness=groundedness,
        warnings=warnings,
        latency_ms=round(latency_ms, 3),
    )


def answer_question(
    question: str,
    *,
    top_k: int = 5,
    index_path: Path = DEFAULT_INDEX_PATH,
    retriever: RetrieverName = "lexical",
    model_name: str = DEFAULT_DENSE_MODEL,
    cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
) -> AnswerResult:
    started = perf_counter()
    results = retrieve(
        question,
        top_k=top_k,
        index_path=index_path,
        retriever=retriever,
        model_name=model_name,
        cache_dir=cache_dir,
    )
    result = answer_from_retrieved(
        question,
        results,
        retriever=retriever,
    )
    return replace(result, latency_ms=round((perf_counter() - started) * 1000, 3))


def format_answer(result: AnswerResult) -> str:
    lines = [
        "Answer draft:",
        result.answer,
        "",
        f"Retriever: {result.retriever}",
        f"Refused: {'yes' if result.refused else 'no'}",
        f"Confidence: {result.confidence}",
        f"Requires human review: {'yes' if result.requires_human_review else 'no'}",
        f"Groundedness: {result.groundedness['status']}",
        "",
        "Citations:",
    ]
    if not result.citations:
        lines.append("- None")
    else:
        for citation in result.citations:
            section = citation.get("section_zh") or citation["section"]
            slug = citation.get("section_slug")
            section_label = f"{section} ({slug})" if slug else section
            lines.append(
                f"- {citation['doc']} / {section_label} / {citation['chunk_id']}"
            )
    if result.warnings:
        lines.extend(("", "Warnings:"))
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)


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
    parser.add_argument(
        "--json", action="store_true", help="Print the full AnswerResult as JSON"
    )
    args = parser.parse_args()

    result = answer_question(
        args.question,
        top_k=args.top_k,
        index_path=args.index,
        retriever=args.retriever,
        model_name=args.model,
        cache_dir=args.embedding_cache,
    )
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_answer(result))


if __name__ == "__main__":
    main()
