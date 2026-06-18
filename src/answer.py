"""Answer support questions from retrieved Day 1 corpus evidence."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[SearchResult]
    refused: bool


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
        ("how much", "price in", "monthly price", "exact price", "價格是多少", "多少台幣", "精確價格"),
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


def answer_question(
    question: str,
    *,
    top_k: int = 5,
    index_path: Path = DEFAULT_INDEX_PATH,
    retriever: RetrieverName = "lexical",
    model_name: str = DEFAULT_DENSE_MODEL,
    cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
) -> Answer:
    results = retrieve(
        question,
        top_k=top_k,
        index_path=index_path,
        retriever=retriever,
        model_name=model_name,
        cache_dir=cache_dir,
    )
    if not results or results[0]["score"] < MIN_RELEVANCE_SCORES[retriever]:
        return Answer(_generic_refusal(question), [], True)

    top = results[0]
    refusal = _refusal_evidence(question, results)
    if refusal:
        reason, primary = refusal
        return Answer(
            _supported_refusal(question, reason),
            _related_citations(results, primary, reason),
            True,
        )

    return Answer(top["text"], [top], False)


def format_answer(answer: Answer) -> str:
    lines = ["Answer draft:", answer.text, "", "Citations:"]
    if not answer.citations:
        lines.append("- None")
    else:
        for citation in answer.citations:
            section = citation.get("section_zh") or citation["section"]
            slug = citation.get("section_slug")
            section_label = f"{section} ({slug})" if slug else section
            lines.append(
                f"- {citation['doc']} / {section_label} / {citation['chunk_id']}"
            )
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
    args = parser.parse_args()

    print(
        format_answer(
            answer_question(
                args.question,
                top_k=args.top_k,
                index_path=args.index,
                retriever=args.retriever,
                model_name=args.model,
                cache_dir=args.embedding_cache,
            )
        )
    )


if __name__ == "__main__":
    main()
