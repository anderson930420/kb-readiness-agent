"""Answer support questions from retrieved Day 1 corpus evidence."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from .ingest import DEFAULT_INDEX_PATH
from .retrieve import SearchResult, retrieve


MIN_RELEVANCE_SCORE = 1.0
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


def _refusal_reason(question: str, top: SearchResult) -> str | None:
    slug = top.get("section_slug", "")

    if slug == "unsupported_exceptions":
        return "refund_exception"
    if slug == "unsupported_privacy_questions":
        return "privacy"

    exact_sla = _contains_any(question, ("exact", "uptime", "精確", "可用率"))
    if exact_sla and slug in {"enterprise_support_response_time", "sla_escalation"}:
        return "sla"

    exact_price = _contains_any(
        question,
        ("how much", "price in", "monthly price", "exact price", "價格是多少", "多少台幣", "精確價格"),
    )
    if exact_price and slug in {"enterprise_pricing", "enterprise_pricing_quote", "sales_escalation"}:
        return "pricing"

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
) -> Answer:
    results = retrieve(question, top_k=top_k, index_path=index_path)
    if not results or results[0]["score"] < MIN_RELEVANCE_SCORE:
        return Answer(_generic_refusal(question), [], True)

    top = results[0]
    reason = _refusal_reason(question, top)
    if reason:
        return Answer(
            _supported_refusal(question, reason),
            _related_citations(results, top, reason),
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
    args = parser.parse_args()

    print(
        format_answer(
            answer_question(args.question, top_k=args.top_k, index_path=args.index)
        )
    )


if __name__ == "__main__":
    main()
