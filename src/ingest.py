"""Build the Day 1 Ask Mode index from the support corpus."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "corpus"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "index" / "chunks.jsonl"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SLUG_IN_TITLE_RE = re.compile(r"[（(]([a-zA-Z0-9_-]+)[）)]")

# English aliases make Chinese-primary evidence discoverable by English queries.
# They are retrieval metadata, not additional policy claims.
SECTION_ALIASES: dict[str, str] = {
    "plans": "plans Starter Growth Enterprise plan features monthly orders",
    "billing_cycle": "billing cycle monthly billing annual billing annual discount",
    "overages": "overages monthly order limit suspended immediately overage notice upgrade fee",
    "payment_failure": "payment failure email reminders day 1 day 3 day 7 14 days account suspended",
    "enterprise_pricing": "Enterprise pricing exact price monthly price USD TWD not publicly listed custom quote sales team Enterprise 方案每月價格 台幣 不公開列價",
    "standard_refund_window": "refund window standard monthly subscribers annual subscribers initial purchase money back return funds 購買 拿回款項 7 days 14 days",
    "renewal_payments": "renewal payments refundable non-refundable applicable law",
    "digital_services": "completed onboarding migration professional services fees non-refundable delivered",
    "enterprise_refunds": "Enterprise refunds automatic refunds manual review account management support promise",
    "refund_processing_time": "approved refund processing time 5 to 10 business days",
    "unsupported_exceptions": "refund exceptions medical emergency hardship 90 days unsupported not defined insufficient evidence",
    "data_collection": "data collection account billing storefront order metadata support conversations",
    "data_deletion": "account data deletion request records removed removal admin console 資料移除 移除資料 30 days",
    "data_export": "data export storefront configuration product listings order metadata admin console",
    "enterprise_data_requests": "Enterprise custom data retention terms account manager",
    "sensitive_data": "sensitive data medical records support tickets government identification payment card numbers upload",
    "unsupported_privacy_questions": "GDPR CCPA Article 17 regional privacy laws legal advice compliance insufficient evidence 是否符合 GDPR 所有要求 地區性隱私法規 個別法律意見",
    "enterprise_plan_inclusions": "Enterprise plan included single sign-on SSO audit logs API limits dedicated support custom onboarding service-level agreements",
    "enterprise_automatic_refunds": "Enterprise automatic refunds manual review customer success account management support promise 客服可以向 Enterprise 客戶承諾自動退款",
    "enterprise_support_response_time": "Enterprise exact uptime SLA support response time signed service-level agreement account manager insufficient evidence Enterprise 方案精確系統可用率 已簽署 SLA",
    "enterprise_data_retention": "Enterprise custom data retention terms account manager",
    "enterprise_pricing_quote": "support quote Enterprise pricing custom quote sales team",
    "standard_onboarding": "standard self-service onboarding Starter Growth admin console",
    "enterprise_onboarding": "Enterprise custom onboarding migration SSO security review training",
    "migration_services": "custom migration services statement of work schedule",
    "onboarding_fees": "completed onboarding custom migration services non-refundable delivered",
    "support_during_onboarding": "onboarding support basic setup custom migration scope onboarding specialist escalate",
    "general_rule": "support answer only sufficient knowledge base evidence",
    "refund_escalation": "refund escalation Enterprise hardship exception unclear refund window manual review",
    "privacy_escalation": "regional country-specific privacy legal compliance escalation",
    "sla_escalation": "Enterprise SLA signed agreement unavailable response time account manager escalation",
    "sales_escalation": "Enterprise pricing sales escalation support must not quote",
    "low_confidence_handling": "low confidence reliable source insufficient evidence missing policy area",
    "evidence_conflict_handling": "conflicting evidence policy conflict old policy new policy manual review no final answer",
}


def _slug(title: str) -> str:
    match = SLUG_IN_TITLE_RE.search(title)
    if match:
        return match.group(1).lower()
    value = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    return value or "section"


def _chinese_title(title: str) -> str:
    return re.sub(r"\s*[（(][a-zA-Z0-9_-]+[）)]\s*", "", title).strip()


def _sections(markdown: str) -> Iterable[tuple[str, str]]:
    section = "Document"
    body: list[str] = []

    def flush() -> tuple[str, str] | None:
        text = "\n".join(body).strip()
        return (section, text) if text else None

    for line in markdown.splitlines():
        heading = HEADING_RE.match(line)
        if not heading:
            if line.strip():
                body.append(line.strip())
            continue

        item = flush()
        if item:
            yield item
        body = []
        section = heading.group(2).strip()

    item = flush()
    if item:
        yield item


def build_chunks(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> list[dict]:
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    chunks: list[dict] = []
    for path in sorted(corpus_dir.glob("*.md")):
        markdown = path.read_text(encoding="utf-8")
        for number, (section, text) in enumerate(_sections(markdown), start=1):
            section_slug = _slug(section)
            chunk_id = f"{path.stem.replace('_', '-')}-{section_slug}-{number:03d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc": path.name,
                    "section": section,
                    "section_zh": _chinese_title(section),
                    "section_slug": section_slug,
                    "page": None,
                    "text": text,
                    "content": text,
                    "aliases": SECTION_ALIASES.get(section_slug, ""),
                }
            )
    return chunks


def write_index(chunks: Iterable[dict], output_path: Path = DEFAULT_INDEX_PATH) -> int:
    rows = list(chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as index_file:
        for row in rows:
            index_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def ingest(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    output_path: Path = DEFAULT_INDEX_PATH,
) -> int:
    return write_index(build_chunks(corpus_dir), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX_PATH)
    args = parser.parse_args()

    count = ingest(args.corpus_dir, args.output)
    print(f"Indexed {count} corpus chunks")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
