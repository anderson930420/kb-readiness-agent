"""Deterministic, process-local follow-up resolution for Ask Mode."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .answer import (
    AnswerMode,
    AnswerResult,
    answer_question,
    format_answer,
    uses_kb_pipeline,
)
from .generation import LLM_PROVIDERS, LLMProvider
from .ingest import DEFAULT_INDEX_PATH
from .retrieve import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_EMBEDDING_CACHE_DIR,
    RETRIEVAL_METHODS,
    RetrieverName,
)


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
ENGLISH_FOLLOW_UP_RE = re.compile(
    r"^\s*(?:what|how)\s+about\b|^\s*and\s+(?:what|how|for|the)\b",
    re.IGNORECASE,
)
CHINESE_FOLLOW_UP_RE = re.compile(r"(?:呢|那(?:麼)?|那如果|又如何)[？?]?\s*$")

TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("refund", ("refund", "退款", "退費")),
    ("pricing", ("price", "pricing", "quote", "價格", "定價", "報價")),
    ("privacy", ("privacy", "data", "隱私", "資料")),
    ("onboarding", ("onboarding", "migration", "導入", "移轉")),
    ("sla", ("sla", "uptime", "response time", "可用率", "回覆時間")),
)


@dataclass(frozen=True)
class SessionTurn:
    original_question: str
    resolved_question: str
    answer: AnswerResult

    def to_dict(self) -> dict:
        return asdict(self)


def _contains(text: str, values: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(value.lower() in lowered for value in values)


def _topic_from_turn(turn: SessionTurn) -> str | None:
    context = " ".join(
        (
            turn.resolved_question,
            *(citation["section_slug"] for citation in turn.answer.citations),
        )
    )
    for topic, markers in TOPICS:
        if _contains(context, markers):
            return topic
    return None


def _is_follow_up(question: str) -> bool:
    return bool(
        ENGLISH_FOLLOW_UP_RE.search(question)
        or CHINESE_FOLLOW_UP_RE.search(question)
    )


def resolve_follow_up(question: str, turns: list[SessionTurn]) -> str:
    """Resolve an underspecified follow-up using only this session's last turn."""

    normalized = question.strip()
    if not turns or not _is_follow_up(normalized):
        return normalized

    topic = _topic_from_turn(turns[-1])
    if topic == "refund" and _contains(normalized, ("enterprise", "企業客戶")):
        if CJK_RE.search(normalized):
            return (
                "Enterprise 客戶的退款政策是什麼，包括自動退款、人工審查與"
                "退款升級處理要求？"
            )
        return (
            "What is the Enterprise customer refund policy, including automatic "
            "refunds, manual review, and refund escalation requirements?"
        )

    if topic is None:
        return normalized

    topic_labels = {
        "refund": ("refund policy", "退款政策"),
        "pricing": ("pricing policy", "定價政策"),
        "privacy": ("privacy policy", "隱私政策"),
        "onboarding": ("onboarding policy", "導入政策"),
        "sla": ("SLA policy", "SLA 政策"),
    }
    english_topic, chinese_topic = topic_labels[topic]
    if CJK_RE.search(normalized):
        return f"關於{chinese_topic}，{normalized.rstrip('？?')}的適用規則是什麼？"
    return f"Regarding the {english_topic}, {normalized.rstrip('?')}?"


class AnswerSession:
    """Hold follow-up context in memory for the lifetime of one Python object."""

    def __init__(
        self,
        *,
        top_k: int = 5,
        index_path: Path = DEFAULT_INDEX_PATH,
        retriever: RetrieverName = "lexical",
        model_name: str = DEFAULT_DENSE_MODEL,
        cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
        mode: AnswerMode = "extractive",
        llm_provider: LLMProvider | None = None,
        llm_model: str | None = None,
    ) -> None:
        self.top_k = top_k
        self.index_path = index_path
        self.retriever = retriever
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.mode = mode
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.turns: list[SessionTurn] = []

    def ask(self, question: str) -> SessionTurn:
        resolved = resolve_follow_up(question, self.turns)
        answer = answer_question(
            resolved,
            top_k=self.top_k,
            index_path=self.index_path,
            retriever=self.retriever,
            model_name=self.model_name,
            cache_dir=self.cache_dir,
            mode=self.mode,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
        )
        turn = SessionTurn(
            original_question=question,
            resolved_question=resolved,
            answer=answer,
        )
        self.turns.append(turn)
        return turn

    def clear(self) -> None:
        self.turns.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", nargs="+")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--retriever", choices=RETRIEVAL_METHODS, default="lexical")
    parser.add_argument("--model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument(
        "--mode", choices=("extractive", "generative"), default="extractive"
    )
    parser.add_argument("--llm-provider", choices=LLM_PROVIDERS)
    parser.add_argument("--llm-model")
    parser.add_argument(
        "--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE_DIR
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if (
        args.mode == "generative"
        and args.llm_provider is None
        and any(uses_kb_pipeline(question) for question in args.questions)
    ):
        parser.error("--llm-provider is required when --mode generative")

    session = AnswerSession(
        top_k=args.top_k,
        index_path=args.index,
        retriever=args.retriever,
        model_name=args.model,
        cache_dir=args.embedding_cache,
        mode=args.mode,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )
    turns = [session.ask(question) for question in args.questions]
    if args.json:
        print(json.dumps([turn.to_dict() for turn in turns], ensure_ascii=False, indent=2))
        return

    for index, turn in enumerate(turns, start=1):
        print(f"Turn {index}")
        print(f"Original question: {turn.original_question}")
        print(f"Resolved question: {turn.resolved_question}")
        print(format_answer(turn.answer))
        if index != len(turns):
            print()


if __name__ == "__main__":
    main()
