# AI Support KB Readiness Agent — Day 5

This repository contains a bilingual Ask Mode vertical slice: a Chinese-primary
support corpus, lexical and multilingual dense retrieval, deterministic hybrid
fusion, structured extractive answers, citations, evidence-based refusal, and
deterministic groundedness checks. The bilingual eval gate can also produce a
machine-readable metrics file and a human-readable Knowledge Base Readiness
Report. Change Impact Mode compares old/new Markdown policies and identifies
possibly invalidated eval answers and KB sections with deterministic rules.

## Run it

Python 3.10 or newer is required. Install the dense retrieval dependency first:

```bash
python -m pip install -r requirements.txt
```

```bash
python -m src.ingest

python -m src.retrieve "What is the refund window for standard monthly subscribers?" --retriever lexical --top-k 3
python -m src.retrieve "標準月付用戶的退款期限是多久？" --retriever dense --top-k 3
python -m src.retrieve "客戶可以把醫療紀錄上傳到客服工單嗎？" --retriever hybrid --top-k 3

python -m src.answer "What is the refund window for standard monthly subscribers?" --retriever lexical
python -m src.answer "標準月付用戶的退款期限是多久？" --retriever dense
python -m src.answer "Can customers get a refund after 90 days for medical reasons?" --retriever hybrid
python -m src.answer "標準月付用戶的退款期限是多久？" --retriever hybrid --json
```

Default ingestion reads only `corpus/`. Files under `compare_docs/` are isolated
inputs for Change Impact Mode and are not available to Ask Mode.

The index is written to `data/index/chunks.jsonl`. Each Markdown chunk preserves
`chunk_id`, `doc`, `section`, `section_zh`, `section_slug`, `page`, and `text`.
The `content` field remains as a compatibility alias for `text`.

Dense retrieval uses `paraphrase-multilingual-MiniLM-L12-v2` by default. Override
it with `--model`. Corpus embeddings are generated on the first dense or hybrid
query and cached under `data/index/embeddings/`; the cache key changes when the
model or indexed content changes.

The answer API returns an `AnswerResult` with the question, retriever, answer,
refusal and review state, confidence, citations, retrieved chunks, groundedness,
warnings, and latency. Groundedness is a deterministic validation of citation
provenance, citation coverage, numeric claims, and refusal support; it is not an
LLM judge or a semantic correctness score.

## Ask Mode eval gate

Run the official gate across both Chinese and English active Ask Mode questions:

```bash
python -m eval.run_eval --retriever hybrid
```

The gate reports retrieval hits, refusal correctness, citation coverage,
groundedness, and per-case failures. It excludes the five P2 conflict/change
cases. Pass `--retrievers lexical,dense,hybrid` for comparative diagnostics.

## Knowledge Base Readiness Report

Run the official hybrid gate and write both Day 4 report artifacts:

```bash
python -m eval.run_eval --retriever hybrid --write-report
```

This writes:

- `data/reports/metrics.json`: stable metrics, scope counts, gate status,
  failure details, and deterministic knowledge gaps.
- `data/reports/readiness_report.md`: executive summary, metrics table,
  launch recommendation, gaps, failures, limitations, and next steps.

Generated reports are gitignored. Report generation accepts exactly one
retriever because each artifact represents one readiness gate run. The current
small synthetic corpus can be recommended for an internal pilot when every
core metric passes, but it is not classified as externally ready.

## Change Impact Mode

Compare an old and new policy document:

```bash
python -m src.compare \
  --old compare_docs/old_refund_policy.md \
  --new compare_docs/new_refund_policy.md
```

This writes `data/reports/change_impact.json` and
`data/reports/change_impact_report.md`. Add `--json` to print the structured
result or `--output-dir PATH` to change the report directory.

The comparison is structure-aware and rule-based: it aligns H1/H2 Markdown
sections by slug, normalized heading, then lexical overlap; detects explicit
numeric, eligibility, refundability, exception, and manual-review changes; and
maps them to eval cases and existing corpus sections. It does not use an LLM,
perform full-corpus conflict scanning, or claim full semantic/legal analysis.

## Regression tests

```bash
python -m unittest discover -s tests -v
```
