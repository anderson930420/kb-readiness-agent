# AI Support KB Readiness Agent — Day 1

This repository contains the smallest bilingual Ask Mode vertical slice: a
Chinese-primary support corpus, bilingual lexical retrieval, extractive answers,
citations, and evidence-based refusal. Day 1 uses only the Python standard
library.

## Run it

Python 3.10 or newer is required.

```bash
python -m src.ingest

python -m src.answer "What is the refund window for standard monthly subscribers?"
python -m src.answer "標準月付用戶的退款期限是多久？"
python -m src.answer "Can customers get a refund after 90 days for medical reasons?"
python -m src.answer "客戶可以把醫療紀錄上傳到客服工單嗎？"
```

Default ingestion reads only `corpus/`. Files under `compare_docs/` are reserved
for a later Change Impact workflow and are not available to Ask Mode.

The index is written to `data/index/chunks.jsonl`. Each Markdown chunk preserves
`chunk_id`, `doc`, `section`, `section_zh`, `section_slug`, `page`, and `text`.
The `content` field remains as a compatibility alias for `text`.

## Regression tests

```bash
python -m unittest discover -s tests -v
```
