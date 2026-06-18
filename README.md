# AI Support KB Readiness Agent — Day 2

This repository contains a bilingual Ask Mode vertical slice: a Chinese-primary
support corpus, lexical and multilingual dense retrieval, deterministic hybrid
fusion, extractive answers, citations, and evidence-based refusal.

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
```

Default ingestion reads only `corpus/`. Files under `compare_docs/` are reserved
for a later Change Impact workflow and are not available to Ask Mode.

The index is written to `data/index/chunks.jsonl`. Each Markdown chunk preserves
`chunk_id`, `doc`, `section`, `section_zh`, `section_slug`, `page`, and `text`.
The `content` field remains as a compatibility alias for `text`.

Dense retrieval uses `paraphrase-multilingual-MiniLM-L12-v2` by default. Override
it with `--model`. Corpus embeddings are generated on the first dense or hybrid
query and cached under `data/index/embeddings/`; the cache key changes when the
model or indexed content changes.

## Retrieval diagnostics

Compare all retrieval backends across both Chinese and English eval questions:

```bash
python -m eval.run_eval --top-k 3
```

## Regression tests

```bash
python -m unittest discover -s tests -v
```
