# AI Support KB Readiness Agent — Day 6

This project is a local, bilingual demo for assessing whether a support knowledge
base is ready for an AI assistant. It provides extractive Ask Mode answers with
citations and deterministic groundedness checks, an official readiness eval gate,
and rule-based Markdown policy change impact analysis. It does not use an LLM to
generate answers or make legal judgments.

## Setup

Python 3.10 or newer is required.

```bash
python -m pip install -r requirements.txt
```

The optional UI has one additional dependency:

```bash
python -m pip install -r requirements-ui.txt
```

No API key or `.env` file is required. Dense and hybrid retrieval use
`paraphrase-multilingual-MiniLM-L12-v2` locally; the first run may download the
model.

## Index the Ask Mode corpus

```bash
python -m src.ingest
```

This indexes only `corpus/` and writes exactly 34 chunks to
`data/index/chunks.jsonl`. Files in `compare_docs/` remain isolated from Ask Mode.
Dense embeddings are cached under `data/index/embeddings/`.

## Demo flow

### 1. Ask Mode

```bash
python -m src.answer "標準月付用戶的退款期限是多久？" --retriever hybrid
python -m src.answer "客戶是否應該把醫療紀錄上傳到客服工單？" --retriever hybrid
python -m src.answer "Can customers get a refund after 90 days for medical reasons?" --retriever hybrid
```

Each result includes an extractive answer or refusal, citations, confidence,
human-review state, groundedness status, and warnings. Add `--json` for the full
structured `AnswerResult`. `lexical`, `dense`, and `hybrid` retrievers are
available; hybrid is recommended for this demo.

### 2. Readiness Audit

```bash
python -m eval.run_eval --retriever hybrid --write-report
```

The official gate evaluates Chinese and English Ask Mode cases and writes:

- `data/reports/metrics.json`
- `data/reports/readiness_report.md`

### 3. Change Impact

```bash
python -m src.compare \
  --old compare_docs/old_refund_policy.md \
  --new compare_docs/new_refund_policy.md
```

This writes:

- `data/reports/change_impact.json`
- `data/reports/change_impact_report.md`

To run the complete CLI sequence:

```bash
./scripts/demo.sh
```

## Optional Streamlit UI

After installing `requirements-ui.txt` and indexing the corpus:

```bash
streamlit run app.py
```

The UI has three tabs: Ask, Readiness Audit, and Change Impact. It calls the same
Python APIs as the CLI and writes reports to the same gitignored locations.

## Tests

```bash
python -m unittest discover -s tests
```

## Expected outputs

- Ingestion reports `Indexed 34 corpus chunks`.
- The current official hybrid readiness gate passes and recommends
  `Internal Pilot Ready`, not external production readiness.
- The bundled old/new refund policies produce deterministic high-risk changes,
  impacted eval cases, required KB updates, and human-review recommendations.
- Generated chunks, embeddings, and reports remain untracked because they are
  gitignored.

## Known limitations

- The corpus is six synthetic Markdown documents and the eval set is small and
  curated; results are not statistically representative of production traffic.
- Answers are top-chunk extracts or deterministic refusals, not LLM-generated
  responses.
- Groundedness validates citation provenance, citation coverage, numeric claims,
  and refusal support; it is not semantic answer correctness.
- Citations are chunk-level. Markdown sources have no page numbers.
- Retrieval thresholds and hybrid fusion weights are calibrated only for the
  current local dataset.
- Change Impact uses explicit structure and policy-language rules. It is not a
  full-corpus conflict scan, semantic diff, or legal analysis.
