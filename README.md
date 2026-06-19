# AI Support KB Readiness Agent

AI Support KB Readiness Agent is a local, bilingual RAGOps-lite tool for support
knowledge bases. It answers policy questions with citations, refuses unsupported
questions, evaluates answer reliability across a curated eval set, generates a
readiness report, and analyzes policy-document changes to identify which existing
AI answers may become stale.

It is more than a RAG chatbot because the primary output is not just an answer. The
project exposes a deterministic reliability workflow around retrieval: structured
answer evidence, groundedness checks, an Ask Mode quality gate, a launch-readiness
recommendation, and policy-change impact mapping. It is a take-home demonstration,
not a production-ready support or legal-analysis system.

The demo has three modes:

- **Ask Mode:** bilingual extractive answers, chunk-level citations, and conservative
  refusal/manual-review behavior.
- **Readiness Audit:** eval metrics, gate status, knowledge gaps, and an
  `Internal Pilot Ready` or remediation recommendation.
- **Change Impact:** deterministic old/new policy comparison with changed sections,
  risk levels, impacted eval cases, and required KB updates.

## Quickstart

Python 3.10 or newer is required. No API key or `.env` file is needed.

```bash
python -m pip install -r requirements.txt
python -m src.ingest
python -m src.answer "標準月付用戶的退款期限是多久？" --retriever hybrid
python -m eval.run_eval --retriever hybrid --write-report
python -m src.compare --old compare_docs/old_refund_policy.md --new compare_docs/new_refund_policy.md
```

The first dense or hybrid run may download
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Embeddings are then
cached locally.

For the optional three-tab UI:

```bash
python -m pip install -r requirements-ui.txt
streamlit run app.py
```

The main outputs are:

- CLI `AnswerResult` summaries, with `--json` available for the full schema.
- `data/reports/metrics.json` and `data/reports/readiness_report.md`.
- `data/reports/change_impact.json` and
  `data/reports/change_impact_report.md`.

Generated chunks, embeddings, reports, and caches are gitignored.

## How the project works

```text
Corpus Markdown
→ Ingested chunks
→ Hybrid retrieval
→ AnswerResult
→ Eval gate
→ Readiness report

Old/New policy docs
→ Section alignment
→ Rule-based change detection
→ Impacted eval cases / KB updates
→ Change impact report
```

Ask Mode ingestion reads only `corpus/` and writes exactly 34 chunks to
`data/index/chunks.jsonl`. `compare_docs/` is intentionally isolated and is loaded
only by Change Impact Mode. Hybrid retrieval combines the local BM25-style lexical
path with multilingual dense retrieval using fixed score fusion.

`AnswerResult` includes the question, retriever, answer or refusal, citations,
confidence, human-review state, groundedness status, warnings, retrieved chunks,
and latency. Answers are extracts from the highest-ranked evidence chunk; the
project does not use an LLM to generate answers.

## Demo

Run the complete CLI flow:

```bash
./scripts/demo.sh
```

For the reviewer-oriented questions, expected observations, and a three-minute
walkthrough, see [DEMO.md](DEMO.md).

## Final validation

```bash
python -m src.ingest
python -m unittest discover -s tests
python -m eval.run_eval --retriever hybrid --write-report
python -m src.compare --old compare_docs/old_refund_policy.md --new compare_docs/new_refund_policy.md
./scripts/demo.sh
```

Expected baseline:

- Ingestion reports `Indexed 34 corpus chunks` from `corpus/` only.
- All tests pass.
- The Ask Mode gate is `PASS`.
- The readiness recommendation is `Internal Pilot Ready`, not external production
  readiness.
- Change Impact reports 6 changed sections, 4 high-risk changes, 13 impacted eval
  cases, and 9 required KB updates.
- Generated runtime files remain ignored by git.

## Known limitations

- The corpus contains six synthetic Markdown documents and the eval set is small
  and curated; results are not statistically representative of production traffic.
- Answers are top-chunk extracts or deterministic refusals, not LLM-generated
  responses.
- Groundedness checks citation provenance and coverage, numeric claims, and refusal
  support; it is not semantic answer correctness or an LLM judge.
- Citations are chunk-level. Markdown sources have no page numbers.
- Retrieval thresholds and hybrid fusion weights are calibrated only for this local
  dataset.
- Change Impact depends on document structure and explicit policy-language rules.
  It is not a semantic/legal diff, a full-corpus conflict scan, or automatic policy
  update application.
- The demo has no production authentication, authorization, monitoring, or
  deployment hardening.

See [DESIGN.md](DESIGN.md) for implementation details, metric definitions, design
tradeoffs, and explicit scope boundaries.
