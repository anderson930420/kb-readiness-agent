# Reviewer Demo Guide

This sequence demonstrates all three product modes without a real LLM provider or
API key. It is ordered around reviewer decisions: verify grounded answers and safe
refusal first, prove that generated hallucinations cannot pass the release gate,
then inspect complete versus incomplete KB readiness and document-change impact.

## Setup

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m src.ingest
```

Expected observation: ingestion reports `Indexed 34 corpus chunks`. Ask Mode reads
only `corpus/`; `compare_docs/` remains isolated from retrieval.

The first hybrid command may download the multilingual embedding model. Later runs
reuse its local embedding cache.

## 1. Normal grounded answer

```bash
python -m src.answer \
  "標準月付用戶的退款期限是多久？" \
  --retriever hybrid \
  --mode extractive
```

Expected observations:

- The answer is source text from the standard refund-window policy.
- The result includes a chunk citation, groundedness `supported`, and no human-review
  requirement.
- `Answer mode` is `extractive` and `Generation validator` is `not_run`.

This is the default no-API-key path. `--mode extractive` is shown only to make the
review step explicit.

## 2. Unsupported question refusal

```bash
python -m src.answer \
  "Can customers get a refund after 90 days for medical reasons?" \
  --retriever hybrid \
  --mode extractive
```

Expected observations:

- The system does not invent a medical exception.
- The final result refuses or escalates based on retrieved policy evidence.
- The result is grounded and marked for human review where policy confirmation is
  required.

## 3. Fake LLM hallucination blocked

```bash
python -m src.answer \
  "客戶如果因為醫療因素，90 天後還可以退款嗎？" \
  --retriever hybrid \
  --mode generative \
  --llm-provider fake_hallucination
```

Expected observations:

- The deterministic fake backend proposes an unsupported 90-day medical refund.
- The validator reports `blocked` because the numeric claim is not supported by
  the cited retrieved chunks.
- The unsafe text does not become the final answer. The safe extractive refusal is
  preserved, confidence is low, and human review is required.
- With `--json`, the rejected proposal is visible in
  `blocked_generated_answer` for auditability.

This proves validator behavior reproducibly; it does not call a real model.

## 4. Healthy readiness audit

```bash
python -m eval.run_eval \
  --retriever hybrid \
  --write-report \
  --report-dir data/reports/healthy
```

Expected observations:

- The bilingual Ask Mode gate is `PASS`.
- The readiness recommendation is `Internal Pilot Ready`, not production ready.
- Metrics and the Markdown report are isolated under
  `data/reports/healthy/`.
- Change-impact-only eval cases remain outside the Ask Mode gate.

## 5. Degraded readiness audit

```bash
python -m src.degraded
python -m eval.run_eval \
  --retriever hybrid \
  --index data/degraded/index/chunks.jsonl \
  --write-report \
  --report-dir data/reports/degraded
```

Expected observations:

- The fixture contains 26 chunks, leaves the primary corpus unchanged, omits
  `refund_policy.md`, and removes selected Enterprise FAQ sections.
- The same readiness gate is `FAIL` and the recommendation is `Not Ready`.
- The report names concrete gaps: refund windows, renewal refunds, refund
  processing, Enterprise SLA, and Enterprise quote handling.
- The eval command exits with status 1 by design. That non-zero status is the
  machine-readable audit result, not an infrastructure failure.

The full demo script handles this expected status and continues only when it is
exactly 1.

## 6. Process-local session follow-up

```bash
python -m src.session \
  "What is the standard refund window?" \
  "What about enterprise customers?" \
  --retriever hybrid
```

Expected observations:

- Before retrieval, the second turn is resolved into a standalone Enterprise
  refund question using the previous turn.
- Retrieval then runs normally and cites Enterprise refund evidence.
- Follow-up resolution does not bypass refusal or groundedness validation.
- Turns exist only in this process; no session history is written to disk.

## 7. Markdown Change Impact

```bash
python -m src.compare \
  --old compare_docs/old_refund_policy.md \
  --new compare_docs/new_refund_policy.md
```

Expected observations:

- 6 changed sections are aligned and classified.
- 4 changes are high risk.
- 13 eval cases are identified as potentially stale or affected.
- 9 KB updates are recommended.
- The result requires policy-owner review and does not apply updates.

## 8. Large PDF Change Impact

```bash
python -m scripts.build_large_pdf_fixture \
  --old compare_docs/large_old_refund_policy.pdf \
  --new compare_docs/large_new_refund_policy.pdf \
  --pages 50
python -m src.compare \
  --old compare_docs/large_old_refund_policy.pdf \
  --new compare_docs/large_new_refund_policy.pdf \
  --write-report
```

Expected observations:

- The PDF result matches the Markdown baseline: 6 changes, 4 high risk, 13
  impacted eval cases, and 9 KB updates.
- Changed sections preserve 1-based page metadata; meaningful fixture sections are
  on pages 2, 3, 11, 19, 27, and 35 among unchanged boilerplate.
- PyMuPDF extracts layout and page metadata, then comparison runs section by
  section. The complete 50-page document is never loaded as one prompt.
- Reports are written to `data/reports/change_impact.json` and
  `data/reports/change_impact_report.md`.

## Run the complete deterministic demo

```bash
./scripts/demo.sh
```

The script executes the same eight steps, treats the degraded audit’s status 1 as
expected, and passes without OpenAI or Anthropic credentials.

## Reviewer baseline

- 49 tests pass.
- Ask Mode gate: `PASS`.
- Healthy audit: `Internal Pilot Ready`.
- Degraded audit: `Not Ready`.
- Markdown and 50-page PDF Change Impact: 6 changes, 4 high risk, 13 impacted
  eval cases, 9 KB updates.
- `demo.sh` passes.

Use `--json` on Ask or Change Impact commands when the complete structured evidence
contract is more useful than the CLI summary.
