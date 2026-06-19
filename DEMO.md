# Reviewer Demo Guide

This walkthrough demonstrates the repository's three product modes without adding
or modifying data. Allow extra time on the first hybrid run because the embedding
model may need to download.

## Setup

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m src.ingest
```

Expected setup result: `Indexed 34 corpus chunks`. Ask Mode indexes only `corpus/`;
the old/new documents under `compare_docs/` remain isolated.

For the optional UI:

```bash
python -m pip install -r requirements-ui.txt
streamlit run app.py
```

The UI presents the extractive Ask, Readiness Audit, and Change Impact workflows.
Optional generative mode is currently exposed through the CLI.

## 1. Ask Mode

Run these questions in order:

```bash
python -m src.answer "標準月付用戶的退款期限是多久？" --retriever hybrid --mode extractive
python -m src.answer "客戶是否應該把醫療紀錄上傳到客服工單？" --retriever hybrid
python -m src.answer "Can customers get a refund after 90 days for medical reasons?" --retriever hybrid
python -m src.answer "客戶如果因為醫療因素，90 天後還可以退款嗎？" --retriever hybrid --mode generative --llm-provider fake_hallucination
python -m src.session "What is the standard refund window?" "What about enterprise customers?" --retriever hybrid
```

Expected observations:

1. The standard refund question returns a normal extractive answer with a citation
   to the relevant refund-policy section.
2. The medical-record question returns grounded privacy/sensitive-data guidance
   with source evidence.
3. The unsupported 90-day medical exception is not invented. The result refuses
   or routes the request to manual review using policy evidence.
4. The fake generative backend intentionally claims that a 90-day medical refund
   is allowed. The validator blocks that output, preserves the extractive refusal
   as the final answer, and sets low confidence plus human review.
5. The session command resolves the second question into a standalone Enterprise
   refund question before retrieval. Its evidence includes the Enterprise refund
   FAQ, refund policy, and refund escalation SOP. Session turns exist only in the
   current process; no history is written to disk.

Point out that each result exposes confidence, citation identity, groundedness,
warnings, review state, answer mode, validator decision, and optional generation
trace. Add `--json` to any command to inspect the full `AnswerResult` contract.
Extractive mode remains the default and requires no API key. The two fake providers
also require no credentials; OpenAI and Anthropic are optional and require their
respective environment keys.

## 2. Readiness Audit

Run the healthy audit into its own report directory:

```bash
python -m eval.run_eval --retriever hybrid --write-report --report-dir data/reports/healthy
```

Then generate the reproducible incomplete corpus and audit it separately:

```bash
python -m src.degraded
python -m eval.run_eval \
  --retriever hybrid \
  --index data/degraded/index/chunks.jsonl \
  --write-report \
  --report-dir data/reports/degraded
```

The second eval command intentionally exits with status 1 because the degraded
gate fails. This is the expected audit result, not a command failure to ignore in
automation.

Expected observations:

- The healthy bilingual Ask Mode gate is `PASS` and recommends
  `Internal Pilot Ready`, deliberately narrower than production readiness.
- The degraded fixture leaves `corpus/` unchanged, omits `refund_policy.md`, and
  removes selected Enterprise FAQ sections. Its gate is `FAIL` with a `Not Ready`
  recommendation.
- The healthy and degraded reports are isolated under `data/reports/healthy/` and
  `data/reports/degraded/`. The degraded report identifies concrete missing areas,
  including refund windows, renewal refunds, refund processing, Enterprise SLA,
  and Enterprise quote handling.
- Change-impact-only eval cases remain outside the Ask Mode gate.

## 3. Change Impact

```bash
python -m src.compare --old compare_docs/old_refund_policy.md --new compare_docs/new_refund_policy.md
```

Expected observations:

- 6 changed sections are aligned and classified.
- 4 changes are high risk.
- 13 eval cases are identified as potentially stale or affected.
- 9 KB updates are recommended.
- The result requires policy-owner/human review; it does not claim legal judgment
  or automatically apply updates.

The generated outputs are `data/reports/change_impact.json` and
`data/reports/change_impact_report.md`.

## Three-minute walkthrough

1. **First minute — Ask Mode:** run the supported extractive refund question, the
   unsupported medical-exception question, and the fake hallucination. Show that
   the validator does not release the unsupported generated answer.
2. **Second minute — Readiness Audit:** compare the healthy and degraded reports.
   Emphasize that the same unchanged gate passes the complete corpus and rejects
   the intentionally incomplete fixture with actionable missing areas.
3. **Third minute — Change Impact:** compare the bundled policies. Show how a
   policy edit maps to risky sections, potentially stale eval answers, and concrete
   KB update work.

The key distinction to state: this is a deterministic RAGOps-lite review workflow
around a local extractive QA baseline with optional validated generation, not a
production chatbot or an LLM-based legal analyzer.

## Final validation checklist

```bash
python -m src.ingest
python -m unittest discover -s tests
python -m eval.run_eval --retriever hybrid --write-report
python -m src.answer "標準月付用戶的退款期限是多久？" --retriever hybrid --mode extractive
python -m src.answer "客戶如果因為醫療因素，90 天後還可以退款嗎？" --retriever hybrid --mode generative --llm-provider fake_hallucination
python -m src.compare --old compare_docs/old_refund_policy.md --new compare_docs/new_refund_policy.md
./scripts/demo.sh
```

Expected baseline:

- 34 corpus-only chunks.
- All tests pass.
- Ask Mode gate: `PASS`.
- Launch recommendation: `Internal Pilot Ready`.
- Degraded fixture: 26 chunks; gate `FAIL`; recommendation `Not Ready`.
- Change Impact: 6 changed sections, 4 high risk, 13 impacted eval cases, 9 KB
  updates.
- Generated chunks, embeddings, reports, `__pycache__/`, and `.pytest_cache/`
  remain ignored by git.
