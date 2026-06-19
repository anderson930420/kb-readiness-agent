# Optional live generation sample runs

This artifact records an optional MiniMax-M3 live success and its controlled
guardrail companion case. Live output is non-deterministic. Neither case is an
input to the official deterministic no-key eval gate or its 100% baseline table.

## MiniMax success case

Run:

```bash
MINIMAX_API_KEY="$MINIMAX_API_KEY" python -m src.answer \
  "標準月付用戶的退款期限是多久？" \
  --retriever lexical \
  --mode generative \
  --llm-provider minimax \
  --json
```

Expected evidence: `validator_decision` is `allowed`, every released citation
resolves to a retrieved `chunk_id`, and groundedness is `supported`.

Observed on 2026-06-19 with the default `MiniMax-M3` model:

- Proposal answer: `根據標準退款期限政策，標準月付訂閱用戶可於首次購買後 7 天內申請退款。`
- Claim citation: `refund-policy-standard_refund_window-001`
- Validator: `allowed`; groundedness: `supported`; validation errors: none

## Blocked generated-answer case

The reproducible controlled case is covered by
`test_minimax_unsupported_medical_exception_is_blocked`. It proposes an invented
medical 90-day refund exception while labeling the provider as MiniMax. Expected
result: `validator_decision` is `blocked`, the deterministic refusal remains the
final answer, human review is required, and the rejected text remains in
`blocked_generated_answer`.

Observed controlled proposal:

- Rejected answer: `Medical circumstances allow a refund 90 days after purchase.`
- Rejected citation: `invented-medical-policy`
- Validator: `blocked`; the existing extractive medical-exception refusal remained
  final and `requires_human_review` was `true`

## Regenerating locally

```bash
MINIMAX_API_KEY="$MINIMAX_API_KEY" python -m pytest -k "minimax or generation"
```

If no key is configured, the live tests skip automatically. The controlled
`fake_hallucination` demo remains the no-key, reproducible proof that validation
blocks unsupported generation.
