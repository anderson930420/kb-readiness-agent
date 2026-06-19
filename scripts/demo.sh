#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m src.ingest
python -m src.answer "標準月付用戶的退款期限是多久？" --retriever hybrid --mode extractive
python -m src.answer "Can customers get a refund after 90 days for medical reasons?" \
  --retriever hybrid \
  --mode extractive
python -m src.answer "客戶如果因為醫療因素，90 天後還可以退款嗎？" \
  --retriever hybrid \
  --mode generative \
  --llm-provider fake_hallucination
python -m eval.run_eval \
  --retriever hybrid \
  --write-report \
  --report-dir data/reports/healthy
python -m src.degraded
if python -m eval.run_eval \
  --retriever hybrid \
  --index data/degraded/index/chunks.jsonl \
  --write-report \
  --report-dir data/reports/degraded; then
  echo "Degraded corpus unexpectedly passed the readiness gate" >&2
  exit 1
else
  status=$?
  if [[ $status -ne 1 ]]; then
    exit "$status"
  fi
fi
python -m src.session \
  "What is the standard refund window?" \
  "What about enterprise customers?" \
  --retriever hybrid
python -m src.compare \
  --old compare_docs/old_refund_policy.md \
  --new compare_docs/new_refund_policy.md
python -m scripts.build_large_pdf_fixture \
  --old compare_docs/large_old_refund_policy.pdf \
  --new compare_docs/large_new_refund_policy.pdf \
  --pages 50
python -m src.compare \
  --old compare_docs/large_old_refund_policy.pdf \
  --new compare_docs/large_new_refund_policy.pdf \
  --write-report
