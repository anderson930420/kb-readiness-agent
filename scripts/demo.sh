#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m src.ingest
python -m src.answer "標準月付用戶的退款期限是多久？" --retriever hybrid
python -m src.answer "客戶是否應該把醫療紀錄上傳到客服工單？" --retriever hybrid
python -m src.answer "Can customers get a refund after 90 days for medical reasons?" --retriever hybrid
python -m eval.run_eval --retriever hybrid --write-report
python -m src.compare \
  --old compare_docs/old_refund_policy.md \
  --new compare_docs/new_refund_policy.md
