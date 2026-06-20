# AI Support KB Readiness Agent

> English version: [README.md](README.md)

## 專案概要

這是一個以 Streamlit 呈現的 RAG readiness（檢索增強生成就緒度）與可靠性稽核工具，用來評估企業客服知識庫是否足以安全支援 AI assistant 上線。

它不是一般的文件問答 chatbot。這個系統關注的不只是「AI 能不能回答」，而是「AI 什麼時候可以回答、什麼時候應該拒答、什麼時候應該交給人工複核」。主要輸出包括引用證據、拒答結果、groundedness（回答是否有知識庫依據）風險、readiness report，以及政策變更影響報告。

這是一個範圍明確、以 deterministic（可重現）流程為主的作品集 demo，不是 production-ready 的客服或法務分析系統。

## Demo 影片

**1. 完整功能展示 — 模式、provider、問答與引用**
切換 retriever／answer mode／provider（`fake_hallucination` 與 `minimax`），提出可回答與
不可回答的問題，並展示 citation 與 validator 的判決。（約 5 分半）

[![完整功能展示](https://img.youtube.com/vi/YyByXKVbiyU/maxresdefault.jpg)](https://youtu.be/YyByXKVbiyU)

**2. Edge case 處理 — 範圍外與非知識庫問題**
面對不相關或偏離任務的提問（例如「今天天氣如何」「幫我寫一個 Python script」）時，
系統如何回應而不是硬掰答案。（約 1 分半）

[![Edge case 處理](https://img.youtube.com/vi/QG-6bGVLawQ/maxresdefault.jpg)](https://youtu.be/QG-6bGVLawQ)

## Demo 情境

Demo 使用一組虛構的 SaaS 客服知識庫，內容包括：

- refund policy
- pricing policy
- privacy policy
- enterprise plan FAQ
- onboarding guide
- support escalation SOP
- old and new refund policy PDFs for change-impact analysis

核心情境是：

> 在部署 AI 客服助理之前，能否先確認客服知識庫具備足夠的可回答性、引用依據與安全邊界？

Streamlit 的 Ask 頁面預設使用 `hybrid` retriever、`extractive` answer mode，以及問題 `標準月付用戶的退款期限是多久？`。切換到 `generative` 後才會顯示 provider selector，UI 預設選擇 `fake_hallucination`，需要 live generation 時再選擇 `minimax`。Readiness Audit 使用官方的 hybrid evaluation gate。Change Impact 預設比較系統產生的 50-page 新舊 refund policy PDF，也可改用 Markdown fixture 或上傳自訂文件。

## Streamlit Demo 展示什麼

### 1. Ask Mode

Ask Mode 展示核心 RAG 回答流程：

- 從客服知識庫檢索相關 chunks
- 回答有知識庫依據的問題並顯示 citation
- 在同一個 session 內解析簡短 follow-up question
- 知識庫依據不足時拒答或要求人工確認
- 比較不同 retrieval strategy
- 比較 extractive 與 generative 回答方式
- 當生成內容包含 unsupported claim 時顯示 groundedness risk

建議依序展示以下問題：

```text
標準月付用戶的退款期限是多久？
那年度用戶呢？
客戶如果因為醫療因素，90 天後還可以退款嗎？
退款升級到人工審查的情境有哪些？
```

### 2. Audit Mode

Audit Mode 把 evaluation harness 轉成面向產品決策的 readiness report。它檢查：

- answerability 與 retrieval 結果
- unsupported question 是否正確拒答
- citation 與 groundedness coverage
- 每個失敗案例與具體 knowledge gaps
- Ask Mode gate 狀態
- 有範圍限制的 readiness recommendation

完整 fixture 預期得到 Ask Mode gate `PASS` 與 `Internal Pilot Ready`；這只是目前 corpus 與 eval set 的 gate 結果，不代表 production-ready。專案也提供 deterministic degraded fixture，刻意移除 refund policy 與部分 Enterprise 知識，預期得到 `FAIL`、`Not Ready` 與具體缺漏主題。

### 3. Change Impact Mode

Change Impact Mode 比較新舊 Markdown、純文字或可擷取文字的 PDF 政策文件。它不是只輸出文字 diff，而是整理：

- 已變更的政策規則
- 變更風險等級
- 可能受影響的 eval questions
- 需要更新的知識庫項目
- 應交由人工複核的答案

內建 refund policy 情境包含 `refund window: 14 days -> 7 days`。系統會按 section 對齊規則並產生 JSON 與 Markdown 報告；Change Impact 流程不呼叫 LLM，也不會自動套用政策更新。

## 功能總覽

| 範圍 | 功能 | 用途 |
|---|---|---|
| Retrieval | lexical retriever | 處理精準關鍵字、政策詞與數字匹配 |
| Retrieval | dense retriever | 處理不同措辭與多語語意匹配 |
| Retrieval | hybrid retriever | 結合 lexical 與 dense，是 Streamlit 的預設路徑 |
| Answering | extractive mode | 產生貼近 retrieved evidence 的保守回答 |
| Answering | generative mode | 產生較自然的回答，但仍需 citation 與 groundedness validation |
| Provider | minimax | 正常生成路徑使用的選用外部 LLM provider |
| Provider | fake_hallucination | 故意不安全的 mock provider 與 reliability test fixture |
| Reliability | refusal behavior | KB 依據不足時拒答或升級人工處理 |
| Reliability | citations | 顯示支援答案的來源 chunks |
| Reliability | groundedness check | 暴露未被來源支持的 generated claims |
| Audit | readiness report | 彙整有範圍限制的 AI assistant 部署判斷依據 |
| Change Impact | policy comparison | 標記可能讓既有答案失效的政策變更 |

## Ask Mode 設計

Ask Mode 在 UI 中公開三個可設定層，讓使用者能直接觀察 retrieval、answer generation 與 validation 的差異。

### Retriever

- `lexical`：BM25-style 關鍵字檢索，適合精準政策詞、專有名詞與數字。
- `dense`：以 multilingual embeddings 進行語意檢索，適合使用者改用不同說法或不同語言提問。
- `hybrid`：將 lexical 與 dense 分數正規化後等權合併，是主要 Streamlit demo 與官方 readiness audit 的預設路徑。

為了相容既有 CLI，answer CLI 的程式預設仍是 `lexical`；本文件中的 demo commands 會明確指定 `--retriever hybrid`。

### Answer Mode

- `extractive`：預設的保守模式。直接回傳排名最高的來源內容，或執行 deterministic refusal；不需要 API key。
- `generative`：要求選定 provider 根據 retrieved chunks 產生結構化、帶引用的回答。只有通過 chunk citation、retrieval provenance 與 groundedness validation 的 proposal 才能成為最終答案。

這兩種模式用來呈現可靠性與自然語言流暢度之間的取捨。官方 deterministic tests、audits 與完整 demo 預設都走不需要 key 的 extractive path，除非指令明確選擇 generative provider。

### LLM Provider

Provider 選單只會在 `generative` mode 顯示。

- `minimax`：正常 demo path 使用的真實外部 LLM provider，需要 `MINIMAX_API_KEY`。
- `fake_hallucination`：故意設計成不安全的 mock provider，用來模擬沒有知識庫依據的 generated claim。它不需要 credentials，也不會發出外部網路請求。

不論使用哪一個 provider，生成內容都先被視為 untrusted proposal。系統不會因為回答看起來流暢，就直接把它當成可信答案。

Ask Mode 在 retrieval 前還有 deterministic positive-admission router。單純問候、致謝與 app 介紹問題會得到固定 capability message；明顯無關的 weather、trivia 或 creative request 會收到 scope refusal；資訊量過低的輸入會在 retrieval 前被拒絕。這些 canned responses 不呼叫 provider，也不納入 readiness metrics。

## 為什麼有 `minimax` 和 `fake_hallucination`

### `minimax`

`minimax` 用來展示正常生成路徑。它只接收問題、generation contract 與 retrieved chunks。即使外部 LLM 回傳流暢答案，仍必須引用實際 retrieved `chunk_id`，並通過 citation provenance 與 groundedness check。

MiniMax 是選用整合，不屬於官方 no-key validation baseline。非 deterministic 的 live sample 另放在 [generative_sample_runs.md](generative_sample_runs.md)。

### `fake_hallucination`

`fake_hallucination` 不是實際 LLM provider，而是 reliability test fixture。它會刻意提出「醫療因素可在 90 天後退款」這類未被知識庫支持的內容，用來測試 validation layer 是否能辨識 unsupported claim。

預期行為是 validator 阻擋不安全 proposal，保留安全的 extractive refusal 作為最終答案，將 confidence 設為 low，並要求人工複核。使用 `--json` 時，被拒絕的內容仍會保留在 `blocked_generated_answer`，方便稽核。

因此兩者分工是：

> `minimax` 展示正常生成。
>
> `fake_hallucination` 展示 validation layer 為什麼必要。

這項設計的重點是 reliability，而不是只追求生成內容流暢。

## 這個系統能做到什麼

這個系統可以：

- 根據客服知識庫回答有依據的問題
- 顯示 chunk-level citation 與引用來源
- 在知識庫沒有足夠依據時拒答或要求人工確認
- 在目前 process 內解析簡單 follow-up question
- 比較 `lexical`、`dense`、`hybrid` retrieval
- 比較 `extractive`、`generative` answer mode
- 使用 `minimax` 串接真實外部 LLM provider
- 使用 `fake_hallucination` 模擬不可靠生成
- 阻擋 unsupported generated claims 並顯示 groundedness risk
- 透過目前 eval set 產生 readiness report
- 報告 answerability、citation coverage、refusal behavior 與 knowledge gaps
- 比較支援格式的新舊政策文件
- 找出可能受政策變更影響的 eval questions 與知識庫項目

## 這個系統不能做到什麼

這個系統不會：

- 保證找出全語料中所有可能的矛盾
- 取代人工法務、政策或客服審查
- 在發現政策變更後自動更新知識庫
- 實作 user-level document permission 或 row-level access control
- 提供 production-grade monitoring、alerting、安全性或 deployment hardening
- 在沒有代表性 eval set 的情況下，證明任意外部知識庫已經 ready
- 在沒有 citation 與 groundedness check 的情況下，保證 generative answer 正確

本專案的範圍是 RAG reliability、readiness evaluation 與 policy change impact analysis 的聚焦型 demo。

## 如何執行

需要 Python 3.10 或更新版本。

```bash
python -m pip install -r requirements.txt
python -m src.ingest
python -m src.answer "標準月付用戶的退款期限是多久？" --retriever hybrid
```

預期 ingestion 結果為 `Indexed 34 corpus chunks`。Ask Mode 只讀取 `corpus/`；`compare_docs/` 下的政策比較 fixtures 不會進入 retrieval index。

第一次執行 dense 或 hybrid retrieval 時，可能會下載 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。之後 corpus embeddings 會使用本機快取。

不使用 API key 執行完整 reviewer sequence：

```bash
./scripts/demo.sh
```

八個 review steps 與預期結果請見 [DEMO.md](DEMO.md)。

### CLI Ask Mode

有知識庫依據的 extractive 問題：

```bash
python -m src.answer \
  "標準月付用戶的退款期限是多久？" \
  --retriever hybrid
```

依據不足、應拒絕捏造例外的問題：

```bash
python -m src.answer \
  "Can customers get a refund after 90 days for medical reasons?" \
  --retriever hybrid
```

可重現的 validator-blocking 範例：

```bash
python -m src.answer \
  "客戶如果因為醫療因素，90 天後還可以退款嗎？" \
  --retriever hybrid \
  --mode generative \
  --llm-provider fake_hallucination
```

選用 MiniMax live generation：

```bash
export MINIMAX_API_KEY="..."
python -m src.answer \
  "標準月付用戶的退款期限是多久？" \
  --retriever hybrid \
  --mode generative \
  --llm-provider minimax
```

MiniMax 預設使用 `https://api.minimax.io/v1` 與 `MiniMax-M3`。可透過 `MINIMAX_BASE_URL`、`MINIMAX_MODEL`、`MINIMAX_TIMEOUT_SECONDS`、`MINIMAX_MAX_RETRIES`、`MINIMAX_RETRY_BASE_SECONDS` 覆寫有界的 request 與 retry 設定。`.env.example` 只是參考；專案不會自動載入 `.env`。

Process-local follow-up 範例：

```bash
python -m src.session \
  "What is the standard refund window?" \
  "What about enterprise customers?" \
  --retriever hybrid
```

## 啟動 Streamlit App

```bash
python -m pip install -r requirements-ui.txt
streamlit run app.py
```

三個 tabs 分別對應 Ask、Readiness Audit 與 Change Impact，並直接使用既有 Python APIs。第一次執行內建 large PDF demo 時，系統會依需要產生 50-page PDF pair。

## 執行 Evaluation / Audit

稽核完整知識庫並輸出獨立 report artifacts：

```bash
python -m eval.run_eval \
  --retriever hybrid \
  --write-report \
  --report-dir data/reports/healthy
```

建立 deterministic incomplete KB，並使用相同 gate 執行稽核：

```bash
python -m src.degraded
python -m eval.run_eval \
  --retriever hybrid \
  --index data/degraded/index/chunks.jsonl \
  --write-report \
  --report-dir data/reports/degraded
```

Degraded audit 的 exit status 1 是預期的 machine-readable `Not Ready` 結果，不代表基礎設施執行失敗。輸出 artifacts 為指定 report directory 內的 `metrics.json` 與 `readiness_report.md`。

## 執行 Change Impact

比較 Markdown policy fixtures：

```bash
python -m src.compare \
  --old compare_docs/old_refund_policy.md \
  --new compare_docs/new_refund_policy.md
```

產生並比較 deterministic 50-page PDF fixtures：

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

若未指定其他 output directory，報告會寫入 `data/reports/change_impact.json` 與 `data/reports/change_impact_report.md`。

產生用於測試自訂上傳的另一組 50-page support-contract PDFs：

```bash
python -m scripts.build_custom_pdf_fixtures
```

這會在 `compare_docs/` 產生被 gitignore 的 `custom_old_support_contract.pdf` 與 `custom_new_support_contract.pdf`。在 Streamlit 選擇 **Change Impact**、**Upload custom documents** 並上傳兩個檔案。預期結果為 5 個 changed sections、5 個 high-risk changes、需要更新 KB，且需要 human review。

## 測試與驗證

執行穩定測試套件：

```bash
python -m pytest
```

完整 validation sequence：

```bash
python -m src.ingest
python -m pytest
python -m eval.run_eval --retriever hybrid --write-report
python -m src.compare --old compare_docs/old_refund_policy.md --new compare_docs/new_refund_policy.md
python -m scripts.build_large_pdf_fixture --old compare_docs/large_old_refund_policy.pdf --new compare_docs/large_new_refund_policy.pdf --pages 50
python -m src.compare --old compare_docs/large_old_refund_policy.pdf --new compare_docs/large_new_refund_policy.pdf --write-report
python -m scripts.build_custom_pdf_fixtures
./scripts/demo.sh
```

目前 frozen reviewer baseline：

- 68 個 deterministic pytest tests 通過；5 個選用 MiniMax live tests 需要 `MINIMAX_API_KEY`，未設定時會 skip。共 73 個 tests，最近一次設定 key 的 validation run 為 `73 passed`。
- Ask Mode gate：`PASS`。
- Healthy audit：`Internal Pilot Ready`。
- Degraded audit：`Not Ready`。
- Markdown Change Impact：6 個 changes、4 個 high risk、13 個 impacted evaluation cases、9 個 KB updates。
- 50-page PDF Change Impact：6 個 changes、4 個 high risk、13 個 impacted evaluation cases、9 個 KB updates。
- `demo.sh` 不需要真實 API key 即可通過。

## 專案結構

```text
app.py                         三個 workflows 的 Streamlit UI
corpus/                        六份虛構客服 KB Markdown 文件
compare_docs/                  Change Impact 的新舊政策 fixtures
eval/eval_set.jsonl            中英文 evaluation cases
eval/run_eval.py               Readiness evaluation runner
src/ingest.py                  依文件結構進行 corpus ingestion
src/retrieve.py                Lexical、dense 與 hybrid retrieval
src/answer.py                  Routing、回答、拒答與 validation flow
src/generation.py              Live 與 deterministic generation providers
src/audit.py                   Readiness metrics 與 reports
src/compare.py                 Change Impact analysis 與 reports
src/document_loader.py         Markdown、text 與 PDF section loading
src/session.py                 Process-local follow-up resolution
scripts/                       Demo 與 deterministic fixture builders
tests/                         Reliability 與 regression tests
DEMO.md                        Reviewer 操作步驟與預期結果
DESIGN.md                      詳細實作決策與系統邊界
```

## 架構與證據契約

```text
corpus/*.md
  -> 34 section chunks
  -> lexical / multilingual dense / hybrid retrieval
  -> extractive answer or optional generated proposal
  -> deterministic groundedness validator
  -> Ask Mode eval gate
  -> readiness metrics and recommendation

old/new Markdown, text, or text-based PDF
  -> normalized sections with source metadata
  -> deterministic section alignment and policy-rule comparison
  -> risk, evaluation impact, and required KB updates
```

`AnswerResult` 保留 question、response type、answer 或 refusal、retrieved chunks、citations、confidence、human-review state、groundedness result、warnings、latency、answer mode、validator decision，以及選用 generation trace。使用 `--json` 可查看完整 schema。

Session memory 只存在目前 process。系統會先將資訊不足的 follow-up 改寫為獨立問題，再執行相同 retrieval、refusal 與 groundedness flow；它不會持久化歷史，也不會繞過 evidence validation。

更多實作細節與明確邊界請見 [DESIGN.md](DESIGN.md)。

## 設計取捨

- 預設 extractive path 可重現、容易稽核，但回答是來源 chunks，不是重新合成的客服回覆。
- Lexical 適合精準詞彙；dense 處理語意與中英文變化；hybrid 使用簡單的 min-max score normalization 與等權融合，不是 RRF，也沒有 reranker。
- Positive KB admission 可阻止明顯 chitchat 與無關問題進入 RAG，但規則與 relevance thresholds 只針對目前小型 fixture 校準。
- Generative output 一律視為 untrusted proposal。Deterministic validation 提高可追蹤性，但不等同完整 semantic entailment 或 policy correctness review。
- Section-aligned Change Impact 不需要把整份文件放入單一 prompt，但依賴文件具有可擷取文字與可用結構。

## 已知限制

- Corpus 只有六份虛構 Markdown 文件，eval set 規模小且經人工挑選；通過目前測試不代表 production traffic 的可靠性。
- Groundedness validator 檢查 citation provenance、claim coverage 與 numeric/date/time support，但不是完整 semantic-entailment judge。
- Retrieval thresholds 與 hybrid fusion weights 只針對目前 local fixture 校準。
- Ask citations 為 chunk-level；Markdown 沒有頁碼，PDF Change Impact 保留 section page，但不提供 sentence-level citations。
- Session memory 不持久化，也不是 multi-user conversation store。
- PDF comparison 不支援 scanned PDF OCR，也無法可靠解析複雜表格與模糊 layout。
- Query-relevant conflict check 是保守 hook，不是完整的 corpus-wide contradiction detection。
- 系統沒有 production authentication、authorization、provider observability、rate limiting、cost control、monitoring 或 deployment hardening。
