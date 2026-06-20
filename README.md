# KB Readiness Agent

KB Readiness Agent is a local RAG readiness and reliability audit tool for support
knowledge bases. It is built to answer a reviewer’s operational questions: Does
retrieval find the right evidence? Does the system refuse unsupported requests?
Are citations and claims grounded? Is an incomplete knowledge base correctly
rejected? Which evaluated answers and KB sections are affected when policy changes?

This is not a generic chatbot. Ask Mode provides a small, inspectable QA surface,
but the project’s primary outputs are reliability evidence, readiness gates,
knowledge-gap reports, and change-impact reports. The repository is a deterministic
take-home demonstration, not a production support or legal-analysis system.

## What to review

| Mode | Reviewer question | Output |
|---|---|---|
| **Ask** | Can the system answer supported questions, cite evidence, and refuse unsupported ones? | Structured answer/refusal, chunk citations, confidence, groundedness, review state |
| **Readiness Audit** | Is this KB complete and reliable enough for a limited internal pilot? | Eval metrics, Ask Mode gate, concrete knowledge gaps, readiness recommendation |
| **Change Impact** | Which evaluated answers and KB sections may be stale after a policy update? | Changed sections, risk levels, impacted eval cases, required KB updates |

Extractive Ask Mode is the default. It returns retrieved source text or a
deterministic refusal, runs locally, and requires no API key. All official tests,
audits, and the complete demo use this no-key path.

Generative Ask Mode is optional and explicit. It receives only the question,
generation contract, and retrieved chunks; it must return structured claims with
`chunk_id` citations. The generated answer is treated as an untrusted proposal and
cannot be released unless the deterministic validator accepts its citations,
provenance, and groundedness. Deterministic fake providers demonstrate both the
allowed and blocked paths without credentials.

Ask Mode applies a small deterministic router before retrieval. Standalone greetings,
thanks, empty input, and app-introduction questions return a canned capability message
with `response_type=non_kb_chitchat`. Clearly unrelated general questions such as
weather, trivia, recipes, sports, or creative requests return a canned scope refusal
with `response_type=out_of_scope_general`. Neither response retrieves evidence, calls
an LLM provider, requires citations, or enters readiness metrics. Explicit company,
support, policy, pricing, and service signals always take the existing retrieval and
validator path; ambiguous queries default to that path as well.

## Quickstart

Python 3.10 or newer is required.

```bash
python -m pip install -r requirements.txt
python -m src.ingest
python -m src.answer "標準月付用戶的退款期限是多久？" --retriever hybrid
```

Expected ingestion result: `Indexed 34 corpus chunks`. Ask Mode reads only
`corpus/`; policy fixtures under `compare_docs/` never enter the retrieval index.

The first dense or hybrid run may download
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Corpus embeddings
are cached locally after that.

Run the deterministic reviewer sequence with no API key:

```bash
./scripts/demo.sh
```

See [DEMO.md](DEMO.md) for the eight review steps and expected observations.

## Ask Mode: default extractive and optional generation

Supported question, using the default extractive mode:

```bash
python -m src.answer "標準月付用戶的退款期限是多久？" \
  --retriever hybrid
```

Unsupported question, which must refuse rather than invent an exception:

```bash
python -m src.answer \
  "Can customers get a refund after 90 days for medical reasons?" \
  --retriever hybrid
```

Reproducible validator-blocking example:

```bash
python -m src.answer \
  "客戶如果因為醫療因素，90 天後還可以退款嗎？" \
  --retriever hybrid \
  --mode generative \
  --llm-provider fake_hallucination
```

The fake backend intentionally proposes an unsupported 90-day refund exception.
The validator blocks it, retains the safe extractive refusal as the final answer,
sets low confidence, and requires human review. The rejected proposal remains
visible as `blocked_generated_answer` when `--json` is used.

MiniMax-M3, OpenAI, and Anthropic are available only as optional live integrations.
MiniMax uses `MINIMAX_API_KEY`; select it with `--llm-provider minimax`. No live
provider participates in the official deterministic no-key validation baseline.
`.env.example` is a reference; the project does not automatically load `.env`
files.

```bash
export MINIMAX_API_KEY="..."
python -m src.answer \
  "標準月付用戶的退款期限是多久？" \
  --retriever hybrid \
  --mode generative \
  --llm-provider minimax
```

MiniMax defaults to `https://api.minimax.io/v1` and `MiniMax-M3`. The optional
`MINIMAX_BASE_URL`, `MINIMAX_MODEL`, `MINIMAX_TIMEOUT_SECONDS`,
`MINIMAX_MAX_RETRIES`, and `MINIMAX_RETRY_BASE_SECONDS` variables override its
bounded request and retry settings. Live proposals still pass the same local
citation, retrieval-provenance, and groundedness gate. See
[generative_sample_runs.md](generative_sample_runs.md) for non-deterministic live
samples; they are intentionally separate from official eval results.

## Readiness Audit: healthy versus degraded

Audit the complete KB and write isolated report artifacts:

```bash
python -m eval.run_eval \
  --retriever hybrid \
  --write-report \
  --report-dir data/reports/healthy
```

Build a deterministic incomplete KB, then audit it with the same gate:

```bash
python -m src.degraded
python -m eval.run_eval \
  --retriever hybrid \
  --index data/degraded/index/chunks.jsonl \
  --write-report \
  --report-dir data/reports/degraded
```

The healthy corpus must produce Ask Mode gate `PASS` and recommendation
`Internal Pilot Ready`. The degraded fixture deliberately omits the refund policy
and selected Enterprise knowledge; it must produce gate `FAIL`, recommendation
`Not Ready`, and concrete missing topics. The degraded eval exits with status 1 by
design so automation cannot mistake an unsafe KB for a successful audit.

Audit artifacts are `metrics.json` and `readiness_report.md` in the selected report
directory.

## Change Impact: Markdown, plain text, and large PDF

Compare the Markdown policy fixtures:

```bash
python -m src.compare \
  --old compare_docs/old_refund_policy.md \
  --new compare_docs/new_refund_policy.md
```

Generate deterministic 50-page PDF fixtures and compare them:

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

Markdown uses H1/H2 structure. Headerless Markdown and `.txt` files are split into
bounded paragraph chunks. PDF loading uses PyMuPDF layout metadata to remove repeated
headers and footers, identify headings, preserve 1-based page metadata, and create
page-bounded section chunks. The complete PDF is never loaded as one prompt or
context, and Change Impact does not call an LLM.

Both fixture comparisons produce the same baseline: 6 changes, 4 high-risk
changes, 13 impacted eval cases, and 9 required KB updates. Reports are written as
`data/reports/change_impact.json` and
`data/reports/change_impact_report.md` unless another output directory is selected.

### Custom support-contract upload fixture

Generate the separate 50-page old/new pair used to exercise custom uploads:

```bash
python -m scripts.build_custom_pdf_fixtures
```

This writes:

- `compare_docs/custom_old_support_contract.pdf`
- `compare_docs/custom_new_support_contract.pdf`

The generated PDFs are intentionally gitignored. Each PDF contains repeated headers,
footers, visible page numbers, and the same section layout. The new contract changes
the Support SLA from 24 hours to 4 hours, removes a refund exception, requires manual
Enterprise refund review, changes data deletion from 30 days to 14 days, and adds a
mandatory escalation rule.

Start Streamlit, open **Change Impact**, select **Upload custom documents**, and upload
the old and new files in their respective fields. The expected deterministic result is
5 changed sections, 5 high-risk changes, required KB updates, and human review required.

## Architecture and evidence contracts

```text
corpus/*.md
  -> 34 section chunks
  -> lexical / multilingual dense / hybrid retrieval
  -> extractive answer or optional generated proposal
  -> deterministic groundedness validator
  -> Ask Mode eval gate
  -> readiness metrics and recommendation

old/new Markdown or text-based PDF
  -> normalized sections with source metadata
  -> deterministic section alignment and policy-rule comparison
  -> risk, eval impact, and required KB updates
```

`AnswerResult` preserves the question, response type, answer or refusal, retrieved chunks,
citations, confidence, human-review state, groundedness result, warnings, latency,
answer mode, validator decision, and optional generation trace. `--json` exposes
the complete schema.

Session memory is process-local. It rewrites an underspecified follow-up into a
standalone question before retrieval, then runs the same retrieval, refusal, and
groundedness path. It neither persists history nor bypasses evidence validation.

For implementation details and explicit design boundaries, see
[DESIGN.md](DESIGN.md).

## Optional UI

```bash
python -m pip install -r requirements-ui.txt
streamlit run app.py
```

The three tabs expose Ask, Readiness Audit, and Change Impact through the existing
Python APIs. Change Impact defaults to the built-in 50-page PDF demo, retains the
Markdown fixture, and accepts paired `.pdf`, `.md`, or `.txt` uploads. Uploaded source
files are compared from an operating-system temporary directory and deleted after the
run; generated JSON and Markdown reports remain available by path and download.

## Final validation baseline

```bash
python -m src.ingest
python -m unittest discover -s tests
python -m eval.run_eval --retriever hybrid --write-report
python -m src.compare --old compare_docs/old_refund_policy.md --new compare_docs/new_refund_policy.md
python -m scripts.build_large_pdf_fixture --old compare_docs/large_old_refund_policy.pdf --new compare_docs/large_new_refund_policy.pdf --pages 50
python -m src.compare --old compare_docs/large_old_refund_policy.pdf --new compare_docs/large_new_refund_policy.pdf --write-report
python -m scripts.build_custom_pdf_fixtures
./scripts/demo.sh
```

Frozen reviewer baseline:

- Current no-key validation: 70 deterministic pytest tests pass. Optional MiniMax live tests are key-gated and skip when MINIMAX_API_KEY is not configured.
- Ask Mode gate: `PASS`.
- Healthy audit: `Internal Pilot Ready`.
- Degraded audit: `Not Ready`.
- Markdown Change Impact: 6 changes, 4 high risk, 13 impacted eval cases, 9 KB updates.
- 50-page PDF Change Impact: 6 changes, 4 high risk, 13 impacted eval cases, 9 KB updates.
- `demo.sh` passes without a real API key.

## Limitations

- The corpus is six synthetic Markdown documents and the eval set is small and
  curated; passing results are not production-traffic evidence.
- Extractive answers are top-chunk source text, not synthesized support responses.
- The generative validator checks citation provenance, claim coverage, and
  numeric/date/time support, but is not a complete semantic-entailment or policy
  correctness judge.
- Retrieval thresholds and hybrid fusion weights are calibrated only for this
  local fixture.
- Ask citations are chunk-level. Markdown has no page numbers; PDF Change Impact
  sections preserve pages but do not provide sentence-level citations.
- Session memory exists only in the current process and is not a durable or
  multi-user conversation store.
- PDF comparison requires extractable text and usable visual structure. It does
  not OCR scanned PDFs or reliably interpret complex tables and ambiguous layouts.
- Change Impact is deterministic policy-rule analysis, not a semantic/legal diff,
  a full-corpus conflict scan, or automatic policy update application.
- There is no production authentication, authorization, provider observability,
  rate limiting, cost control, monitoring, or deployment hardening.
