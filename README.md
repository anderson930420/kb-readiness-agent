# AI Support KB Readiness Agent

> 繁體中文版本：[README.zh-TW.md](README.zh-TW.md)

A Streamlit-based RAG readiness and reliability audit tool for enterprise support
knowledge bases.

This project is not only a chatbot over documents. It is designed to answer a more
operational question:

> Is this knowledge base safe and complete enough to power an AI support assistant?

The system treats RAG reliability as a product feature: it shows when the assistant
can answer, when it should refuse, and when a human should review the knowledge
base. It is a focused, deterministic take-home demonstration, not a production
support or legal-analysis system.

## Demo Videos

**1. Full walkthrough — modes, providers, Q&A, and citations**
Switching retriever / answer mode / provider (`fake_hallucination` and `minimax`),
asking supported and unsupported questions, and showing citations and the validator
decision. (~5.5 min)

[![Full walkthrough demo](https://img.youtube.com/vi/YyByXKVbiyU/maxresdefault.jpg)](https://youtu.be/YyByXKVbiyU)

**2. Edge-case handling — out-of-scope and non-KB queries**
How the system responds to unrelated or off-task prompts (e.g. "today's weather",
"write me a Python script") instead of hallucinating an answer. (~1.5 min)

[![Edge-case handling demo](https://img.youtube.com/vi/QG-6bGVLawQ/maxresdefault.jpg)](https://youtu.be/QG-6bGVLawQ)

## Demo Scenario

The demo uses a fictional SaaS customer-support knowledge base.

The knowledge base contains:

- refund policy
- pricing policy
- privacy policy
- enterprise plan FAQ
- onboarding guide
- support escalation SOP
- old and new refund policy documents for change-impact analysis

The default business scenario is:

> Before deploying an AI support assistant, can we verify that the support
> knowledge base is answerable, grounded, and safe enough?

The Streamlit demo is designed around this scenario. It shows normal supported
answers, follow-up questions, unsupported questions, groundedness risk, readiness
reporting, and policy-change impact.

The initial Ask view uses the `hybrid` retriever, `extractive` answer mode, and the
question `標準月付用戶的退款期限是多久？`. The Readiness Audit runs the official
hybrid evaluation gate. Change Impact defaults to a generated old/new 50-page PDF
refund-policy pair; it also supports the smaller Markdown pair and custom uploads.

## What the Streamlit Demo Shows

The main demo is organized around three workflows.

### 1. Ask Mode

Ask Mode demonstrates the core RAG behavior:

- retrieve relevant knowledge-base chunks
- answer with citations
- resolve an underspecified follow-up within the current session
- refuse unsupported questions
- compare retrieval strategies
- compare answer-generation strategies
- expose groundedness risk when a provider returns unsupported claims

Recommended demo questions, in order:

```text
標準月付用戶的退款期限是多久？
那年度用戶呢？
客戶如果因為醫療因素，90 天後還可以退款嗎？
退款升級到人工審查的情境有哪些？
```

### 2. Audit Mode

Audit Mode turns the evaluation harness into a product-facing readiness report. It
checks answerable coverage, correct refusal behavior, citation coverage, retrieval
quality, groundedness risk, and knowledge gaps, then produces a scoped launch
recommendation. The result estimates readiness for this fixture and evaluation set;
it is not a general certification of production readiness.

### 3. Change Impact Mode

Change Impact Mode compares old and new policy documents. Instead of producing only
a plain text diff, it reports changed policy rules, risk levels, impacted evaluated
questions, required knowledge-base updates, and answers that may need human review.

The built-in refund-policy scenario includes this high-risk change:

```text
old_refund_policy.pdf vs new_refund_policy.pdf
refund window: 14 days -> 7 days
```

## Feature Matrix

| Area | Feature | Purpose |
|---|---|---|
| Retrieval | lexical retriever | Handles exact keyword and policy-term matching |
| Retrieval | dense retriever | Handles semantic matching when wording differs |
| Retrieval | hybrid retriever | Combines lexical and dense retrieval for the default Streamlit path |
| Answering | extractive mode | Produces conservative answers closely tied to retrieved evidence |
| Answering | generative mode | Produces more natural responses behind citation and groundedness validation |
| Provider | minimax | Optional live LLM provider for normal answer generation |
| Provider | fake_hallucination | Deliberately unsafe mock provider and test fixture for groundedness checks |
| Reliability | refusal behavior | Refuses or escalates when the KB does not provide enough support |
| Reliability | citations | Shows which source chunks support the answer |
| Reliability | groundedness check | Surfaces unsupported or risky generated claims |
| Audit | readiness report | Summarizes results for a scoped AI-assistant deployment decision |
| Change Impact | policy comparison | Flags aligned policy changes that may invalidate existing answers |

## Ask Mode Design

Ask Mode exposes three configurable layers in the UI.

### Retriever

The retriever controls how evidence is found in the knowledge base.

- `lexical`: keyword-based, BM25-style retrieval. Useful when the user uses exact
  policy terms and numbers.
- `dense`: multilingual embedding-based semantic retrieval. Useful when the user
  asks with different wording or in another supported language.
- `hybrid`: combines normalized lexical and dense scores with equal weights. This
  is the default setting for the Streamlit demo and official readiness audit.

Exposing these options makes retrieval behavior inspectable instead of hiding the
RAG pipeline behind one answer. For backward compatibility, the answer CLI defaults
to `lexical`; the documented demo commands explicitly select `hybrid`.

### Answer Mode

The answer mode controls how the final response is produced from retrieved chunks.

- `extractive`: the conservative default. It returns top-ranked source text or a
  deterministic refusal and requires no API key.
- `generative`: asks the selected provider for a structured, cited proposal. The
  proposal becomes the final answer only if the local validator accepts its chunk
  citations, retrieval provenance, and groundedness.

The two modes expose the trade-off between a conservative reliability baseline and
a more natural answer style. All official deterministic tests, audits, and the
complete demo use the no-key extractive path unless a generation example explicitly
selects a provider.

### LLM Provider

The provider selector appears only in `generative` mode.

- `minimax`: an optional live external LLM provider used to demonstrate normal
  generation. It requires `MINIMAX_API_KEY`.
- `fake_hallucination`: a deliberately unsafe mock provider used to simulate an
  unsupported generated claim. It makes the validator-blocking path reproducible
  without credentials or a network call.

The system does not trust either provider merely because it returns fluent text.
Generated claims remain untrusted until they pass the same local validation gate.

Ask Mode also applies a deterministic positive-admission router before generation.
Greetings, thanks, and app-introduction questions receive a canned capability
message. Clearly unrelated general questions receive a scope refusal. Low-information
input is rejected before retrieval; meaningful ambiguous queries use retrieval only
as a conservative relevance probe. These canned responses do not call a provider or
enter readiness metrics.

## Why `minimax` and `fake_hallucination`?

### `minimax`

`minimax` demonstrates the normal generation path with a real external LLM provider.
It receives only the question, generation contract, and retrieved chunks. Even when
it returns a fluent response, the answer must still cite retrieved `chunk_id` values
and pass deterministic provenance and groundedness checks.

MiniMax is optional and is not part of the official no-key validation baseline.
Live sample output is kept separate in
[generative_sample_runs.md](generative_sample_runs.md) because it is
non-deterministic.

### `fake_hallucination`

`fake_hallucination` is a reliability test fixture, not a real provider. It
deliberately proposes an unsupported 90-day medical refund exception. The validator
blocks that proposal, preserves the safe extractive refusal as the final answer,
sets low confidence, and requires human review. With `--json`, the rejected proposal
remains visible as `blocked_generated_answer` for auditability.

In short:

> `minimax` demonstrates normal generation.
>
> `fake_hallucination` demonstrates why the validation layer exists.

## Audit Mode

Audit Mode evaluates whether this knowledge base and its representative evaluation
set support a limited deployment decision. It reports:

- answerability and retrieval results
- correct refusal behavior
- citation and groundedness coverage
- per-case failures and concrete knowledge gaps
- the Ask Mode gate status
- a readiness recommendation

The healthy fixture must produce Ask Mode gate `PASS` and recommendation
`Internal Pilot Ready`. This label is a fixture-specific gate result, not a claim of
production readiness. A deterministic degraded fixture omits the refund policy and
selected Enterprise knowledge; it must produce gate `FAIL`, recommendation
`Not Ready`, and concrete missing topics. The degraded evaluation exits with status
1 by design so automation cannot mistake an unsafe KB for a successful audit.

Audit artifacts are `metrics.json` and `readiness_report.md` in the selected report
directory. Change-impact-only evaluation cases remain isolated from the active Ask
Mode gate.

## Change Impact Mode

Change Impact compares old and new Markdown, plain-text, or text-based PDF policy
documents without calling an LLM. It normalizes sections, aligns corresponding
policy rules, classifies risk, maps changes to evaluation cases and KB sections, and
produces JSON and Markdown reports for human review.

Markdown uses H1/H2 structure. Headerless Markdown and `.txt` files are split into
bounded paragraph chunks. PDF loading uses PyMuPDF layout metadata to remove repeated
headers and footers, identify headings, preserve 1-based page metadata, and create
page-bounded section chunks. The complete PDF is never loaded as one prompt or
context.

The built-in Markdown and generated 50-page PDF refund-policy fixtures produce the
same frozen baseline: 6 changes, 4 high-risk changes, 13 impacted evaluation cases,
and 9 required KB updates. The UI also accepts paired `.pdf`, `.md`, or `.txt`
uploads. Uploaded source files are compared from an operating-system temporary
directory and deleted after the run; generated report artifacts remain available.

This is deterministic policy-rule impact analysis. It is not a semantic or legal
diff, a full-corpus conflict scan, or an automatic policy-update mechanism.

## What This System Can Do

This system can:

- answer supported customer-support questions from the included knowledge base
- show chunk-level citations for retrieved evidence
- refuse or escalate when the KB does not contain enough support
- resolve simple follow-up questions within the current process
- compare lexical, dense, and hybrid retrieval behavior
- compare extractive and generative answer modes
- call an optional live LLM provider through `minimax`
- simulate unsupported generation through `fake_hallucination`
- block and surface groundedness risk for unsupported generated claims
- run a readiness audit over the included evaluation set
- report answerability, citation coverage, refusal behavior, and knowledge gaps
- compare old and new supported policy-document formats
- identify aligned policy changes that may affect evaluated questions and KB entries

## What This System Cannot Do

This system does not:

- guarantee that every possible contradiction in the entire corpus is detected
- replace human legal, policy, or support review
- automatically update the knowledge base after detecting a policy change
- implement user-level document permissions or row-level access control
- provide production-grade monitoring, alerting, security, or deployment hardening
- prove that an arbitrary external knowledge base is ready without a representative
  evaluation set
- make generative answers trustworthy without citation and groundedness checks

The intended scope is a focused demo of RAG reliability, readiness evaluation, and
policy-change impact analysis.

## How to Run

### Installation and Quickstart

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

Run the complete deterministic reviewer sequence with no API key:

```bash
./scripts/demo.sh
```

See [DEMO.md](DEMO.md) for the eight review steps and expected observations.

### Running the Streamlit App

```bash
python -m pip install -r requirements-ui.txt
streamlit run app.py
```

The three tabs expose Ask, Readiness Audit, and Change Impact through the existing
Python APIs. The generated 50-page PDF pair is created on demand when its built-in
demo is first run.

### Running Ask Mode from the CLI

Supported question using the default extractive answer path:

```bash
python -m src.answer \
  "標準月付用戶的退款期限是多久？" \
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

Optional live MiniMax example:

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
bounded request and retry settings. `.env.example` is a reference; the project does
not automatically load `.env` files.

Process-local follow-up example:

```bash
python -m src.session \
  "What is the standard refund window?" \
  "What about enterprise customers?" \
  --retriever hybrid
```

### Running Evaluation / Audit

Audit the complete KB and write isolated report artifacts:

```bash
python -m eval.run_eval \
  --retriever hybrid \
  --write-report \
  --report-dir data/reports/healthy
```

Build and audit the deterministic incomplete KB with the same gate:

```bash
python -m src.degraded
python -m eval.run_eval \
  --retriever hybrid \
  --index data/degraded/index/chunks.jsonl \
  --write-report \
  --report-dir data/reports/degraded
```

The degraded audit's status 1 is the expected machine-readable `Not Ready` result,
not an infrastructure failure.

### Running Change Impact

Compare the Markdown policy fixtures:

```bash
python -m src.compare \
  --old compare_docs/old_refund_policy.md \
  --new compare_docs/new_refund_policy.md
```

Generate and compare deterministic 50-page PDF fixtures:

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

Reports are written to `data/reports/change_impact.json` and
`data/reports/change_impact_report.md` unless another output directory is selected.

Generate the separate 50-page old/new support-contract pair used to exercise custom
uploads:

```bash
python -m scripts.build_custom_pdf_fixtures
```

This creates gitignored `custom_old_support_contract.pdf` and
`custom_new_support_contract.pdf` files under `compare_docs/`. The fixture changes
the support SLA, refund exception, Enterprise review requirement, data-deletion
window, and escalation rule. In Streamlit, choose **Change Impact**,
**Upload custom documents**, and upload both files. The expected result is 5 changed
sections, 5 high-risk changes, required KB updates, and human review required.

## Tests and Validation

Run the stable test suite:

```bash
python -m pytest
```

The complete validation sequence is:

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

Frozen reviewer baseline:

- Current validation: 68 deterministic pytest tests pass; 5 optional MiniMax live
  tests are key-gated and skip when `MINIMAX_API_KEY` is not set (73 total; all 73
  passed in the latest key-configured validation run).
- Ask Mode gate: `PASS`.
- Healthy audit: `Internal Pilot Ready`.
- Degraded audit: `Not Ready`.
- Markdown Change Impact: 6 changes, 4 high risk, 13 impacted evaluation cases,
  9 KB updates.
- 50-page PDF Change Impact: 6 changes, 4 high risk, 13 impacted evaluation cases,
  9 KB updates.
- `demo.sh` passes without a real API key.

## Project Structure

```text
app.py                         Streamlit UI for the three workflows
corpus/                        Six synthetic support-KB Markdown documents
compare_docs/                  Old/new policy fixtures for Change Impact
eval/eval_set.jsonl            Bilingual evaluation cases
eval/run_eval.py               Readiness evaluation runner
src/ingest.py                  Structure-aware corpus ingestion
src/retrieve.py                Lexical, dense, and hybrid retrieval
src/answer.py                  Routing, answering, refusal, and validation flow
src/generation.py              Live and deterministic generation providers
src/audit.py                   Readiness metrics and reports
src/compare.py                 Change Impact analysis and reports
src/document_loader.py         Markdown, text, and PDF section loading
src/session.py                 Process-local follow-up resolution
scripts/                       Demo and deterministic fixture builders
tests/                         Reliability and regression tests
DEMO.md                        Step-by-step reviewer sequence
DESIGN.md                      Detailed implementation decisions and boundaries
```

## Architecture and Evidence Contracts

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

`AnswerResult` preserves the question, response type, answer or refusal, retrieved
chunks, citations, confidence, human-review state, groundedness result, warnings,
latency, answer mode, validator decision, and optional generation trace. `--json`
exposes the complete schema.

Session memory is process-local. It rewrites an underspecified follow-up into a
standalone question before retrieval, then runs the same retrieval, refusal, and
groundedness path. It neither persists history nor bypasses evidence validation.

For implementation details and explicit design boundaries, see
[DESIGN.md](DESIGN.md).

## Design Trade-offs

- The default extractive path is deterministic and easy to audit, but its answers
  are source chunks rather than synthesized support responses.
- Lexical retrieval is lightweight and exact-term friendly. Dense retrieval handles
  semantic and bilingual variation. Hybrid uses simple min-max score normalization
  and equal-weight fusion; it is not RRF and has no reranker.
- Positive KB admission prevents obvious chitchat and unrelated questions from
  entering RAG, but its rules and relevance thresholds are calibrated to this small
  fixture.
- Generative output is treated as an untrusted proposal. Deterministic validation
  improves traceability but is narrower than full semantic entailment or policy
  correctness review.
- Section-aligned Change Impact scales beyond a single prompt and remains
  deterministic, but it depends on extractable text and usable document structure.

## Known Limitations

- The corpus is six synthetic Markdown documents and the evaluation set is small and
  curated; passing results are not production-traffic evidence.
- The groundedness validator checks citation provenance, claim coverage, and
  numeric/date/time support, but is not a complete semantic-entailment judge.
- Retrieval thresholds and hybrid fusion weights are calibrated only for this local
  fixture.
- Ask citations are chunk-level. Markdown has no page numbers; PDF Change Impact
  preserves section pages but does not provide sentence-level citations.
- Session memory is not durable or multi-user.
- PDF comparison does not OCR scanned PDFs or reliably interpret complex tables and
  ambiguous layouts.
- Query-relevant conflict checks are conservative hooks, not exhaustive corpus-wide
  contradiction detection.
- There is no production authentication, authorization, provider observability,
  rate limiting, cost control, monitoring, or deployment hardening.
