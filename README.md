# RAG_app

A Retrieval-Augmented Generation app that ingests mixed document types — PDF, HTML, DOCX, and images — and answers questions about them with citations back to the source.

Status: **planning / not yet implemented.** This README is the build plan. Tasks are grouped into phases meant to be completed one at a time.

Repo: https://github.com/lakshmiGanesh009/RAG_app

---

## Goals

- Point the app at a folder of mixed documents and ask questions in plain language.
- Every answer cites which file (and where in it) the information came from.
- Runs locally with no paid API required; hosted models optional.
- Add one format or capability at a time without reworking the pipeline.

## Features

**Document parsing**
- PDF, including scanned PDFs via OCR
- HTML pages and local HTML files
- DOCX (and other Office formats, effectively free once the parser is in)
- Images (PNG, JPEG, TIFF, BMP, WEBP) via OCR
- One unified internal representation, so downstream code does not branch per format

**Retrieval**
- Layout-aware chunking that respects headings, tables, and reading order
- Local embeddings, no API key needed
- Persistent vector store that survives restarts
- Re-ingesting an unchanged file does not create duplicates

**Answering**
- Answers grounded strictly in retrieved chunks
- Inline citations with filename and page/section
- Says "not found in the documents" instead of guessing
- Pluggable model: local (Ollama) or hosted (OpenAI-compatible)

**Interfaces**
- CLI for ingest and ask
- HTTP API for integration
- Minimal web UI for demos

## Supported formats

| Type | Formats | Notes |
| --- | --- | --- |
| PDF | `.pdf` | Text layer read directly; OCR fallback for scans |
| Word | `.docx` | Legacy `.doc` needs LibreOffice installed |
| Web | `.html`, `.xhtml`, URLs | |
| Images | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp` | OCR required |
| Bonus | `.pptx`, `.xlsx`, `.md`, `.csv`, `.epub` | Come along for free with the chosen parser |

## How it works

```
documents ──> parse ──> chunk ──> embed ──> vector store
                                                 │
question ──> embed ──> similarity search ────────┘
                                 │
                                 v
                        retrieved chunks + question ──> LLM ──> answer + citations
```

Two separate flows share one embedding model: an **ingest** flow that runs when documents change, and a **query** flow that runs per question. Keeping them separate is what makes ingestion cheap to re-run and queries fast.

## Tools we are using

Every version below was checked to install on this machine's Python 3.14.

| Concern | Tool | Version | Why this one |
| --- | --- | --- | --- |
| Parsing | [Docling](https://github.com/docling-project/docling) | 2.126.0 | One library covers PDF, DOCX, HTML, and images with OCR, so there is a single parser to learn instead of four. MIT, runs locally. |
| Chunking | Docling hybrid chunker | bundled | Hierarchy- and token-aware, not blind character splitting. Carries page and heading metadata through, which is what makes citations possible. |
| Embeddings (dense) | [FastEmbed](https://github.com/qdrant/fastembed) + `bge-small-en-v1.5` | 0.8.0 | ONNX-based, so no PyTorch in the dependency tree. 384-dim, fast on CPU. |
| Keyword index (sparse) | [bm25s](https://github.com/xhluca/bm25s) | 0.3.11 | BM25 over sparse matrices, considerably faster than the pure-Python alternative. Powers the lexical half of hybrid search. |
| Fusion | Reciprocal Rank Fusion | hand-rolled | ~20 lines. Rank-based, so dense and BM25 scores never need normalising onto a common scale. |
| Reranker | [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) | 0.2.10 | Small ONNX cross-encoder, no torch. Optional Phase 6 quality pass. |
| Vector store | Chroma | 1.5.9 | Persistent, local, minimal setup. |
| LLM client | `openai` SDK | latest | One code path serves Ollama, LM Studio, vLLM, OpenAI, and OpenRouter. |
| Config | pydantic-settings | latest | Typed settings with validation at startup. |
| CLI | Typer | 0.27.2 | |
| API | FastAPI + Uvicorn | 0.141.1 | |
| Tests | pytest | latest | |

**No RAG framework.** We call these components directly rather than going through LlamaIndex or LangChain. Reasoning in the next section.

## Tools we are deliberately not using

None of these are bad choices. They are choices that do not earn their weight *for this project*, and the point of listing them is so the decision is revisitable rather than accidental.

| Tool | What it is good at | Why not here | Adopt it when |
| --- | --- | --- | --- |
| **LlamaIndex** (`llama-index-core` 0.14.24) | Purpose-built retrieval/indexing framework with the widest data-connector ecosystem; MIT core. Widely treated as a strong default for production RAG. | Our pipeline is one embedder, one store, one collection. LlamaIndex would wrap components we are calling directly anyway, and abstraction makes step-by-step learning harder — you debug the framework instead of the retrieval. | You need many data connectors (Notion, Slack, Confluence, SQL), several index types side by side, or composable query engines. |
| **LangChain** (1.4.0) | Multi-step agent orchestration; its `create_agent` API on LangGraph is the mature option for agentic flows. Huge integration catalogue. | We have no agent, no tool calling, and no multi-step reasoning loop. A linear retrieve-then-answer flow does not need an orchestration layer. | The app needs tool use, multi-step planning, or stateful conversational graphs. |
| **pypdf** (6.18.0) | Tiny pure-Python PDF text extraction. Excellent for page counts, metadata, splitting, and merging. | Gives you a text dump: no OCR, no table structure, no reading-order model. Two-column layouts and tables come out scrambled, which quietly poisons retrieval. | You need cheap PDF metadata or page manipulation — worth keeping as a utility. Also a reasonable fast path for PDFs you know have a clean text layer. |
| **unstructured** | The broadest raw format coverage for mixed-format pipelines. | Docling already covers all four formats we need, with stronger PDF layout handling. | Your format list grows past what Docling handles. |
| **sentence-transformers** (6.0.1) | The default embedding library, largest model selection. | Pulls in PyTorch, and its published classifiers stop at Python 3.13 — it does not yet advertise 3.14 support. | You need a specific model FastEmbed has not packaged, on Python ≤3.13. |
| **Elasticsearch / OpenSearch** | Production-grade BM25 and hybrid search at scale. | A whole service to run for a local app. `bm25s` in-process gets us the same retrieval concept. | Corpus outgrows one machine, or you need multi-tenant search infrastructure. |

**The honest version of the framework question:** LangChain and LlamaIndex have converged and now overlap heavily on both retrieval and agents, so the familiar "LangChain for agents, LlamaIndex for data" split describes the landscape less well than it used to ([premai.io](https://www.premai.io/blog/langchain-vs-llamaindex-2026-complete-production-rag-comparison/), [uvik.net](https://uvik.net/blog/llamaindex-vs-langchain/)). Guidance from several 2026 comparisons also notes that with a straightforward workflow and solid Python, you may not need either ([addepto.com](https://addepto.com/blog/langchain-vs-llamaindex-main-differences/)). That is the situation here. Build it directly first; you will understand each component well enough to judge whether a framework helps. Reach for LlamaIndex first if you do adopt one, since retrieval is this project's centre of gravity.

## Hybrid indexing

Worth doing, and planned as Phase 6. The short argument: the two retrieval methods fail in opposite directions.

- **Dense/vector search** is built to generalise meaning, so it handles paraphrase well — "car" matching "automobile". It is unreliable for rare exact tokens: an error code like `0x80070005`, a part number, or an unusual proper noun gets smoothed into semantic neighbourhood rather than matched.
- **BM25/lexical search** anchors precisely on those tokens and needs no training. It has no notion of meaning, so it misses paraphrase entirely.

Running both and merging covers each one's blind spot. Reported gains are real but moderate: one 2026 write-up measured a tuned hybrid at 0.7497 NDCG on the WANDS benchmark against 0.6983 for BM25 alone and 0.6953 for vector alone, roughly a 7.4% lift ([denser.ai](https://denser.ai/blog/hybrid-search-for-rag/)). That is an e-commerce benchmark rather than a document-QA one, so treat it as directional — but the direction is consistent across sources.

Planned design:

```
question
   ├──> dense retrieval  (FastEmbed + Chroma)  ──> ranked list A
   └──> sparse retrieval (bm25s)               ──> ranked list B
                                                      │
                            Reciprocal Rank Fusion ───┤
                                                      v
                                          merged candidates (~25)
                                                      │
                                    FlashRank rerank (optional)
                                                      v
                                              top 5 ──> LLM
```

Two implementation notes for when you get there:

- **RRF over weighted score fusion.** RRF combines rank positions, so it sidesteps the awkward problem that cosine similarity and BM25 scores live on completely different scales. Weighted fusion can edge it out once tuned, but it needs normalisation and per-corpus tuning to work at all.
- **Build the dense path first and measure it.** Add BM25 only once you have a question set that shows where dense retrieval is failing. Otherwise you are adding a second index and a fusion step with no evidence either is helping.

*Content from linked sources was rephrased for compliance with licensing restrictions.*

## Two things worth knowing before you start

- **Docling covers all four requested formats natively** — its documented input list includes PDF, DOCX, HTML, and image formats with OCR support ([supported formats](https://docling-project.github.io/docling/usage/supported_formats/)).
- **Chunking strategy affects accuracy more than you would expect.** A 2026 peer-reviewed comparison of PDF-to-Markdown pipelines found hierarchical splitting with image descriptions scored highest for downstream question answering, with table-dependent questions showing the largest gap between basic and hierarchical splitting ([arXiv:2604.04948](https://arxiv.org/abs/2604.04948)). Budget real time for Phase 3.

### Environment note

This machine has **Python 3.14.0**. Everything in the stack above was confirmed to resolve against it: Docling, FastEmbed, bm25s, FlashRank, Chroma, Typer, FastAPI, and ONNX Runtime. The rejected tools resolve fine too — `pypdf`, `llama-index-core`, and `langchain` all install on 3.14 — so nothing here was ruled out for compatibility reasons.

The one real exception is **`sentence-transformers`**, whose published classifiers stop at Python 3.13. That is a second, independent reason the plan uses FastEmbed. If you later adopt a library without 3.14 wheels, build the virtualenv against Python 3.12 or 3.13 instead.

A bare `.venv` already exists in the repo with an up-to-date pip and nothing else installed. Delete it if you would rather start fresh.

## Planned project structure

```
RAG_app/
├── rag_app/
│   ├── config.py        # settings from environment / .env
│   ├── parsing.py       # any file -> unified document
│   ├── chunking.py      # document -> chunks with metadata
│   ├── embedding.py     # text -> dense vectors (FastEmbed)
│   ├── store.py         # vector store read/write (Chroma)
│   ├── keyword_index.py # BM25 sparse index (bm25s)      <- Phase 6
│   ├── retrieval.py     # dense + sparse + RRF fusion    <- Phase 6
│   ├── rerank.py        # cross-encoder rerank, optional <- Phase 6
│   ├── llm.py           # chat model client
│   ├── pipeline.py      # ties ingest and query together
│   ├── cli.py           # command line entry point
│   └── api.py           # HTTP endpoints
├── tests/
├── data/
│   ├── samples/         # test documents, one per format
│   └── vector_store/    # generated, git-ignored
├── .env.example
├── requirements.txt
└── README.md
```

---

## Tasks

### Phase 0 — Project setup
- [ ] Create the virtualenv and pin dependencies in `requirements.txt`
- [ ] Add `.gitignore` covering `.venv/`, `data/vector_store/`, `.env`, `__pycache__/`
- [ ] Add `.env.example` documenting every setting
- [ ] Create the package skeleton and confirm it imports
- [ ] Make the first commit and push to `main`

### Phase 1 — Configuration
- [ ] Define a settings object with working defaults for everything
- [ ] Load from environment variables and `.env`
- [ ] Validate values on startup so bad config fails loudly and early
- [ ] Keep secrets out of the repo, read them from the environment only

### Phase 2 — Parsing (the core of this project)
- [ ] Collect sample documents: one PDF, one HTML, one DOCX, one image, plus one scanned PDF
- [ ] Parse a single PDF end to end and inspect the output
- [ ] Add DOCX
- [ ] Add HTML, from both local file and URL
- [ ] Add images with OCR
- [ ] Make OCR toggleable — it is slow, and most PDFs do not need it
- [ ] Handle a directory of mixed formats in one call
- [ ] Skip unsupported files with a clear warning instead of crashing the batch
- [ ] Preserve source metadata: filename, page number, section heading

### Phase 3 — Chunking
- [ ] Split parsed documents into chunks with hierarchy awareness
- [ ] Carry metadata through onto every chunk so citations work later
- [ ] Keep chunk size within the embedding model's token limit
- [ ] Verify tables survive chunking without being cut mid-row
- [ ] Inspect chunk boundaries by eye on a real document before moving on

### Phase 4 — Embeddings and vector store
- [ ] Embed a list of texts and confirm the vector dimensions
- [ ] Initialise a persistent store and confirm it reloads after restart
- [ ] Write chunks with their metadata
- [ ] Use content-hash IDs so re-ingesting a file replaces rather than duplicates
- [ ] Query by similarity and sanity-check that results are actually relevant
- [ ] Add `stats` and `reset` operations

### Phase 5 — Retrieval and answering
- [ ] Embed the question and retrieve the top-k chunks
- [ ] Apply a relevance floor to drop weak matches
- [ ] Build a prompt that numbers the sources and demands citations
- [ ] Wire up the chat model client
- [ ] Instruct the model to refuse when the context does not contain the answer
- [ ] Return the answer alongside the chunks it came from
- [ ] Support a no-LLM mode that returns raw chunks, useful for debugging retrieval

### Phase 6 — Hybrid indexing and reranking
Do this only after Phase 5 works and you have a question set showing where dense retrieval falls short.
- [ ] Assemble ~20 real questions with the source you expect each to hit; record dense-only accuracy as the baseline
- [ ] Build a BM25 index over the same chunks with `bm25s`
- [ ] Keep the two indexes in sync — one ingest pass must write both, or they drift silently
- [ ] Retrieve from both paths independently and eyeball where each one wins
- [ ] Merge with Reciprocal Rank Fusion; make the `k` constant configurable
- [ ] Re-measure against the baseline. Keep hybrid only if it actually wins
- [ ] Widen the candidate set (~25–50) and add FlashRank reranking on top
- [ ] Re-measure again; reranking costs latency, so confirm it pays for itself
- [ ] Add a config switch for `dense` / `hybrid` / `hybrid+rerank` so you can compare any time

### Phase 7 — Interfaces
- [ ] CLI: `ingest`, `ask`, `stats`, `reset`, `formats`
- [ ] Readable CLI output with progress on long ingests
- [ ] HTTP API: health, ingest, ask, stats
- [ ] **Add authentication before the API leaves localhost** — see Security below
- [ ] Minimal web UI: upload, ask, show answer with sources

### Phase 8 — Testing and quality
- [ ] Unit tests for chunking boundaries and metadata propagation
- [ ] One integration test per supported format using the sample files
- [ ] Test the failure paths: corrupt file, empty file, unsupported extension
- [ ] A small question/expected-source set to catch retrieval regressions
- [ ] Set up linting and formatting
- [ ] Add CI to run tests on push

### Phase 9 — Deployment
- [ ] Write a Dockerfile, pre-downloading the embedding model into the image
- [ ] Add `docker-compose.yml` with Ollama alongside, for a fully local stack
- [ ] Mount the vector store as a volume so it survives container restarts
- [ ] Persist or rebuild the BM25 index on startup — decide which, and document it
- [ ] Externalise all configuration; no secrets baked into the image
- [ ] Add a health check endpoint for the orchestrator
- [ ] Document resource requirements — OCR and parsing are memory-hungry
- [ ] Decide on a target: single VM, container service, or serverless with external vector DB
- [ ] Add structured logging and basic request metrics

## Security

Two items not to defer:

1. **The HTTP API has no authentication in the plan above.** Bind it to `127.0.0.1` during development. Add an API key or a reverse proxy with auth before exposing it anywhere, and treat that as blocking for Phase 9.
2. **Ingestion accepts arbitrary files.** Validate file types and cap upload size. Anything a user uploads is untrusted input, and text extracted from a document is untrusted content — do not let it steer the model's instructions.

Also worth noting: documents you ingest are sent to whichever model you configure. If they are sensitive, use a local model via Ollama rather than a hosted API.

## Ideas for later

- Chart and figure descriptions fed into the index
- Query rewriting / multi-query generation before retrieval
- SPLADE learned-sparse retrieval as an alternative to BM25
- Conversational follow-up questions with history
- Per-document access control for multi-user use
- Incremental re-ingest driven by file modification times

## License

MIT
