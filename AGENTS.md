# PROPHET — Agent Instructions

## Project Identity

PROPHET is a local-first intelligence dashboard for scraping, storing, indexing, and querying news and event data.

The system is intended to support:
- date-based news ingestion
- persistent local corpus storage
- persistent local vector indexing
- local LLM question-answering through Ollama
- future probabilistic / forecasting workflows

The project should feel like an operator console, not a toy demo.

---

## Core Architecture

PROPHET currently has four distinct workflow layers:

### 1. Discovery
Lightweight source query.
- Count available articles for a given source/date
- Return metadata only, not full article content

### 2. Scrape
Full article acquisition and persistence.
- Fetch article pages
- Extract cleaned article text
- Deduplicate via content hash
- Write flat files to local corpus storage

### 3. Index
Persistent local vector indexing.
- Read saved processed article text from disk
- Chunk text
- Generate embeddings through local Ollama
- Persist vectors and indexing metadata locally
- Do not re-index already indexed content

### 4. Ask The Prophet
Local RAG-style answering.
- Verify the relevant local corpus has been indexed
- Index missing processed data if needed
- Retrieve relevant chunks from the persistent local index
- Answer questions using a local Ollama answer model
- Do not answer from unsupported generic world knowledge

These workflows must remain distinct in code.

Do not collapse scrape and index into one vague pipeline.

---

## Frontend Expectations

The current frontend is a web-based local dashboard.

Preferred stack:
- Python
- Streamlit, unless there is a strong reason otherwise

The UI should feel:
- dark
- sleek
- technical
- operator-console-like
- compact but readable

Do not make the UI look like a generic form app.

### Current UI Priorities

The main dashboard should prioritize:
- header / branding
- UTC time/date
- local runtime / Ollama controls
- source-specific ingestion controls
- Ask The Prophet interaction
- index health / runtime diagnostics

Avoid clutter and dead sections.

If an old section has become obsolete, remove it cleanly instead of leaving placeholders.

---

## Source Ingestion Model

Source ingestion should be source-card based, not generic homepage-URL driven.

The generic "Source Homepage URL" and "Max Articles to Scrape" style controls are obsolete.

### Source Card Pattern

Each source card should support:
- source identity
- date input
- lightweight query action
- full scrape action
- source-specific index action if needed
- source-specific status / progress feedback

### Current Active Source Direction

The source-card system should be designed to support multiple sources, but implementation should remain incremental.

Current priority sources include:
- AP News
- BBC
- Al Jazeera

Do not assume all sources work the same way.

Source-specific logic is acceptable and expected.

---

## Discovery Behavior

"Query Site Article Count" should be lightweight.

It should:
- determine how many candidate articles exist for a source/date
- return count and optional lightweight metadata preview
- avoid fully scraping article bodies

This is meant to be fast and cheap.

Do not silently turn it into a full scrape.

---

## Scrape Behavior

"Data Scrape" performs the full scrape workflow for the selected source/date.

It should:
- fetch article pages
- extract cleaned article text
- deduplicate using content hash
- save flat files to local storage
- provide visible status / progress

The scrape button should become **Stop Scraping** while active.

Stopping should be graceful:
- do not kill the app
- do not discard already completed work
- stop the loop safely

---

## Date Handling

Date handling is critical.

When a user selects a date in the UI, scraped articles should be saved under the directory corresponding to the article's actual publication date whenever available.

At minimum, date-driven archive scrapes must save into the directory matching the selected/query date, not today's date.

Preserve both:
- publication date
- scrape timestamp

Do not collapse those concepts.

---

## Local Data Storage

The corpus is local-first and file-based.

Use a repo-local, gitignored `data/` structure.

Preferred structure:

    data/
      raw/
      processed/
      index/
      cache/
      exports/

Flat article corpus should live under `raw/` and `processed/`.

Processed article text is the main indexing input.

Do not index raw HTML unless there is a compelling explicit reason.

### Save Path Expectations

Use source-partitioned and date-partitioned paths, for example:

    data/processed/ap-news/2026-03-26/
    data/processed/bbc/2026-03-26/

Use filesystem-safe normalized source names.

---

## Deduplication

Deduplication should be based on content hash of cleaned article text.

Do not rely only on URL.

Expected logic:
- scrape article
- clean/extract article text
- hash cleaned content
- compare against local manifest/index
- save only if new

If the content already exists locally:
- do not save a duplicate copy
- log/report it as already present

---

## Indexing Architecture

Indexing must be separate from scraping.

The index should operate only on locally saved processed text files.

Do not index:
- raw HTML
- homepage boilerplate
- intermediate scrape artifacts

### Model-Specific Index Storage

Persistent index storage must be partitioned by embedding model and then by source.

Preferred structure:

    data/index/<embedding-model>/<source>/

Examples:

    data/index/nomic-embed-text/ap-news/
    data/index/embeddinggemma/bbc/

Different embedding models must never share the same persistent vector store path.

Do not mix vector spaces.

### Index Bookkeeping

A lightweight manifest is acceptable for ingestion bookkeeping, such as:
- what processed files exist
- what content hashes have been indexed
- which embedding model indexed them
- which source they belong to
- indexed timestamps

The manifest should not become a second competing database.

Let the vector store remain the source of truth for stored vector content.
Let the manifest remain the source of truth for ingestion bookkeeping.

---

## Ollama Integration

Ollama is the local model runtime.

It is used for:
- embeddings during indexing and retrieval
- answer generation for Ask The Prophet

Ollama should be treated as a local service dependency, not as corpus storage.

### Model Selection

Do not hardcode one model.

The UI should support separate selection for:
- **Embedding Model**
- **Answer Model**

Prefer live model discovery from the local Ollama runtime.

If that fails, fall back cleanly and visibly.

### Runtime Behavior

When Ollama is offline:
- indexing controls should be disabled
- the UI should say why
- Ask The Prophet should fail clearly if answering requires the runtime

When Ollama is online:
- model dropdowns should be enabled
- runtime diagnostics should reflect the active models

### API Behavior

Use the correct Ollama API shapes.

Be careful about endpoint/version mismatches.

Do not assume older and newer embedding endpoints are interchangeable without explicit handling.

If compatibility handling is needed, centralize it in one helper.

---

## Ask The Prophet

Ask The Prophet is a local RAG workflow.

It should:
1. verify whether the relevant processed corpus has already been indexed for the currently selected embedding model
2. index missing processed data if needed
3. embed the user question
4. retrieve relevant chunks from the persistent local index
5. generate an answer with the selected local answer model

It must not:
- answer from unsupported generic world knowledge
- silently fabricate support
- mix indexes from different embedding models

If corpus support is weak, say so clearly.

Citations or supporting URLs are desirable when practical.

---

## Progress and Status Expectations

Long-running workflows should not look frozen.

During scraping and indexing, expose useful progress information such as:
- total items discovered
- total eligible
- total completed
- total skipped
- total remaining
- current item
- current phase
- elapsed time

If a real total exists, use it.

Do not fake progress with decorative timers.

---

## Word Filtering and Text Cleaning

Do not over-filter analysis text.

Earlier broad stopword/source-word exclusions should not return unless specifically justified.

Keep only a small obvious junk/boilerplate filter for things like:
- AP Photo
- file photo
- similar wire-service artifacts

Do not aggressively strip legitimate news terms.

---

## Engineering Principles

Keep code modular.

Keep these concerns separate:
- UI rendering
- source discovery
- full scraping
- persistence
- indexing
- Ollama runtime integration
- retrieval / answering

Prefer incremental, inspectable changes over cleverness.

Prefer deterministic behavior over hidden magic.

Avoid:
- broad rewrites
- generic abstractions too early
- overbuilding for future sources before current sources work
- hiding failures behind vague UI messages

---

## What Codex Should Optimize For

When making changes:
- preserve current working behavior unless the change explicitly replaces it
- avoid introducing dead UI sections
- keep local file paths and runtime state visible enough for debugging
- favor source-aware logic over fake generality
- make progress/status legible
- make storage/index layout easy to inspect manually

Do not optimize for "minimal code diff" if it leaves the product incoherent.

Do optimize for:
- clarity
- local debuggability
- persistence correctness
- model/index separation
- clean operator workflow

---

## Immediate Project Priorities

The current priority order is roughly:

1. reliable date-based source discovery
2. reliable flat-file corpus saving under correct dates
3. reliable source-specific scraping
4. reliable incremental indexing from processed text only
5. clean Ollama runtime integration with selectable embedding/answer models
6. trustworthy Ask The Prophet local RAG behavior
7. gradual expansion to more sources and historical archives

Do not jump ahead into probabilistic forecasting logic until the ingestion and retrieval foundations are stable.