# PROPHET — Current State

## Project Summary

PROPHET is currently a local-first intelligence dashboard under active development.

The project is centered around four linked but distinct workflows:
1. source discovery
2. article scraping and local persistence
3. local vector indexing
4. local question-answering through Ollama

The system is no longer a simple URL-in / summary-out prototype. It has evolved into a date-driven, source-aware local corpus and RAG pipeline.

---

## Current Product Direction

The current direction is to make PROPHET feel like an operator console for ingesting and querying news data.

The main emphasis right now is not forecasting or probabilistic reasoning yet. The emphasis is on getting the ingestion, persistence, indexing, and retrieval foundation correct.

The immediate product goal is:

- choose a source
- choose a date
- query how many articles exist for that source/date
- scrape and persist those articles locally
- index the processed corpus incrementally
- ask questions against the locally indexed corpus using Ollama

---

## Current Architecture

### 1. Discovery
Discovery is intended to be lightweight.

It should answer:
- how many candidate articles exist for a source/date
- what basic metadata is available

It should not fully scrape article bodies.

### 2. Scrape
Scraping is the full acquisition workflow.

It should:
- fetch article pages
- extract cleaned article text
- deduplicate by content hash
- save raw and processed data locally

### 3. Index
Indexing is a separate workflow from scraping.

It should:
- read processed article text from disk
- chunk text
- generate embeddings through Ollama
- persist vectors and indexing metadata locally
- skip content already indexed under the current embedding model

### 4. Ask The Prophet
Ask The Prophet is a local RAG flow.

It should:
- verify whether relevant processed data has been indexed
- index missing data if necessary
- retrieve relevant chunks from the persistent index
- answer using the selected Ollama answer model

It should not rely on unsupported generic model knowledge.

---

## Current UI Direction

The UI is currently a local web dashboard, built around a dark, technical, operator-console aesthetic.

The UI has gone through multiple iterations and is still evolving.

The current intended high-level UI structure is:
- header / PROPHET branding
- UTC date and time
- local runtime / Ollama controls
- source-specific ingestion cards
- Ask The Prophet panel
- compact index/runtime health information

Several older sections have become obsolete and are being removed or replaced.

---

## Source Ingestion Direction

The ingestion model is now source-card based.

Generic homepage URL input is no longer the desired workflow.

The current source-card direction is:
- one card per source
- source-specific logic
- date-driven workflows
- lightweight query action
- full scrape action
- source-specific status and progress

The immediate supported/targeted sources are:
- AP News
- BBC
- Reuters

However, source support is not yet equally mature.

---

## AP News Status

AP News is one of the most developed sources in the current workflow.

The project has moved away from relying only on the AP homepage.

The intended AP workflow is date-driven archive/sitemap-style discovery.

The desired behavior is:
- user selects a date
- query action returns candidate article count for that date
- scrape action processes articles for that date
- saved files land in the date directory corresponding to the article/query date

There has been prior filtering logic for AP-related wire/photo artifacts. That filtering should now remain minimal and only remove obvious junk such as AP Photo / file photo style boilerplate.

---

## Reuters Status

Reuters support exists conceptually, but has encountered request-blocking issues when approached like a naive homepage scraper.

The project direction for Reuters is now archive/date-driven discovery rather than homepage-driven discovery.

The desired Reuters behavior is:
- user selects a date
- query action uses archive/discovery logic for that day
- scrape action processes Reuters articles from that day

Reuters still needs a robust final archive/discovery implementation.

---

## BBC Status

BBC has been incorporated as a target source, but language filtering has become important.

BBC discovery can surface non-English articles if filtering is too loose.

The current intended BBC behavior is:
- restrict to English BBC news surfaces
- reject non-English language-service pages
- verify page language before saving/indexing where practical

BBC source-card behavior should align with the same date-driven/source-specific operator workflow as AP and Reuters.

---

## Local Corpus Storage

The project uses local flat-file storage inside a repo-local, gitignored `data/` directory.

The intended structure is:

    data/
      raw/
      processed/
      index/
      cache/
      exports/

The corpus is intended to live primarily under `data/raw/` and `data/processed/`.

Processed article text is the intended source of truth for indexing.

The system should save scraped content under the correct publication/query date directory, not simply today's date.

That date-handling behavior still needed correction in recent work and should be treated as important.

---

## Deduplication Status

Deduplication is intended to be content-based.

The system should:
- clean article text
- hash the cleaned content
- compare against local saved/indexed data
- save only if the content is new

This is preferable to URL-only deduplication.

A local manifest/ledger is acceptable for tracking what has already been saved or indexed.

---

## Indexing Status

Indexing is already conceptually separated from scraping, but this separation has needed reinforcement.

The intended indexing behavior is:
- only index processed article text
- do not index raw HTML
- do not index homepage boilerplate
- do not index intermediate scrape artifacts

Indexing should be incremental.

The system should only embed and store newly saved processed content that has not already been indexed for the currently selected embedding model.

Progress visibility during indexing has been a known concern. Better progress reporting, elapsed time, and step-level feedback are desired.

---

## Persistent Index Storage

Persistent index storage is intended to be embedding-model specific.

The desired storage pattern is:

    data/index/<embedding-model>/<source>/

Examples:

    data/index/nomic-embed-text/ap-news/
    data/index/embeddinggemma/bbc/

The goal is to prevent mixing incompatible embedding spaces.

A `vector_store.sqlite` and a `vector_manifest.json` have been observed during indexing. That is acceptable for now, as long as:
- the vector store remains the source of truth for stored vector content
- the manifest remains a lightweight ingestion ledger
- the two are not treated as competing databases

---

## Ollama Integration Status

Ollama is the local model runtime for both:
- embeddings
- answer generation

The project now expects explicit runtime awareness.

Desired runtime behavior:
- if Ollama is offline, indexing should be disabled
- if Ollama is online, model dropdowns should be enabled
- runtime/model state should be visible in the UI

There should be separate user-selectable dropdowns for:
- Embedding Model
- Answer Model

Hardcoded model assumptions are no longer desired.

The current known local models from development include:
- `nomic-embed-text`
- `llama3.1`

Additional candidate models discussed include:
- `embeddinggemma`
- `qwen3:8b`
- `qwen3:4b`

The project should treat embedding-model changes as requiring separate indexes, while answer-model changes should not require re-indexing.

---

## Ask The Prophet Status

Ask The Prophet is intended to be grounded in the local indexed corpus.

The current desired flow is:
- user asks a question
- app verifies whether processed data has already been indexed for the selected embedding model
- app indexes missing content if needed
- app retrieves relevant chunks
- app calls the selected local answer model
- app answers only from retrieved support

The system should not fabricate answers if corpus support is weak.

Citations or supporting article URLs are desirable where practical.

---

## Current Pain Points

The main current pain points are:

### 1. Date handling
Saved data has at times landed in today's date folder instead of the intended publication/query-date directory.

### 2. Source-specific archive discovery
Homepage scraping is no longer sufficient, especially for Reuters and archive-style workflows.

### 3. Progress visibility
Scraping and indexing need clearer progress bars, elapsed time, and step-level status.

### 4. Ollama/runtime correctness
Runtime status, model discovery, endpoint compatibility, and index gating need to remain robust and transparent.

### 5. UI drift
Several older sections have become obsolete and need to be removed cleanly instead of lingering as dead UI.

---

## Recent UI Direction

Recent requested UI changes include:
- removal of stale summary/output sections
- improved local runtime/operator area
- model selection dropdowns
- source-card based ingestion
- per-source Data Scrape and Index Data controls
- disabled indexing when Ollama is offline
- cancellable scrape workflow
- cleaner progress reporting

The interface should continue becoming tighter and more operator-focused, not more generic.

---

## What Is Working Conceptually

The key project ideas are now much clearer than they were at the beginning.

The correct conceptual model is now:

- scrape saves flat files
- index reads those flat files
- Ollama provides embeddings and answers
- the index persists across restarts
- different embedding models get different index directories
- sources should be handled explicitly and realistically
- archive/date-driven discovery is the right long-term ingestion path

This is a much stronger foundation than the original homepage-summary prototype.

---

## What Still Needs Hardening

The system still needs hardening in these areas:
- source-specific archive discovery paths
- correct date-based save paths
- clearer scrape/index status in the UI
- robust model/runtime selection behavior
- stronger Ask The Prophet grounding and citations
- more reliable source-language/source-quality filtering
- cleaner operator-console organization

---

## Immediate Next Priorities

The next practical priorities are:

1. finalize date-based save-path correctness
2. stabilize AP / BBC / Reuters source-card workflows
3. make scrape cancellation and progress reporting trustworthy
4. ensure indexing only uses processed text
5. ensure model-specific index directories are respected
6. improve runtime diagnostics and model selection
7. keep Ask The Prophet grounded and index-aware

---

## Guidance for Future Development

Do not jump into large forecasting/probabilistic features yet.

Do not overbuild abstractions for sources that are not working yet.

Do not mix scrape and index logic.

Do not mix embedding model spaces.

Do not hide local state or runtime state.

Do prioritize:
- persistence correctness
- source realism
- index correctness
- runtime clarity
- operator workflow coherence