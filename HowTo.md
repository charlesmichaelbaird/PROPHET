# PROPHET How-To Guide (Current Repository State)

This guide documents how to build, configure, run, and use **the project exactly as it is currently implemented**.

## 1) Project Overview (What currently works)

PROPHET currently provides a **Streamlit web dashboard** that runs a **zero-cost, no-LLM news analysis pipeline**:

1. You enter a homepage URL (default: `https://apnews.com`).
2. The app fetches the homepage HTML.
3. It extracts likely article links from that homepage.
4. It fetches each article and extracts paragraph text.
5. It computes top-word frequency metrics (excluding built-in stopwords).
6. It displays:
   - number of links found,
   - number of articles successfully scraped,
   - top 10 words with counts and coverage,
   - preview list of scraped article links.

There is **no OpenAI/LLM usage** in the current implementation.

---

## 2) Repository Structure Summary

```text
PROPHET/
├── frontend/
│   └── app.py              # Streamlit UI entry point
├── mcp_server/
│   ├── server.py           # run_pipeline wrapper + TOOLS dict
│   ├── tools.py            # scraping/parsing/analysis implementation
│   └── storage.py          # local flat-file persistence helpers
├── data/                   # local scrape data (gitignored contents)
│   ├── raw/                # raw article HTML by source/date
│   ├── processed/          # cleaned text + metadata by source/date
│   ├── index/              # per-run summary records
│   ├── cache/              # reserved for future local caches
│   └── exports/            # reserved for future export files
├── main.py                 # PyCharm sample script; not the app runtime
└── requirements.txt        # Python dependencies
```

### Important entry points

- **Primary app entry point:** `frontend/app.py`
- **Analysis pipeline function:** `mcp_server/server.py::run_pipeline`
- **Core scraping + analysis helpers:** `mcp_server/tools.py`
- **Not an app entry point:** `main.py` (simple hello script)

---

## 3) Prerequisites

## Python version

The repo does not pin an exact Python version in config files, but code uses modern type hints and `from __future__ import annotations`.

Recommended:
- **Python 3.10+** (3.11 works well in most environments)

## Dependencies

Install from `requirements.txt`:
- `requests>=2.31.0`
- `streamlit>=1.30.0`

## Environment variables

Current code does **not require any environment variables**.

- No API key is required.
- No OpenAI configuration is referenced.
- No `.env` loader is present.

---

## 4) Setup Instructions (Fresh Clone)

## Step 1: Clone

```bash
git clone <YOUR_REPO_URL>
cd PROPHET
```

## Step 2: Create virtual environment

```bash
python3 -m venv .venv
```

## Step 3: Activate virtual environment

macOS/Linux:
```bash
source .venv/bin/activate
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

## Step 4: Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5) Run Instructions

## Run the web app

From repository root:

```bash
streamlit run frontend/app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## MCP-related components (current status)

The folder `mcp_server/` exists, but there is **no standalone MCP protocol server process** (no socket/http server start function, no CLI command, no long-running daemon scaffold).

What is currently runnable:
- `run_pipeline(...)` in `mcp_server/server.py` as an in-process Python function.
- `analyze_homepage(...)` in `mcp_server/tools.py` as an in-process Python function.

So currently there is **no separate MCP component to start first**. The Streamlit app imports and calls the pipeline directly.

---

## 6) Usage Instructions (Web UI)

After starting Streamlit:

## Step 1: Enter homepage source

In **Source Homepage URL**, enter a full URL such as:
- `https://apnews.com`
- another news homepage with article links under the same domain

URL must include `http://` or `https://`.

## Step 2: Set scrape limit

Choose **Max articles to scrape** (5 to 50).

## Step 3: Run analysis

Click **Run Zero-Cost Analysis**.

The app will:
- fetch homepage,
- collect likely article links,
- fetch up to max articles,
- extract paragraph text,
- compute top word metrics.

## Step 4: Review output

The right panel displays:
- **Links found** (candidate article links)
- **Articles scraped** (articles that yielded usable text/tokens)
- **Top 10 most common words** table with:
  - `word`
  - `total occurrences`
  - `article count`
  - `article coverage %`
- **Preview of scraped articles** (title + link list)

## Step 5: Reset

Click **Reset Results** to clear session output.

---

## 7) Current Implemented Features

- Streamlit dashboard UI with custom styling.
- Homepage URL input + article count input.
- URL validation for required scheme/host.
- Homepage scraping via `requests`.
- HTML anchor parsing using `html.parser`.
- Domain-constrained link extraction and basic article-link heuristics.
- Article text extraction from paragraph tags, skipping noisy tags.
- Word tokenization, stopword filtering, and top-10 frequency summary.
- Basic error handling with UI-safe response structure.

---

## 8) Current Limitations / Stubbed or Incomplete Areas

- **No LLM integration** (by design in current code).
- **No OpenAI integration** (not optional/configured; simply absent).
- **No standalone MCP server runtime** despite `mcp_server/` naming.
- Link extraction is heuristic-based and may miss or include irrelevant links depending on site structure.
- Article parsing relies on `<p>` tags and may fail on heavily scripted/dynamic websites.
- Failures while scraping individual articles are silently skipped.
- Scraped results are persisted locally to `data/` as flat files (gitignored contents).
- No automated test suite present in repository.

---

## 10) Local Data Persistence

Scraped data is now saved to a repo-local `data/` folder during each run.

- `data/raw/{source}/{YYYY-MM-DD}/`
  - Raw article HTML files (`*.html`)
- `data/processed/{source}/{YYYY-MM-DD}/{article_id}/`
  - `metadata.json` (source/homepage URL, article URL, title, scrape timestamp, file paths, article metadata)
  - `clean_text.txt` (cleaned extracted article text)
- `data/index/{source}/{YYYY-MM-DD}/`
  - run-level summary JSON records for each scrape execution

The `data/` contents are intentionally gitignored for local development use.

---

## 9) Troubleshooting

## A) Streamlit command not found

Symptom:
```text
streamlit: command not found
```

Fix:
1. Ensure virtualenv is activated.
2. Reinstall dependencies:

```bash
pip install -r requirements.txt
```

Or run with module syntax:

```bash
python -m streamlit run frontend/app.py
```

## B) Import errors (`ModuleNotFoundError`)

Typical cause: dependencies not installed in active environment.

Fix:
```bash
pip install -r requirements.txt
```

Then retry:
```bash
streamlit run frontend/app.py
```

## C) Invalid URL errors in UI

Symptom in UI: `Invalid URL: URL must include http:// or https:// and a valid host`

Fix:
- Provide a fully-qualified URL, e.g. `https://apnews.com`.

## D) Request/network failures

Symptom in UI: `Request failed: Failed to fetch URL: ...`

Common causes:
- no internet connectivity,
- DNS/network restrictions,
- target site blocking requests,
- timeout on slow websites.

Fixes:
- verify internet access,
- try another homepage,
- reduce scrape count,
- retry later.

## E) Low/empty analysis results

Symptom:
- `Articles scraped` is low or 0,
- no top words,
- few preview links.

Causes:
- homepage link patterns do not match current heuristics,
- article body not in `<p>` tags,
- site renders content dynamically with JS.

Fixes:
- try a different source homepage,
- increase max articles,
- prefer sites with server-rendered article pages.

## F) Environment variable confusion

Current project requires **no env vars**. If you set API keys, they are ignored by present code.

---

## 10) Validation Checklist (How findings were reflected)

- **Actual app entry point identified:** `frontend/app.py`
- **Dependency file identified:** `requirements.txt`
- **OpenAI integration status:** not implemented / not used
- **MCP server status:** helpers are runnable as imported Python functions; standalone MCP server process is not currently implemented
