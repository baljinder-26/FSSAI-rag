# FSSAI Table-Aware RAG Assistant

A question-answering prototype over a 13-page extract of the FSSAI *Contaminants, Toxins and
Residues Regulations, 2011* (Version VI, 27.01.2022, `data/Compendium_Contaminants_Regulations.pdf`).
Ask for a limit, for example "lead in tea", and get the exact number **with the PDF page, the table
and the source row**. If the document does not support an answer, the system returns exactly:

`Information not found in the current guidelines.`

## 1. Run it locally

One `uv` project and one virtual environment for both the API and the UI.

```bash
uv sync                                  # 1. install dependencies
copy .env.example .env                   # 2. create .env, then put your OPENAI_API_KEY in it
uv run python -m scripts.ingest          # 3. parse the PDF and build the indexes (uses OpenAI embeddings)
uv run uvicorn app.main:app --reload     # 4. terminal 1: API    -> http://localhost:8000
uv run streamlit run streamlit_app.py    # 5. terminal 2: UI     -> http://localhost:8501
uv run pytest                            # tests (no paid OpenAI calls)
```

- `uv run python -m scripts.ingest --no-embed` builds only the SQLite data (no OpenAI call).
- Models are set in `.env` (`OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL`). If the chat model is not
  available to your account, the API returns a clear 503 configuration error and never invents an answer.
- The API key is read only by the Python backend. The browser/Streamlit code never sees it.

## 2. The problem and the approach

Regulatory tables break ordinary RAG: if a table is cut into fixed-size text pieces, a limit such as
`5.0` ends up in a chunk that no longer says *which metal* or *which food* it belongs to. The
approach below keeps every number attached to its row and context, and refuses to answer when the
evidence is not there.

```
PDF
 |  Step 1  pdfplumber: read each page (table rows, prose, page text)
 v
 |  Step 2  table_parser: rows -> validated records (metal, food, limit, unit, condition, page)
 v
 |  Step 3  chunking: one chunk per table row + one chunk per prose clause
 v
 |  Step 4  indexing: SQLite (exact records) + Chroma (embeddings) + BM25 (keywords)
 v
 |  Step 5  retrieval: exact lookup first, hybrid search for explanatory text
 v
 |  Step 6  answer: OpenAI chat sees only the evidence, Python guardrails check the result
 v
 Answer + sources (document, page, table, row)   ->   FastAPI /query   ->   Streamlit UI
```

## 3. Each step: what it does, why it is needed, what we get

### Step 1: PDF ingestion (`app/ingestion.py`)
- **What:** reads every page with **pdfplumber**, separating the table rows, the text outside the table,
  and the full page text.
- **Why:** the table must be read cell by cell. Plain text extraction would merge the three columns into
  one line and lose which number belongs to which food.
- **Why pdfplumber :** the PDF is a digital document with ruled table lines, so pdfplumber
  keeps every cell intact and needs far fewer dependencies.
- **Result:** 13 pages, 386 raw table rows (382 data rows) plus the prose of chapters 1-2.

### Step 2: Table parsing and validation (`app/table_parser.py`)
- **What:** turns raw rows into structured records with these fields: metal contaminant, food, limit
  value, limit as printed, unit, limit type, condition, regulation, document, page, table id, original row.
- **Why:** the PDF has quirks that would otherwise give wrong answers:
  - The metal name is printed only on the first row of each group, so it is **carried to every row**.
  - A row cut by a page break (cadmium "Cereal grains ... and Quinoa", pages 11-12) is **joined back**.
  - Conditions sit inside the limit cell ("5.0 on dry matter basis", "15 (But not less than 2.8)"). They
    are **kept** because they change the meaning of the limit.
  - "expressed in mg/L" rows get the right unit.
- **Validation:** every row's number must be found on the page next to that row's text. Otherwise the row
  is flagged `needs_review` and printed by the ingest script. Nothing is silently corrected.
- **Result:** **382 records, 0 flagged.** The one long table is split into 11 table ids, one per metal
  (for example `Regulation 2.1.1(2) table - 1. Lead`).

### Step 3: Chunking (`app/chunking.py`)
Two chunk types, and **no fixed-size splitting anywhere**.

| Type | Rule | Example content |
|---|---|---|
| **Row chunk** (table-aware) | One chunk per table row. Each chunk repeats the regulation, table id, metal, food, limit with condition, unit and source page. | `Metal contaminant: Lead / Article of food: Tea / Limit: 5.0 on dry matter basis ... page 6` |
| **Prose chunk** (section-aware) | Split at clause boundaries (1.1.1, 1.2.1, 2.1.1) and prefixed with the chapter and section heading. | The definition of "Crop contaminant" stays in one piece with its heading. |

- **Why:** a row chunk can be understood on its own, which is exactly what the assignment asks ("every chunk
  retains its overarching food category and regulatory context"). Clause-based prose chunks avoid cutting a
  definition in half.
- **Result:** 382 row chunks + 4 prose chunks = 386 chunks.

### Step 4: Indexing (`app/indexing.py`)
- **SQLite** stores the validated records and chunks. *Why:* exact, auditable data that does not depend on
  similarity search. Re-running ingestion rewrites the tables, so nothing is duplicated.
- **Chroma + OpenAI `text-embedding-3-small`** stores the embeddings in a local vector store. *Why:* finds
  chunks by meaning (for example "heavy metal" wording).
- **BM25** is a keyword index. *Why:* exact words such as "Tea", "Quinoa" or "vanaspati" must match
  precisely, which keyword search does better than embeddings.
- **Result:** three indexes under `storage/` (generated locally and not committed).

### Step 5: Retrieval (`app/retrieval.py`)
- **What:** the question is first parsed for a metal ("lead", "methyl mercury") and food words ("tea",
  "polished rice"). Then:
  1. **Exact lookup** in the records: a row qualifies only if *every* food word is in its food name.
  2. **Hybrid search** (BM25 + Chroma, combined with reciprocal-rank fusion) finds explanatory prose.
  3. Prose is kept only if it shares the question's wording.
- **Why:** exact lookup gives the right number for the right metal and food. "Lead in rice" returns
  nothing, because rice is listed only under cadmium. Words not in the document at all (for example
  "aflatoxin") also block a table answer.
- **Result:** a short list of numbered evidence items (E1, E2, ...), or nothing.

### Step 6: Answer generation and guardrails (`app/answer.py`)
- **What:** if there is no evidence, the system returns the fallback **without calling the LLM**. Otherwise
  OpenAI chat (via LangChain, temperature 0) receives only the question and the evidence, with rules:
  never guess, convert or calculate, quote the limit with its unit and condition, and report each matching row.
- **Python checks after the model answers:**
  - It must cite at least one real evidence id.
  - Every number in the answer must appear in the cited evidence (or the question), or the fallback is returned.
  - Sources (document, page, table, row) come from stored metadata, never from model-written text.
- **Why:** a language model can sound confident while being wrong. These checks make a made-up limit
  very unlikely to reach the user.
- **Result:** an answer plus structured sources, or the exact fallback.

### Step 7: API (`app/main.py`)
- `GET /health` reports the document name, how many rows and chunks are indexed, and whether a key is set.
- `POST /query` takes `{"question": "..."}` and returns `{"answer": "...", "sources": [{document, page, table, evidence}]}`.
- Provider or configuration errors return a clear message (HTTP 503) with no secrets and no invented answer.
- **Why:** the assignment asks for a FastAPI microservice, and keeping the model calls here keeps the key safe.

### Step 8: UI (`streamlit_app.py`)
- A question box with a **🎤 mic button** (browser speech recognition, Chrome/Edge: it types your spoken
  question into the box) and an **Ask** button. Enter also submits.
- An **Answer** section. The exact fallback is shown as a warning.
- A **Sources** panel for each cited item: document, **PDF page**, table id and the exact source row.
- A sidebar showing the document name and index status, plus **📄 Show document**, which displays the real
  PDF page (choose a page number) so a user can verify the answer against the original.
- Dark theme with a grid background. **Why:** the user can trust an answer only if they can check the
  page it came from.

## 4. Project structure

```
app/            config, schemas, ingestion, table_parser, chunking, indexing, retrieval, answer, main (FastAPI)
scripts/ingest.py   parse the PDF and build the indexes
streamlit_app.py    UI
tests/              automated tests (see below)
data/               the PDF
storage/            generated SQLite + Chroma (not committed)
```

## 5. Tests

`uv run pytest` runs 20 tests with a fake model (no paid calls, about 10 seconds):

- Row chunks keep headers, metal, food, unit, condition and source page.
- Exact metal + food lookup; the same food under a different metal gives a different limit.
- A condition that changes the meaning of a limit ("15 (But not less than 2.8)") is kept.
- A question combining a table row and explanatory text returns both.
- Missing information returns exactly `Information not found in the current guidelines.`
- Sources carry the real page and table, and an invented number is blocked.
- API errors never expose the API key.
- `tests/test_integration.py` runs real OpenAI calls when a key and an index are present.

## 6. Evaluation test set

| # | Query | Expected (from the PDF) | What it tests |
|---|---|---|---|
| 1 | What is the maximum limit of lead in tea? | 5.0 on dry matter basis, page 6 | Basic table lookup with a condition |
| 2 | What is the limit of copper in infant foods? | 15 (But not less than 2.8), page 7 | A condition that changes the limit |
| 3 | What is the cadmium limit in polished rice? | 0.4, page 12 | Multi-word food match |
| 4 | What is the lead limit in rice? | Information not found in the current guidelines. | Rice exists only under cadmium, so no wrong-metal answer |
| 5 | What is the mercury limit in tuna, and what is a crop contaminant? | 1.0, page 12, plus the definition on page 1 | Table row and explanatory text together |

## 7. Limitations (honest notes)
- Scope is one 13-page extract, not the full compendium. The PDF has one long table, which is split into 11 metal groups.
- Matching is deliberately strict. A question worded differently from the table (for example "dried tea
  leaves", "brown rice") returns the fallback rather than a guess. Synonyms are not expanded.
- "All foods" rows (methyl mercury) are matched only when the question names no specific food.
- The number check is strict: if the model writes "5" where the PDF prints "5.0", the answer can be rejected
  and the fallback returned.
- The mic uses the browser's speech recognition, so it needs Chrome or Edge and an internet connection.
- The automatic validation and guardrails reduce errors but cannot rule them out. Verify important limits
  against the PDF page shown in the sources. No accuracy percentage is claimed.
- Page numbers are PDF page numbers (1-13) of the extract.
