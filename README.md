# LokLens AI

**Multimodal Evidence-Based News & Image Verification**

> Core principle: No generative AI. Every verdict is derived from structured evidence.

---

## Quick Start

### Backend

```bash
cd backend
python -m venv venv
.\venv\Scripts\activate          # Windows
pip install -r requirements-phase1.txt
uvicorn app.main:app --reload --port 8000
```

Visit: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit: http://localhost:5173

---

## Run Tests

```bash
cd backend
.\venv\Scripts\activate
pytest tests/test_phase1_smoke.py -v
```

---

## Project Structure

```
backend/      FastAPI + SQLAlchemy + NLP + Image Forensics
frontend/     React + Vite + TypeScript + Tailwind CSS
data/         uploads, cache, chroma (created at runtime)
docs/         Architecture docs
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, SQLAlchemy, Alembic |
| NLP | NLTK, spaCy, scikit-learn, rank-bm25 |
| Image | OpenCV, Pillow, imagehash, piexif |
| OCR | Tesseract |
| Search | DuckDuckGo (pluggable) |
| Vector DB | ChromaDB (optional) |
| Frontend | React, Vite, TypeScript, Tailwind CSS |
| Database | SQLite (dev) / PostgreSQL (prod) |

---

## Verdict Scale

### Claim
`SUPPORTED` · `LIKELY_SUPPORTED` · `PARTIALLY_SUPPORTED` · `MISLEADING_CONTEXT` · `UNSUPPORTED` · `CONTRADICTED` · `INSUFFICIENT_EVIDENCE`

### Image
`LIKELY_AUTHENTIC` · `POSSIBLY_AUTHENTIC` · `POSSIBLY_MANIPULATED` · `LIKELY_MANIPULATED` · `POSSIBLY_AI_GENERATED` · `LIKELY_AI_GENERATED` · `INCONCLUSIVE`

---

## Development Phases

| Phase | Status |
|---|---|
| 1. Project skeleton + DB + API + Frontend | ✅ Complete |
| 2. NLP pipeline + claim extraction | ✅ Complete |
| 3. Query Generation | ✅ Complete |
| 4. Search provider (DuckDuckGo) | ✅ Complete |
| 5. Document Scraping (Scrapy) | ✅ Complete |
| 6. Evidence Retrieval (TF-IDF & BM25) | ✅ Complete |
| 7. Verdict Engine (NLI / Classification) | ✅ Complete |
| 8-20. APIs, UI Integration, Image Forensics | ⬜ Pending |

---

## Current Pipeline Progress (Phases 1-7)

We have successfully built the core end-to-end evidence verification pipeline. The system takes raw political claims, breaks them down, searches the web, extracts documents, scores evidence mathematically, and outputs a structured report. 

### Tools & Technologies Used:
1. **NLP Preprocessing & Claim Extraction**: 
   - `spaCy` (en_core_web_sm) for entity extraction (identifying quantities, political actors, locations, dates).
   - Regex-based compound sentence splitting designed to preserve dependent clauses and noun-chunks.
2. **Search Provider**: 
   - `duckduckgo_search` via asynchronous API requests.
3. **Web Scraping**:
   - `Scrapy` integrated dynamically via subprocesses to bypass anti-bot JavaScript firewalls.
   - `BeautifulSoup4` for DOM parsing and cleaning.
4. **Evidence Retrieval**:
   - `NLTK` for precise sentence boundary chunking.
   - `scikit-learn` for TF-IDF Vectorization.
   - `rank-bm25` for BM25 term frequency matching.
5. **Verdict Classification Engine**:
   - Zero-Shot Natural Language Inference logic mapping to `SUPPORTS`, `CONTRADICTS`, `CONTEXT`, and `ATTRIBUTED_CLAIM`.
   - Advanced heuristic mock implemented (with fallback architectures for HuggingFace Transformers `distilbert-base-uncased-mnli`).

---

## Example Output

You can currently test the pipeline via the console:
```bash
python ./backend/test_pipeline.py "The Prime Minister announced that the government had created 10 crore new jobs in India since 2014. The government said the figure showed the success of its economic policies, while opposition leaders argued that the number included informal and existing jobs and therefore did not represent 10 crore new jobs."
```

**What it returns:**
1. A **Claim-vs-Evidence Matrix** scoring individual atomic sentences against scraped web sources using mathematical TF-IDF & BM25 retrieval vectors.
2. Distinct tags highlighting if evidence **SUPPORTS**, **CONTRADICTS**, or is simply an **ATTRIBUTED_CLAIM** (e.g., merely repeating an accusation/quote rather than proving it).
3. A **Final Structured Report** summarizing:
   - **Established Facts**: Verified, supported evidence.
   - **Disputed / Contradicted**: Counter-evidence found.
   - **Allegations / Unverified**: Parts of the claim lacking external proof, representing pure political accusations.

---

## Important

- No OpenAI / Claude / Gemini / any generative LLM is used in the core pipeline.
- Every verdict comes from: NLP, TF-IDF, BM25, image forensics, source quality scoring.
- All evidence is provenance-tracked — no fabricated sources.
