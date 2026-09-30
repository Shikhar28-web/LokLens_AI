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
| 7. Classification Engine | ✅ Complete |
| 8. Support/Contradiction Engine | ✅ Complete |
| 9. Source Quality Scoring + Corroboration | ✅ Complete |
| 10. Deterministic Verdict API Report | ✅ Complete |
| 11. Image Preprocessing & Hashing | ✅ Complete |
| 12-16. Advanced Image Forensics | ⬜ Pending |
| 17. React Frontend UI | ⬜ Pending |

---

## Current Pipeline Progress (Phases 1-11)

We have successfully built the core end-to-end evidence verification pipeline! The system can take raw political claims (text) and images, break them down, and generate a structured JSON report. 

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
6. **Image Preprocessing & Hashing (Phase 11)**:
   - `Pillow` and `piexif` for reading EXIF camera data.
   - `imagehash` for calculating perceptual hashes (`pHash`, `dHash`, `aHash`) to detect visually similar but edited/cropped fake images.
   - Cryptographic `SHA-256` hashing to detect exact duplicate viral media.

---

## Example Output & Custom Queries

You can test **ANY** custom query or image right now! The backend API is fully functional and accepts any text/image submission. 

### Method 1: Using the Test Scripts
We have python scripts that hit the local API.
- For text: Edit `test_api_flow.py` and replace the hardcoded `"text": "..."` string with any political claim you want to test.
- For images: Edit `test_image_flow.py` and replace the `image_path` variable with any image on your computer.
```bash
python test_api_flow.py
python test_image_flow.py
```

### Method 2: Interactive API Docs (Swagger UI)
Since the FastAPI server is running, you can open your browser and go to:
http://localhost:8000/docs#/submissions/create_submission_api_submissions_post
Click **"Try it out"**, type any claim into the `text` box, and hit execute!

**What it returns for Text:**
1. A **Claim-vs-Evidence Matrix** scoring individual atomic sentences against scraped web sources using mathematical TF-IDF & BM25 retrieval vectors.
2. Distinct tags highlighting if evidence **SUPPORTS**, **CONTRADICTS**, or is simply an **ATTRIBUTED_CLAIM**.
3. **Source Quality Scores**: Evaluation of the domains (e.g., scoring `.gov` vs a blog).
4. A **Final Structured JSON Report** summarizing all extracted evidence, source URLs, and an overall deterministic verdict.

**What it returns for Images:**
1. **File Properties**: Extracted dimensions (width/height), mime-type, and file size.
2. **Hashes**: `SHA-256` for cryptographic integrity, and `pHash` (Perceptual Hash) for visual similarity matching against fake-news databases.
3. **EXIF Metadata**: Hidden camera, GPS, and timestamp data extracted directly from the raw image file.

*(Note: We will build the actual webpage with a beautiful text box UI in **Phase 17**, so end-users don't have to use the API directly!)*

---

## Important

- No OpenAI / Claude / Gemini / any generative LLM is used in the core pipeline.
- Every verdict comes from: NLP, TF-IDF, BM25, image forensics, source quality scoring.
- All evidence is provenance-tracked — no fabricated sources.
