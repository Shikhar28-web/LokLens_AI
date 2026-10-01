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
pip install -r requirements.txt
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
# Smoke tests
pytest tests/test_phase1_smoke.py -v

# Image forensics unit tests
set PYTHONPATH=.
pytest tests/unit/test_image_forensics.py -v

# OCR unit tests
pytest tests/unit/test_ocr.py -v
```

---

## Project Structure

```
backend/      FastAPI + SQLAlchemy + NLP + Image Forensics + OCR
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
| OCR | Tesseract (via pytesseract) |
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
| 12. Image Forensics (ELA, noise, frequency, copy-move) | ✅ Complete |
| 13. OCR Integration (Tesseract) | ✅ Complete |
| 13.5 Pipeline Bug Fixes (verdict bubbling, INSUFFICIENT_EVIDENCE, query diversity) | ✅ Complete |
| 14. Advanced Image Forensics (pHash Web Matching) | ⬜ Pending |
| 15. Multimodal Consistency (Image ↔ Claim separation) | ⬜ Pending |
| 16. Explanation Generation (human-readable forensic summaries) | ⬜ Pending |
| 17. React Frontend UI | ⬜ Pending |

---

## Current Pipeline Progress (Phases 1–13.5)

We have successfully built the core end-to-end evidence verification pipeline! The system can take raw political claims (text) **and** images, break them down, and generate a structured JSON report. The image pipeline and text pipeline are now **bridged** via OCR.

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

7. **Advanced Image Forensics (Phase 12)**:
   - Error Level Analysis (ELA) using `Pillow` and `numpy` to detect mismatched compression rates.
   - Noise variance analysis with `cv2` block-based operations to find copy/paste splices.
   - 2D FFT Frequency analysis to distinguish AI-generated (GAN/Diffusion) imagery from authentic ones.
   - ORB feature matching for copy-move forgery detection.
   - Combined heuristics for an overall AI-generation likelihood score with thresholds:
     - `ai_score ≥ 0.75` → `LIKELY_AI_GENERATED`
     - `ai_score ≥ 0.50` → `POSSIBLY_AI_GENERATED`
     - `ai_score ≥ 0.25` → `POSSIBLY_AUTHENTIC`
     - Below → `LIKELY_AUTHENTIC`

8. **OCR Integration (Phase 13)**:
   - `pytesseract` wrapping the Tesseract OCR binary to extract text from uploaded images.
   - OpenCV preprocessing pipeline (grayscale → Gaussian blur → Otsu's thresholding) applied before extraction for better accuracy on noisy images.
   - **Smart Pipeline Routing**: If a user uploads a screenshot of a claim without entering text manually, the pipeline automatically extracts the text via OCR and feeds it into the NLP fact-checking pipeline.
   - Graceful degradation: If the Tesseract binary is not installed on the host, the system logs a warning and skips OCR without crashing.
   - `ocr_text` and `ocr_confidence` (normalized 0.0–1.0) are stored in the database and returned in the API report.

9. **Pipeline Bug Fixes (Phase 13.5)**:
   - **Verdict Bubbling**: `image_verdict` and `image_confidence` now correctly bubble up to the root level of the JSON report alongside `claim_verdict`.
   - **INSUFFICIENT_EVIDENCE**: The pipeline no longer defaults to `UNSUPPORTED` when the web search retrieves zero articles (e.g., due to rate-limiting). It now correctly returns `INSUFFICIENT_EVIDENCE`.
   - **Diverse Search Queries**: The query generator now produces distinct strategies (exact phrase, broad keyword, entity-focused, contextual, news-targeted) instead of repetitive variations of the same string.

---

## Example Output & Custom Queries

You can test **ANY** custom query or image right now! The backend API is fully functional and accepts any text/image submission.

### Method 1: Using the Test Scripts
We have Python scripts that hit the local API.
- For text: Edit `test_api_flow.py` and replace the hardcoded `"text": "..."` string with any political claim you want to test.
- For images: Edit `test_image_flow.py` and replace the `image_path` variable with any image on your computer.
```bash
python test_api_flow.py
python test_image_flow.py
```

### Method 2: Interactive API Docs (Swagger UI)
Since the FastAPI server is running, open your browser and go to:
http://localhost:8000/docs#/submissions/create_submission_api_submissions_post

Click **"Try it out"**, type any claim into the `text` box, and hit execute!

**What it returns for Text:**
1. A **Claim-vs-Evidence Matrix** scoring individual atomic sentences against scraped web sources using mathematical TF-IDF & BM25 retrieval vectors.
2. Distinct tags highlighting if evidence **SUPPORTS**, **CONTRADICTS**, or is simply an **ATTRIBUTED_CLAIM**.
3. **Source Quality Scores**: Evaluation of the domains (e.g., scoring `.gov` vs a blog).
4. A **Final Structured JSON Report** summarizing all extracted evidence, source URLs, and an overall deterministic verdict.

**What it returns for Images:**
1. **File Properties**: Extracted dimensions (width/height), mime-type, and file size.
2. **Hashes**: `SHA-256` for cryptographic integrity, and `pHash` for visual similarity matching.
3. **EXIF Metadata**: Hidden camera, GPS, and timestamp data extracted directly from the raw image file.
4. **Forensic Scores**: ELA score, noise anomaly, frequency anomaly, copy-move score, and AI likelihood score (0.0–1.0) with a human-readable label.
5. **OCR Text**: Any text found in the image, along with an extraction confidence score.
6. **Root-Level Verdicts**: Both `image_verdict` and `claim_verdict` are returned at the top level of the report.

*(Note: We will build the actual webpage with a beautiful text box UI in **Phase 17**, so end-users don't have to use the API directly!)*

---

## Important

- No OpenAI / Claude / Gemini / any generative LLM is used in the core pipeline.
- Every verdict comes from: NLP, TF-IDF, BM25, image forensics, OCR, source quality scoring.
- All evidence is provenance-tracked — no fabricated sources.
- Image authenticity and claim truth are treated as **separate independent dimensions**.
