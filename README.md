# LokLens AI

**Multimodal Evidence-Based News & Image Verification**

LokLens AI is a comprehensive pipeline designed to verify political news claims and analyze image authenticity. It breaks down raw text or images, searches the web for independent evidence, and uses AI-driven image forensics to produce highly structured, explainable verdicts.

---

## 🚀 Key Features

1. **Explainable AI Reasoning (Local LLMs)**
   - Integrates seamlessly with **Ollama** (e.g., `llama3.1`) or OpenAI to synthesize web evidence into a structured JSON report. 
   - Dynamically cites the exact domains and URLs that support or contradict a claim, providing human-readable context.
2. **Robust Image Forensics**
   - **Error Level Analysis (ELA)** & **Noise Variance**: Detects image splices and copy-move forgery.
   - **Frequency Analysis**: Uses 2D FFT to identify AI-generated/GAN checkerboard artifacts.
   - **Perceptual Hashing (pHash)**: Identifies viral recirculation of near-duplicate images.
3. **Automated Evidence Retrieval**
   - Extracts claims via OCR and NLP (spaCy/NLTK).
   - Generates targeted queries and scrapes web results via DuckDuckGo.
   - Ranks evidence mathematically using TF-IDF and BM25.
4. **Multimodal Consistency Verification**
   - Treats claim truth and image authenticity as independent dimensions (e.g., correctly flags real news portrayed by an AI-generated synthetic image).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, FastAPI, SQLAlchemy, SQLite (dev) |
| **NLP & Search** | spaCy, NLTK, scikit-learn, rank-bm25, DuckDuckGo |
| **Explainability** | Ollama (Local Llama 3), OpenAI (fallback) |
| **Image Forensics**| OpenCV, Pillow, imagehash, piexif |
| **OCR Extraction** | Tesseract (pytesseract) |
| **Frontend** | React, Vite, TypeScript, Tailwind CSS (Pending Phase 17) |

---

## 🚦 Quick Start

### 1. Setup the Backend
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Start Ollama for Explainable Verdicts
Install [Ollama](https://ollama.com/) and run the Llama 3.1 model in a separate terminal:
```bash
ollama run llama3.1
```

### 3. Run the Server
In your backend terminal, set the environment variables to use Ollama and start the API:
```powershell
$env:USE_OLLAMA="true"
$env:OLLAMA_MODEL="llama3.1"
uvicorn app.main:app --reload --port 8000
```

Visit the interactive API docs at: http://localhost:8000/docs

---

## 📈 Development Phases

- [x] **Phase 1-10**: Core NLP Pipeline, Claim Extraction, Evidence Retrieval, and Verdict Classification.
- [x] **Phase 11-13**: Image Preprocessing, Advanced Image Forensics (ELA/Noise), and OCR Integration.
- [x] **Phase 14**: pHash Web Matching and Duplicate Detection.
- [x] **Phase 15**: Multimodal Consistency (Image vs Claim Separation).
- [x] **Phase 16**: Explanation Generation (Structured LLM Reasoning via Ollama).
- [ ] **Phase 17**: React Frontend UI.

---

## 🧪 Testing the Pipeline

You can test the entire pipeline end-to-end (including the new Ollama integration) using the built-in test scripts:

```bash
# Test a raw text claim
python test_api_flow.py

# Test a fake/synthetic image containing real news
python test_real_news_image.py
```
