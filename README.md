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

### 1. Install Requirements
First, create a virtual environment and install the required packages. You can use the root `requirements.txt` file which contains all necessary dependencies:

```bash
# Navigate to the backend directory and create a virtual environment
cd backend
python -m venv venv

# Activate the virtual environment (Windows)
.\venv\Scripts\activate

# Go back to root and install the project requirements
cd ..
pip install -r requirements.txt
```

### 2. Start the Ollama Server
Install [Ollama](https://ollama.com/) if you haven't already. Open a **new, separate terminal** and start the Llama 3.1 model to power the AI reasoning features:

```bash
ollama run llama3.1
```

### 3. Start the Backend Server
Return to your first terminal (where the virtual environment is activated). Navigate into the `backend` folder and start the FastAPI server:

```bash
cd backend
uvicorn app.main:app --reload
```

Your API is now running! Visit the interactive docs at: http://127.0.0.1:8000/docs

---

## 📈 Development Phases

- [x] **Phase 1-10**: Core NLP Pipeline, Claim Extraction, Evidence Retrieval, and Verdict Classification.
- [x] **Phase 11-13**: Image Preprocessing, Advanced Image Forensics (ELA/Noise), and OCR Integration.
- [x] **Phase 14**: pHash Web Matching and Duplicate Detection.
- [x] **Phase 15**: Multimodal Consistency (Image vs Claim Separation).
- [x] **Phase 16**: Explanation Generation (Structured LLM Reasoning via Ollama).
- [ ] **Phase 17**: React Frontend UI.

---

