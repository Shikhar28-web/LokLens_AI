# LokLens AI 🔍⚖️

**Multimodal Evidence-Based News, Social Media & Image Verification Pipeline**

LokLens AI is an end-to-end investigative platform engineered to detect political misinformation, verify viral claims, and analyze digital image authenticity. Unlike black-box generative models that produce hallucinated answers, LokLens AI derives every verdict deterministically from independent web sources, fact-checking registries, social media discourse (**X/Twitter, Instagram, Reddit, YouTube**), and mathematical digital image forensics.

---

## 🚀 Key Features

1. **Social Media Intelligence & Discourse Monitoring**
   - **X (Twitter), Instagram, Reddit, and YouTube** site-targeted query generation.
   - Extracts post text, author handles (`@username`, `r/subreddit`), engagement metadata, and publication context.
   - Synthesizes social media discourse alongside mainstream news in the final executive summary.

2. **Industrial-Grade OCR & Layout Classification**
   - Detects layout formats automatically: `screenshot`, `meme`, `document`, `news_graphic`, or `pure_photo`.
   - Multi-variant image preprocessing pipeline: **CLAHE** (for dark-mode screenshots), **Bilateral edge-preserving denoising**, **Adaptive Gaussian binarization**, **Sharpening**, and **Automatic deskewing**.
   - Filters out mobile UI noise (battery %, carrier signals like `5G`/`VoLTE`, clock timestamps) to prevent claim corruption.
   - Native multilingual support for English and Hindi / Devanagari Unicode (`\u0900-\u097F`).

3. **Digital Image Forensics (Pixel & Frequency Domain)**
   - **Error Level Analysis (ELA)**: Pinpoints compression inconsistencies from photo splicing.
   - **Local Noise Variance**: Block-level variance analysis to detect clone and cut-paste manipulations.
   - **2D Fourier Transform (FFT)**: Inspects radial power distribution to detect GAN/Diffusion synthetic artifacts.
   - **Copy-Move Clone Detection**: ORB keypoint clustering with Hamming distance matching to identify duplicated regions.
   - **Perceptual Hashing (pHash)**: 64-bit DCT fingerprinting to detect viral image recirculation and recycling.

4. **Independent Multimodal Verification**
   - Decouples claim validity from image authenticity (e.g., flags authentic news depicted using an AI-generated image, or genuine historical photos recycled with false new claims).
   - Gracefully handles **pure photos without text** through deep visual forensic reporting without failing.

5. **Interactive Intelligence Dashboard**
   - Modern React + TypeScript + Tailwind UI featuring social media post cards with platform badges, claim breakdown, confidence meters, and external source citations.

---

## ⚖️ Thresholds, Decision Rules & Calibration Values

LokLens AI relies on strictly defined, calibrated mathematical thresholds across all analysis modules. Below is the complete specification:

### 1. AI Generation & Image Manipulation Calibration (`ai_score_estimator.py`)

The composite AI likelihood score combines frequency, noise, and compression anomalies:
$$\text{AI Score} = (\text{Frequency Score} \times 0.50) + (\text{Noise Score} \times 0.25) + (\text{ELA Score} \times 0.25)$$

| Metric / Score Condition | Verdict Label | Interpretation |
|---|---|---|
| `Frequency > 0.80` **AND** `ELA < 0.15` **AND** `Noise < 0.15` | `UNKNOWN` (Infographic) | Guard against false positives on text-heavy screenshots where character edges produce high frequency. |
| `AI Score >= 0.75` | `LIKELY_AI_GENERATED` | Strong frequency anomalies characteristic of GANs and Diffusion models (Midjourney, DALL-E, Flux). |
| `AI Score >= 0.55` **OR** (`AI Score >= 0.40` **AND** `Frequency > 0.60`) | `POSSIBLY_AI_GENERATED` | Elevated synthetic spectral signatures detected. |
| `ELA Score >= 0.75` **OR** `Copy-Move Score >= 0.60` | `LIKELY_MANIPULATED` | High confidence of manual Photoshop, clone-stamping, or splicing. |
| `ELA Score >= 0.60` **OR** `Noise Score >= 0.60` **OR** `Copy-Move >= 0.40` | `POSSIBLY_MANIPULATED` | Localized compression or sensor noise discrepancies detected. |
| `AI, ELA, Noise, Copy-Move` all `< 0.20` | `LIKELY_AUTHENTIC` | Natural camera sensor noise, consistent 1/f² frequency dropoff, uniform JPEG compression. |
| `AI Score < 0.35` **AND** `max(ELA, Noise, Copy-Move) < 0.35` | `POSSIBLY_AUTHENTIC` | Generally consistent photographic attributes. |

---

### 2. Error Level Analysis (ELA) (`ela_analyzer.py`)

- **Resave Quality**: `quality = 90`
- **Standard Deviation Normalization**:
  $$\text{Score} = \min\left(\max\left(\frac{\sigma - 10}{70}, 0.0\right), 1.0\right)$$
- **Calibrated Values**:
  - $\sigma < 30$ ($\text{Score} < 0.28$): Natural, uniform compression across entire frame.
  - $30 \le \sigma \le 52$ ($\text{Score} \in [0.28, 0.60]$): Mild re-compression or minor overlay.
  - $\sigma > 52$ ($\text{Score} > 0.60$): Distinct compression levels indicating spliced elements.

---

### 3. Local Noise Variance Analysis (`noise_analyzer.py`)

- **Analysis Window**: `16 × 16` pixel non-overlapping blocks.
- **Noise Isolation**: Median filter ($3\times3$) subtraction: $\text{Noise} = |I - \text{medianBlur}(I)|$.
- **Normalization Formula**:
  $$\text{Score} = \min\left(\max\left(\frac{\text{Var}(\text{Block Variances}) - 50}{300}, 0.0\right), 1.0\right)$$
- **Calibrated Values**:
  - $\text{Variance of Variances} < 50$ ($\text{Score} = 0.0$): Uniform sensor noise profile (camera capture).
  - $\text{Variance of Variances} > 230$ ($\text{Score} > 0.60$): Heterogeneous noise profile indicative of cut-and-paste compositing.

---

### 4. 2D FFT Frequency Analysis (`frequency_analyzer.py`)

- **Radially Averaged Power Spectrum**: Evaluates energy across spatial frequency radii $r = \sqrt{(x - x_0)^2 + (y - y_0)^2}$.
- **Frequency Bands**:
  - Mid-frequency: $r \in [20, \min(N/2, 100)]$
  - High-frequency: $r \in [\min(N/2, 100), \max]$
- **Energy Ratio**: $\text{Ratio} = \frac{\sum E_{\text{high}}}{\sum E_{\text{mid}}}$
- **Normalization Formula**:
  $$\text{Score} = \min\left(\max\left(\frac{\text{Ratio} - 0.05}{0.15}, 0.0\right), 1.0\right)$$
- **Calibrated Values**:
  - $\text{Ratio} < 0.08$ ($\text{Score} < 0.20$): Natural $1/f^2$ spectral decay.
  - $\text{Ratio} > 0.14$ ($\text{Score} > 0.60$): Checkerboard frequency spikes caused by upsampling / deconvolution layers in generative models.

---

### 5. Copy-Move Forgery Detection (`copy_move_detector.py`)

- **Keypoints**: ORB detector with $N = 1000$ points.
- **Minimum Spatial Distance Threshold**: $d_{\text{spatial}} > 50\text{ px}$ (filters out self-matches and contiguous textures).
- **Hamming Descriptor Distance**: $d_{\text{Hamming}} < 30$ (strict feature equivalence).
- **Scoring Formula**:
  $$\text{Score} = \min\left(\max\left(\frac{N_{\text{matches}} - 5}{20}, 0.0\right), 1.0\right)$$
- **Calibrated Values**:
  - $N_{\text{matches}} \le 5$ ($\text{Score} = 0.0$): Stochastic background matches.
  - $N_{\text{matches}} \ge 17$ ($\text{Score} \ge 0.60$): Clustered feature duplication (cloning/healing brush).

---

### 6. Perceptual Hashing (pHash) Deduplication (`phash_matcher.py`)

- **Hash Architecture**: 64-bit Discrete Cosine Transform (DCT) pHash.
- **Hamming Distance Threshold**: $D \le 10\text{ bits}$ (out of 64 bits $\approx 84.4\%$ similarity).
  - $D = 0$: Exact duplicate.
  - $1 \le D \le 10$: Near-duplicate (handles cropping, resaving, minor color shifts, watermarks).
  - $D > 10$: Dissimilar image.
- **Recirculation Classification**:
  - Prior Sightings $= 0$: `NO_MATCH_FOUND` (First time seen).
  - Prior Sightings $\in [1, 2]$: `PREVIOUSLY_SEEN`.
  - Prior Sightings $> 2$: `VIRAL_RECIRCULATION` (High probability of recycled context).

---

### 7. Multimodal Consistency Scoring (`multimodal_consistency.py`)

Combines text, date, and visual signals into a unified consistency score $\in [0, 1]$:
$$\text{Score} = \frac{0.35 \times S_{\text{OCR}} + 0.20 \times S_{\text{Date}} + 0.25 \times S_{\text{Viral}} + 0.20 \times S_{\text{AI}}}{\sum \text{Weights}}$$

- **OCR $\leftrightarrow$ Claim Keyword Overlap ($S_{\text{OCR}}$)**:
  - Overlap $< 0.05$: Flag `OCR_TEXT_MISMATCH`
  - Overlap $> 0.40$: Flag `OCR_TEXT_CONSISTENT`
- **Date Consistency ($S_{\text{Date}}$)** (EXIF creation date vs Claim year):
  - $\Delta_{\text{years}} = 0 \rightarrow S_{\text{Date}} = 1.0$
  - $\Delta_{\text{years}} \le 1 \rightarrow S_{\text{Date}} = 0.7$
  - $\Delta_{\text{years}} \in [2, 3] \rightarrow S_{\text{Date}} = 0.4$ (Flag `DATE_MISMATCH`)
  - $\Delta_{\text{years}} > 3 \rightarrow S_{\text{Date}} = 0.0$ (Flag `DATE_MISMATCH_CRITICAL`: Outdated image recycled for breaking news)
- **Viral Recirculation Factor ($S_{\text{Viral}}$)**:
  - Viral recirculation: $0.1$ | Possible duplicate: $0.4$ | Unique: $0.8$
- **AI Synthetic Penalty ($S_{\text{AI}}$)**:
  - `likely_ai_generated`: $0.0$ | `possibly_ai_generated`: $0.3$ | `likely_authentic`: $1.0$

---

### 8. Source Authority & Credibility Scoring (`quality_scorer.py`)

| Source Tier | Authority Score | Example Domains / Platforms |
|---|---|---|
| **Government & Academic** | `1.00` | `.gov`, `.gov.in`, `.nic.in`, `.edu` |
| **Certified Fact-Checkers** | `0.95` | `altnews.in`, `boomlive.in`, `factly.in`, `snopes.com`, `politifact.com`, `pib.gov.in` |
| **Mainstream News Agencies** | `0.85` | `reuters.com`, `apnews.com`, `thehindu.com`, `indianexpress.com`, `bbc.com`, `ndtv.com` |
| **Generic Web Sources** | `0.55` | Unclassified public blogs and regional portals |
| **Social Media Platforms** | `0.45` | `x.com` / `twitter.com`, `instagram.com`, `reddit.com`, `youtube.com`, `threads.net`, `t.me` |

---

### 9. Evidence Stance & Verification Rules (`evidence_classifier.py` & `pipeline.py`)

- **Relevance Gate**:
  - Web articles: Must achieve semantic keyword overlap ratio $\ge 0.18$ and content length $\ge 50$ chars.
  - Social media posts: Content length $\ge 20$ chars and subject match.
- **Verdict Synthesis Logic**:
  - `SUPPORTED`: Reliable citations directly corroborate the claim without contradictions.
  - `CONTRADICTED`: Authoritative sources / fact-checks explicitly refute the claim with debunk terminology.
  - `LIKELY_MISLEADING`: Both supporting and contradictory context found; selective distortion or exaggerated facts.
  - `INSUFFICIENT_EVIDENCE`: Relevant documents located, but conclusive independent confirmation is lacking.
  - `UNVERIFIED`: Zero search results found across web and social media.
  - `NO_TEXT_FOUND`: Image-only upload; assessed purely on forensic integrity markers.

---

### 10. OCR Engine & Layout Detection Thresholds (`ocr_engine.py`)

- **Screenshot Detection**: Aspect ratio $\ge 1.70$ (tall mobile aspect 9:16 or 9:19.5) or $\le 0.60$.
- **Document Detection**: $> 55\%$ of pixels have luminance $> 235$ (white background).
- **Dark Mode Screenshot**: $> 60\%$ of pixels have luminance $< 25$.
- **Meme Detection**: High contrast variance ($\sigma > 60$) in both top 20% and bottom 20% bands.
- **Auto-Deskew**: Corrects rotation if detected skew angle $| \theta | \in [1.0^\circ, 45.0^\circ]$.
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization with $\text{clipLimit} = 2.5$, grid $(8\times8)$.

---

## 🧠 Machine Learning Models & Datasets

Alongside rule-based digital forensics and web evidence retrieval, LokLens AI integrates **three specialized ML models** (configured in [datasets.yaml](backend/ml/config/datasets.yaml), exposed via [ml_routes.py](backend/app/api/ml_routes.py), and orchestrated by [unified_router.py](backend/ml/inference/unified_router.py)).

| Model | Primary Architecture | Baseline Architecture | Dataset(s) Used | Task & Labels | Held-Out Test Accuracy | API Endpoint |
|---|---|---|---|---|---|---|
| **FEVER Claim Verifier** | `cross-encoder/nli-deberta-v3-small` | TF-IDF + Logistic Regression | `pietrolesci/nli_fever` (FEVER v1.0 NLI) | Claim vs. Evidence NLI (`SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`) | **`74.85%`** *(9,999 pairs)* | `POST /api/ml/verify-claim` |
| **FakeNewsNet Political Classifier** | `distilbert-base-uncased` | XGBoost (TF-IDF + spaCy NER) | FakeNewsNet (`BuzzFeed` + `PolitiFact`) + ISOT Political News | Stylistic News Classification (`REAL`, `FAKE`) | **`93.31%`** *(DistilBERT)* / **`92.52%`** *(XGBoost)* *(254 articles)* | `POST /api/ml/classify-news` |
| **NewsCLIPpings Consistency** | PyTorch MLP (`[512, 256]`) | XGBoost (`concat+diff+cos`) | `NewsCLIPpings` (`merged_balanced`) + CLIP embeddings | Image–Caption Consistency (`MATCHED`, `MISMATCHED`) | **`72.16%`** *(ROC-AUC `0.7991`, 7,264 pairs)* | `POST /api/ml/image-text-consistency` |

---

### 1. FEVER Claim-Evidence Verifier (`models/fever_claim_verifier/`)
* **Dataset**: **`pietrolesci/nli_fever`** (HuggingFace NLI-flattened FEVER v1.0 — **208,346** train / **19,998** val / **19,998** test claim-evidence pairs).
* **Why we use it**: A stylistic classifier alone cannot tell whether a specific number, date, or election outcome (e.g., *"240 seats"* vs. *"260 seats"*) is true. FEVER trains a Natural Language Inference (NLI) model to compare a **claim** directly against a **retrieved evidence passage** and decide whether the evidence supports or contradicts it.
* **How it works**:
  * **Preprocessing** (`fever_preprocessor.py`): Maps `nli_fever`'s `premise` (claim) and `hypothesis` (evidence) pairs to LokLens's unified label space (`0 = SUPPORTS`, `1 = REFUTES`, `2 = NOT ENOUGH INFO`) and deduplicates training pairs (`208,346 → 180,295`).
  * **Model & Training** (`train_fever.py`): Fine-tunes `cross-encoder/nli-deberta-v3-small` (classifier head aligned from NLI `contradiction/entailment/neutral` to FEVER's `0/1/2` order) on a stratified balanced subset of **18,000** examples (`6,000` per class) using `bfloat16` mixed precision (`max_length=256`, `epochs=2`, `lr=2e-5`).
  * **Evaluation (9,999 held-out test pairs)**:
    * **Model achieving `74.85%` accuracy**: Fine-tuned **`cross-encoder/nli-deberta-v3-small`** (`models/fever_claim_verifier/best_model/`). *(The TF-IDF + Logistic Regression baseline only achieved `29.54%`.)*
    * **Test Accuracy**: **`74.85%`** | **Macro / Weighted F1**: `0.7484`
    * Per-class F1: `SUPPORTS` (`0.81`), `REFUTES` (`0.76`), `NOT ENOUGH INFO` (`0.67`).
  * **Single Input Format (`POST /api/ml/verify-claim`)**:
    ```json
    {
      "claim": "The BJP won 240 seats in India's 2024 parliamentary election.",
      "evidence": "The Bharatiya Janata Party won 240 seats in India's 2024 Lok Sabha election."
    }
    ```
  * **Inference** (`fever_inference.py`): Accepts `(claim, evidence)` and returns softmax probabilities over `SUPPORTS`, `REFUTES`, and `NOT ENOUGH INFO`.

---

### 2. FakeNewsNet + ISOT Political News Classifier (`models/fakenewsnet_classifier/`)
* **Datasets**:
  * **FakeNewsNet** (`data/datasets/fakenewsnet/`): `BuzzFeed_fake_news_content.csv` (91), `BuzzFeed_real_news_content.csv` (91), `PolitiFact_fake_news_content.csv` (120), `PolitiFact_real_news_content.csv` (120).
  * **ISOT Fake & Real News** (`data/datasets/isot/`): Stratified political subset (`1,200` `Fake.csv` articles from `politics` / `Government News` + `1,200` `True.csv` articles from `politicsNews`).
  * **Combined & Deduplicated Split**: **2,532** balanced articles (**1,266** `REAL` / **1,266** `FAKE`) split into **2,025** train / **253** val / **254** test, with conflicting titles removed to prevent cross-split leakage.
* **Why we use it**: Flags **stylistic and linguistic patterns** associated with clickbait, tabloid, or sensational misinformation (e.g., hyperbole, conspiratorial framing, emotional tone).
  > **Important Distinction**: This model is a **style indicator**, *not* a factual verifier. A fabricated claim written in calm Reuters-style prose must be checked with **FEVER + Web Evidence Retrieval**, not FakeNewsNet alone.
* **How it works**:
  * **Shortcut-Bias Mitigation** (`fakenewsnet_preprocessor.py`): Raw ISOT `True.csv` articles contain `(Reuters)` datelines in `81.89%` of real test articles vs. `0.00%` of fake articles. `clean_article_text()` strips wire-service dateline prefixes (`WASHINGTON (Reuters) -`, `BRUSSELS —`, `(AP)`, `(AFP)`) before both training and inference so the model learns actual writing style rather than publisher tags.
  * **Primary Model (`DistilBERT`)**: Fine-tunes `distilbert-base-uncased` (`max_length=512`, `batch_size=16`, `epochs=3`, `lr=2e-5`, `bfloat16`) on combined `title + text`.
  * **Classical Baseline (`XGBoost`)**: Combines 50,000-d TF-IDF (`ngram_range=(1, 2)`) with 4-d spaCy NER entity counts (`ner_count`, `person_count`, `org_count`, `gpe_count`) across Logistic Regression, Multinomial Naive Bayes, Random Forest, and XGBoost.
  * **Evaluation (254 unseen test articles, dateline-stripped)**:
    * **DistilBERT**: `93.31%` accuracy, `0.9331` macro/weighted F1, confusion matrix `[[118, 9], [8, 119]]`.
    * **XGBoost Baseline**: `92.52%` accuracy, `0.9252` macro/weighted F1, confusion matrix `[[119, 8], [11, 116]]`.

---

### 3. NewsCLIPpings Image–Text Consistency Detector (`models/newsclippings_image_text/`)
* **Dataset**: **`NewsCLIPpings` (`merged_balanced`)** — **71,072** train / **7,024** val / **7,264** test image–caption pairs paired with precomputed **`openai/clip-vit-base-patch32`** embeddings (`512-d` image + `512-d` text).
* **Why we use it**: Detects **out-of-context or repurposed news imagery** — for example, when an authentic historical photograph is attached to an unrelated breaking-news caption (`MISMATCHED` vs. `MATCHED`).
* **How it works**:
  * **Feature Fusion (`concat+diff+cos`, 1,537-d)** (`newsclippings_preprocessor.py`): Given L2-normalized CLIP image embedding $v_{\text{img}} \in \mathbb{R}^{512}$ and text embedding $v_{\text{txt}} \in \mathbb{R}^{512}$, constructs $[v_{\text{img}};\; v_{\text{txt}};\; |v_{\text{img}} - v_{\text{txt}}|;\; \cos(v_{\text{img}}, v_{\text{txt}})]$.
  * **Primary Model (`PyTorch MLP`)**: A `[512, 256]` MLP trained on the 1,537-d fused vectors (`batch_size=256`).
  * **Classical Baseline (`XGBoost`)**: Trained alongside Logistic Regression and Random Forest on the same fused features.
  * **Evaluation (7,264 test pairs)**:
    * **PyTorch MLP**: `72.16%` accuracy, `0.7214` F1, **`0.7991` ROC-AUC**.
    * **XGBoost Baseline**: `69.62%` accuracy, `0.6959` F1, `0.7776` ROC-AUC.
  * **Inference** (`newsclippings_inference.py`): Accepts `(image_path, caption)`, extracts live CLIP embeddings for the image and caption, and outputs `MATCHED` (`0`) or `MISMATCHED` (`1`) with cosine similarity.

---

### 4. Unified Multimodal Router (`POST /api/ml/unified-route`)
* [unified_router.py](backend/ml/inference/unified_router.py) routes text submissions to **FakeNewsNet DistilBERT + FEVER DeBERTa** and image+caption submissions to **NewsCLIPpings MLP**, returning a structured JSON report with calibration disclaimers.
* For full training commands, hardware tuning, and Python SDK examples, see [backend/ML_TRAINING_README.md](backend/ML_TRAINING_README.md).

---

## 🛠️ Architecture & Tech Stack

```
                              ┌──────────────────────────────────────────────┐
                              │            LokLens AI Ingestion              │
                              │  (Text Claim, Image Screenshot, or Photo)    │
                              └──────────────────────┬───────────────────────┘
                                                     │
                        ┌────────────────────────────┴───────────────────────────┐
                        ▼                                                        ▼
         ┌─────────────────────────────┐                         ┌─────────────────────────────┐
         │     Digital Image Forensics │                         │   OCR & NLP Claim Parsing   │
         │ - ELA Compression Anomaly   │                         │ - CLAHE / Deskew Multi-pass │
         │ - 2D FFT Frequency (GAN/AI) │                         │ - Format Detection (Meme/UI)│
         │ - Noise Variance Analysis   │                         │ - Atomic Claim Extraction   │
         │ - ORB Copy-Move Detector    │                         │ - English + Hindi Devanagari│
         │ - 64-bit DCT pHash Index    │                         └──────────────┬──────────────┘
         └──────────────┬──────────────┘                                        │
                        │                                                       ▼
                        │                                        ┌─────────────────────────────┐
                        │                                        │  Multi-Platform Web Search  │
                        │                                        │ - Fact-check Registry Query │
                        │                                        │ - X / Twitter Scoped Search │
                        │                                        │ - Instagram & Reddit Scrape │
                        │                                        │ - Mainstream News Crawl     │
                        │                                        └──────────────┬──────────────┘
                        │                                                       │
                        └────────────────────────────┬──────────────────────────┘
                                                     │
                                                     ▼
                                      ┌─────────────────────────────┐
                                      │ Stance & Consistency Engine │
                                      │ - Evidence Stance Classifier│
                                      │ - Multimodal Consistency    │
                                      │ - Source Authority Scorer   │
                                      │ - LLM Executive Synthesis   │
                                      └──────────────┬──────────────┘
                                                     │
                                                     ▼
                                      ┌─────────────────────────────┐
                                      │ React / Vite Intelligence UI│
                                      │ - Social Evidence Feed      │
                                      │ - Forensics Gauges & Badges │
                                      └─────────────────────────────┘
```

| Component | Technologies |
|---|---|
| **Backend API** | FastAPI, Uvicorn, Python 3.11+, SQLAlchemy, aiosqlite |
| **Search & Crawling** | DuckDuckGo (`ddgs`), Trafilatura, BeautifulSoup4, httpx |
| **Social Media** | OpenGraph & Twitter Cards Parser, Social Media Content Extractor |
| **NLP & Extraction** | spaCy (`en_core_web_sm`), NLTK, BM25, scikit-learn, Devanagari Unicode |
| **Image Forensics** | OpenCV (`cv2`), Pillow (`PIL`), NumPy, imagehash, piexif |
| **OCR Engines** | PaddleOCR (Multilingual), Tesseract OCR (`pytesseract`) |
| **LLM Synthesis** | Ollama (`llama3.1`), OpenAI API (`gpt-4o-mini` fallback) with 15s timeout |
| **Frontend UI** | React 19, TypeScript, Vite, Tailwind CSS, Lucide Icons |

---

## 🚦 Quick Start

### 1. Setup Backend Environment

```bash
cd backend
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Local Reasoning Model (Ollama)

```bash
# In a separate terminal
ollama run llama3.1
```

### 3. Launch the Backend Server

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
Interactive Swagger API documentation: **http://127.0.0.1:8000/docs**

### 4. Launch the Frontend UI

```bash
cd frontend
npm install
npm run dev
```
Open **http://localhost:5173** in your browser.

---

## 📄 License & Attribution

LokLens AI is open-source under the MIT License. Designed for transparent, accountable verification of digital media and political claims.
