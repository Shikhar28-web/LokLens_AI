# TruthLens — Architecture & Implementation Plan

> **Multimodal Evidence-Based News & Image Verification**  
> Core principle: No generative LLM. Every verdict is explainable from structured evidence.

---

## 0. Confirmed Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Search Provider** | DuckDuckGo (free, no API key) | Pluggable abstraction — SerpAPI/Brave/Bing can be swapped in via env var |
| **Deployment** | Local Windows dev → Docker for production | Local first for rapid iteration; Docker handles Tesseract/spaCy portably |
| **ChromaDB** | Included from Phase 1 as optional secondary layer | Kept behind a feature flag; TF-IDF + BM25 remain the primary path |
| **Starting Point** | Fully fresh codebase | No dependency on any prior exploratory notebooks or JSON files |
| **Database** | SQLite for dev, PostgreSQL-compatible schema for prod | Alembic migrations cover both |

---

## 1. Final Architecture

```
┌──────────────────────────────────────────────────────┐
│                    React + Vite Frontend              │
│  Home │ Upload │ Progress │ Results │ Evidence Graph  │
└──────────────┬───────────────────────────────────────┘
               │ HTTP/REST (JSON)
┌──────────────▼───────────────────────────────────────┐
│                  FastAPI Backend                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │              API Layer (/api/*)                 │ │
│  └──┬────────┬─────────┬──────────┬───────────────┘ │
│     │        │         │          │                  │
│  ┌──▼──┐ ┌──▼──┐  ┌───▼───┐  ┌───▼───┐             │
│  │ NLP │ │Search│  │Image  │  │Verdict│             │
│  │Pipe │ │+Retr.│  │Foren. │  │Engine │             │
│  └──┬──┘ └──┬──┘  └───┬───┘  └───┬───┘             │
│     └────────┴─────────┴──────────┘                 │
│              ┌──────────────────┐                    │
│              │  SQLite / SQLAlch│                    │
│              └──────────────────┘                    │
└──────────────────────────────────────────────────────┘
```

### Key Separation of Concerns

| Layer | Responsibility |
|---|---|
| **Preprocessing** | Normalize text/image, OCR, validate input |
| **NLP** | Tokenize, NER, claim extraction, query generation |
| **Search/Retrieval** | Provider-agnostic web search, TF-IDF, BM25 |
| **Evidence** | Extract passages, score support/contradiction |
| **Image Forensics** | EXIF, ELA, pHash, frequency analysis |
| **Verdict Engine** | Deterministic rule-based scoring |
| **Reporting** | Template-based human-readable explanation |

---

## 2. Folder Structure

```
d:\Projects\Political_lens\
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI entry point
│   │   ├── config.py                  # Env-based settings (pydantic-settings)
│   │   ├── database.py                # SQLAlchemy engine + session
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── submissions.py         # POST/GET /api/submissions/*
│   │   │   ├── images.py              # POST /api/images/analyze
│   │   │   ├── text.py                # POST /api/text/analyze
│   │   │   └── search.py              # POST /api/search
│   │   │
│   │   ├── models/                    # SQLAlchemy ORM models
│   │   │   ├── submission.py
│   │   │   ├── claim.py
│   │   │   ├── document.py
│   │   │   ├── evidence.py
│   │   │   ├── image_forensics.py
│   │   │   ├── source.py
│   │   │   ├── verdict.py
│   │   │   └── graph_edge.py
│   │   │
│   │   ├── schemas/                   # Pydantic request/response schemas
│   │   │   ├── submission.py
│   │   │   ├── claim.py
│   │   │   ├── evidence.py
│   │   │   ├── image.py
│   │   │   ├── report.py
│   │   │   └── verdict.py
│   │   │
│   │   ├── services/
│   │   │   ├── pipeline.py            # Orchestrates the full verification pipeline
│   │   │   │
│   │   │   ├── preprocessing/
│   │   │   │   ├── text_preprocessor.py
│   │   │   │   └── image_preprocessor.py
│   │   │   │
│   │   │   ├── nlp/
│   │   │   │   ├── tokenizer.py
│   │   │   │   ├── ner.py             # spaCy-based NER
│   │   │   │   ├── claim_extractor.py
│   │   │   │   └── query_generator.py
│   │   │   │
│   │   │   ├── search/
│   │   │   │   ├── base_provider.py   # Abstract SearchProvider
│   │   │   │   ├── serpapi_provider.py
│   │   │   │   ├── ddg_provider.py    # DuckDuckGo (no-key fallback)
│   │   │   │   └── cache.py           # Disk-based result cache
│   │   │   │
│   │   │   ├── retrieval/
│   │   │   │   ├── tfidf_retriever.py
│   │   │   │   ├── bm25_retriever.py
│   │   │   │   └── ranker.py          # Combines both + deduplicates
│   │   │   │
│   │   │   ├── evidence/
│   │   │   │   ├── extractor.py       # Sentence-level evidence extraction
│   │   │   │   ├── support_scorer.py
│   │   │   │   └── contradiction_detector.py
│   │   │   │
│   │   │   ├── source_analysis/
│   │   │   │   ├── quality_scorer.py
│   │   │   │   └── dedup_detector.py  # Independent source corroboration
│   │   │   │
│   │   │   ├── image_forensics/
│   │   │   │   ├── metadata_extractor.py
│   │   │   │   ├── ela_analyzer.py    # Error Level Analysis
│   │   │   │   ├── frequency_analyzer.py
│   │   │   │   ├── noise_analyzer.py
│   │   │   │   ├── copy_move_detector.py
│   │   │   │   ├── phash_matcher.py
│   │   │   │   └── ai_score_estimator.py
│   │   │   │
│   │   │   ├── ocr/
│   │   │   │   └── ocr_engine.py      # Tesseract wrapper
│   │   │   │
│   │   │   ├── verification/
│   │   │   │   ├── verdict_engine.py  # Deterministic scoring + rules
│   │   │   │   └── multimodal_consistency.py
│   │   │   │
│   │   │   └── reporting/
│   │   │       ├── template_engine.py # Template-based NL explanation
│   │   │       └── graph_builder.py   # Evidence graph construction
│   │   │
│   │   ├── repositories/              # Data access layer (no business logic)
│   │   │   ├── submission_repo.py
│   │   │   ├── claim_repo.py
│   │   │   ├── evidence_repo.py
│   │   │   └── document_repo.py
│   │   │
│   │   └── utils/
│   │       ├── security.py            # File validation, SSRF guard
│   │       ├── hash_utils.py
│   │       ├── date_utils.py
│   │       └── logging_config.py
│   │
│   ├── database/
│   │   ├── migrations/                # Alembic migrations
│   │   └── seed/                      # Test seed data
│   │
│   └── tests/
│       ├── unit/
│       │   ├── test_tokenizer.py
│       │   ├── test_claim_extractor.py
│       │   ├── test_query_generator.py
│       │   ├── test_tfidf.py
│       │   ├── test_bm25.py
│       │   ├── test_contradiction.py
│       │   ├── test_source_scorer.py
│       │   ├── test_phash.py
│       │   ├── test_ela.py
│       │   └── test_verdict_engine.py
│       └── integration/
│           ├── test_text_pipeline.py
│           ├── test_image_pipeline.py
│           └── test_multimodal_pipeline.py
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── UploadZone.tsx
│   │   │   ├── ProgressTracker.tsx
│   │   │   ├── VerdictBadge.tsx
│   │   │   ├── EvidenceCard.tsx
│   │   │   ├── SourceList.tsx
│   │   │   ├── ImageForensicsPanel.tsx
│   │   │   ├── TimelineView.tsx
│   │   │   └── EvidenceGraph.tsx
│   │   ├── pages/
│   │   │   ├── Home.tsx
│   │   │   ├── Upload.tsx
│   │   │   ├── Progress.tsx
│   │   │   ├── Results.tsx
│   │   │   ├── EvidencePage.tsx
│   │   │   ├── SourceDetail.tsx
│   │   │   ├── ImageForensicPage.tsx
│   │   │   ├── TimelinePage.tsx
│   │   │   └── GraphPage.tsx
│   │   ├── services/
│   │   │   └── api.ts
│   │   ├── hooks/
│   │   │   └── useSubmission.ts
│   │   ├── types/
│   │   │   └── index.ts
│   │   └── utils/
│   │       └── formatters.ts
│   ├── index.html
│   ├── vite.config.ts
│   └── tailwind.config.ts
│
├── data/
│   ├── raw/                           # Fakeddit dataset (existing)
│   ├── processed/                     # Normalized benchmark data
│   ├── cache/                         # Cached search results (JSON)
│   └── benchmark/
│       ├── ground_truth.json
│       └── eval_results/
│
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── forensics_methods.md
│
├── .env.example
├── alembic.ini
└── README.md
```

---

## 3. Database Schema

### `users`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| created_at | TIMESTAMP | |
| ip_hash | VARCHAR(64) | Rate limiting |

### `submissions`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| user_id | UUID FK → users | nullable |
| input_type | ENUM | `text`, `image`, `multimodal` |
| raw_text | TEXT | nullable |
| image_path | VARCHAR(512) | nullable (sanitized) |
| status | ENUM | `queued`, `processing`, `complete`, `error` |
| created_at | TIMESTAMP | |
| completed_at | TIMESTAMP | nullable |

### `claims`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| submission_id | UUID FK → submissions | |
| claim_text | TEXT | |
| subject | VARCHAR(512) | |
| predicate | VARCHAR(512) | |
| object | VARCHAR(512) | |
| entities_json | JSON | `{persons, orgs, locations, dates, numbers}` |
| keywords_json | JSON | |
| search_queries_json | JSON | generated queries |
| created_at | TIMESTAMP | |

### `sources`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| url | TEXT UNIQUE | |
| domain | VARCHAR(255) | |
| source_type | ENUM | `government`, `news`, `academic`, `blog`, `social`, `unknown` |
| authority_score | FLOAT | 0–1 |
| quality_score | FLOAT | composite |
| quality_components_json | JSON | individual factor scores |
| first_seen | TIMESTAMP | |

### `documents`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| source_id | UUID FK → sources | |
| title | TEXT | |
| content | TEXT | cleaned article body |
| content_hash | VARCHAR(64) | SHA-256 for dedup |
| publication_date | DATE | nullable |
| author | VARCHAR(512) | nullable |
| language | VARCHAR(10) | |
| retrieved_at | TIMESTAMP | |

### `search_results`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| claim_id | UUID FK → claims | |
| query | TEXT | |
| provider | VARCHAR(64) | |
| result_url | TEXT | |
| title | TEXT | |
| snippet | TEXT | |
| rank | INT | |
| retrieved_at | TIMESTAMP | |

### `evidence`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| claim_id | UUID FK → claims | |
| document_id | UUID FK → documents | |
| evidence_text | TEXT | exact passage from document |
| support_score | FLOAT | 0–1 |
| contradiction_score | FLOAT | 0–1 |
| relevance_score | FLOAT | 0–1 |
| retrieval_method | VARCHAR(32) | `tfidf`, `bm25` |
| contradiction_indicators_json | JSON | negation/entity conflict details |
| created_at | TIMESTAMP | |

### `images`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| submission_id | UUID FK → submissions | |
| original_filename | VARCHAR(255) | |
| stored_path | VARCHAR(512) | |
| mime_type | VARCHAR(64) | |
| file_size_bytes | INT | |
| width | INT | |
| height | INT | |
| sha256_hash | VARCHAR(64) | |
| phash | VARCHAR(64) | perceptual hash |
| dhash | VARCHAR(64) | |
| ahash | VARCHAR(64) | |
| ocr_text | TEXT | nullable |
| created_at | TIMESTAMP | |

### `image_forensics`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| image_id | UUID FK → images | |
| exif_json | JSON | all extracted EXIF fields |
| ela_score | FLOAT | 0–1 |
| noise_anomaly | FLOAT | |
| edge_anomaly | FLOAT | |
| frequency_anomaly | FLOAT | |
| compression_anomaly | FLOAT | |
| metadata_anomaly | FLOAT | |
| copy_move_score | FLOAT | |
| ai_likelihood_score | FLOAT | |
| ai_likelihood_label | ENUM | `likely_authentic`…`likely_ai_generated` |
| forensic_features_json | JSON | all raw feature vectors |

### `verdicts`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| submission_id | UUID FK → submissions | |
| claim_verdict | ENUM | 7-level scale |
| claim_confidence | FLOAT | |
| image_verdict | ENUM | 7-level scale |
| image_confidence | FLOAT | |
| overall_status | VARCHAR(64) | |
| support_score | FLOAT | |
| contradiction_score | FLOAT | |
| source_quality_score | FLOAT | |
| independent_source_count | INT | |
| duplicate_source_group_count | INT | |
| multimodal_consistency_score | FLOAT | nullable |
| explanation_json | JSON | template-generated NL explanation |
| score_components_json | JSON | all individual inputs to verdict |
| created_at | TIMESTAMP | |

### `evidence_graph_edges`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| submission_id | UUID FK → submissions | |
| from_node_type | VARCHAR(32) | `claim`, `source`, `document`, `image`, `entity` |
| from_node_id | UUID | |
| to_node_type | VARCHAR(32) | |
| to_node_id | UUID | |
| relationship | ENUM | `SUPPORTS`, `CONTRADICTS`, `CONTEXT`, `RELATED_IMAGE`, `MENTIONS` |
| weight | FLOAT | |

### `entities`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| name | VARCHAR(512) | |
| entity_type | ENUM | `PERSON`, `ORG`, `LOCATION`, `DATE`, `NUMBER`, `EVENT` |
| normalized_name | VARCHAR(512) | |

### `timelines`
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| submission_id | UUID FK → submissions | |
| event_date | DATE | nullable |
| event_description | TEXT | |
| source_id | UUID FK → sources | nullable |
| document_id | UUID FK → documents | nullable |

---

## 4. Dependency List

### Backend (`backend/requirements.txt`)

```
# Web framework
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9        # file uploads

# Data validation
pydantic==2.7.0
pydantic-settings==2.2.1

# Database
sqlalchemy==2.0.30
alembic==1.13.1
aiosqlite==0.20.0              # async SQLite

# NLP
nltk==3.8.1
spacy==3.7.4                   # en_core_web_sm model
scikit-learn==1.5.0            # TF-IDF, cosine similarity
rank-bm25==0.2.2

# Image
opencv-python-headless==4.9.0.80
Pillow==10.3.0
numpy==1.26.4
imagehash==4.3.1               # pHash / dHash / aHash
piexif==1.1.3                  # EXIF read/write

# OCR
pytesseract==0.3.10            # wrapper (requires Tesseract binary)

# Web / scraping
httpx==0.27.0                  # async HTTP
beautifulsoup4==4.12.3
lxml==5.2.1
readability-lxml==0.8.1        # article extraction
trafilatura==1.9.0             # alternative article extractor
tldextract==5.1.2

# Security
python-magic==0.4.27
bleach==6.1.0

# Caching / utils
diskcache==5.6.3
python-dotenv==1.0.1

# Testing
pytest==8.2.0
pytest-asyncio==0.23.6
httpx                          # for TestClient

# Evaluation
pandas==2.2.2
```

### Frontend (`frontend/package.json` key deps)
```
react 18, react-dom 18
vite 5
typescript 5
tailwindcss 3
react-router-dom 6
react-dropzone              # drag-and-drop upload
recharts                    # evidence score charts
d3                          # evidence graph visualization
lucide-react                # icons
axios                       # API client
```

---

## 5. API Design

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/submissions` | Create submission (upload image + text) |
| `POST` | `/api/submissions/{id}/analyze` | Trigger analysis pipeline |
| `GET` | `/api/submissions/{id}` | Get submission status |
| `GET` | `/api/submissions/{id}/claims` | List extracted claims |
| `GET` | `/api/submissions/{id}/evidence` | List evidence items |
| `GET` | `/api/submissions/{id}/sources` | List sources |
| `GET` | `/api/submissions/{id}/report` | Full structured report |
| `GET` | `/api/submissions/{id}/timeline` | Timeline of events |
| `GET` | `/api/submissions/{id}/graph` | Evidence graph (nodes + edges) |
| `POST` | `/api/images/analyze` | Image-only forensic analysis |
| `POST` | `/api/text/analyze` | Text-only quick claim check |
| `POST` | `/api/search` | Debug: run a manual search query |

### Request — `POST /api/submissions`

```json
{
  "text": "string | null",
  "image_url": "string | null"
}
```
+ `multipart/form-data` with optional `image` file field.

### Response — `GET /api/submissions/{id}/report`

```json
{
  "submission_id": "uuid",
  "overall_status": "CONTRADICTED",
  "claim_verdict": "CONTRADICTED",
  "claim_confidence": 0.82,
  "image_verdict": "POSSIBLY_MANIPULATED",
  "image_confidence": 0.61,
  "summary": "Template-generated explanation string",
  "claims": [ { "claim_id": "...", "text": "...", "verdict": "...", ... } ],
  "supporting_evidence": [ { "evidence_id": "...", "text": "...", "source_url": "...", "score": 0.0 } ],
  "contradicting_evidence": [ { ... } ],
  "sources": [ { "url": "...", "domain": "...", "source_type": "...", "quality_score": 0.0 } ],
  "image_analysis": {
    "ela_score": 0.72,
    "noise_anomaly": 0.64,
    "compression_anomaly": 0.51,
    "ai_likelihood_label": "POSSIBLY_AI_GENERATED",
    "ocr_text": "..."
  },
  "independent_source_count": 3,
  "duplicate_source_group_count": 2,
  "timeline": [ { "date": "...", "event": "...", "source_url": "..." } ],
  "limitations": [ "OCR confidence low", "Only 2 sources retrieved" ],
  "score_components": { ... }
}
```

---

## 6. Data-Flow Diagram

```
USER INPUT (text / image / both)
         │
         ▼
┌─────────────────────┐
│  Input Validation   │  ← MIME check, size limit, SSRF guard
└────────┬────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
TEXT PATH   IMAGE PATH
    │         │
    │    ┌────▼──────────────────────────────┐
    │    │  Image Preprocessor               │
    │    │  → resize / normalize             │
    │    │  → EXIF extraction                │
    │    │  → pHash / dHash / aHash          │
    │    │  → OCR (→ feeds TEXT PATH)        │
    │    └────┬──────────────────────────────┘
    │         │
    │    ┌────▼──────────────────────────────┐
    │    │  Image Forensics                  │
    │    │  → ELA → noise → frequency        │
    │    │  → copy-move → AI score           │
    │    │  → pHash web matching             │
    │    └────┬──────────────────────────────┘
    │         │
    ▼         ▼
┌─────────────────────────────────────────┐
│  Text Preprocessor                      │
│  → Unicode norm → tokenize → stopwords  │
│  → lemmatize → NER → sentiment          │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Claim Extractor                        │
│  → split paragraphs into atomic claims  │
│  → build Claim objects                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Query Generator                        │
│  → 4–6 queries per claim                │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Search Provider (pluggable)            │
│  → check cache first                    │
│  → fetch results + store URLs           │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Document Extractor                     │
│  → fetch URL → article extraction       │
│  → clean / hash / store                 │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Retriever (TF-IDF + BM25)              │
│  → score claim vs all documents         │
│  → top-k candidates per claim           │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Evidence Extractor                     │
│  → sentence-level similarity            │
│  → keyword + entity + date overlap      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Support / Contradiction Engine         │
│  → negation detection                   │
│  → entity conflict                      │
│  → numeric / date conflict              │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Source Quality Scorer                  │
│  → domain authority                     │
│  → source type                          │
│  → recency / corroboration              │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Independent Source Corroboration       │
│  → text similarity dedup                │
│  → origin detection                     │
└──────────────────┬──────────────────────┘
                   │
            ┌──────┴──────┐
            │             │
            ▼             ▼
    CLAIM SCORES    IMAGE SCORES
            │             │
            ▼             ▼
┌─────────────────────────────────────────┐
│  Multimodal Consistency                 │
│  → OCR text ↔ claim                     │
│  → image date ↔ claim date              │
│  → pHash match ↔ claimed source         │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Deterministic Verdict Engine           │
│  → weighted rule table                  │
│  → claim verdict (7 levels)             │
│  → image verdict (7 levels)             │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Evidence Graph Builder                 │
│  → edges: SUPPORTS / CONTRADICTS etc.   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Template Report Generator              │
│  → deterministic NL explanation         │
│  → structured JSON report               │
└──────────────────┬──────────────────────┘
                   │
                   ▼
            API Response → Frontend
```

---

## 7. Verdict Engine Logic (Deterministic Rules)

### Claim Verdict

| Condition | Verdict |
|---|---|
| support_score ≥ 0.75 AND contradiction_score < 0.2 AND independent_sources ≥ 2 | `SUPPORTED` |
| support_score ≥ 0.55 AND contradiction_score < 0.3 | `LIKELY_SUPPORTED` |
| support_score ≥ 0.35 AND contradiction_score < 0.4 | `PARTIALLY_SUPPORTED` |
| support_score ≥ 0.3 AND sources confirm but context differs | `MISLEADING_CONTEXT` |
| contradiction_score ≥ 0.65 AND independent_sources ≥ 2 | `CONTRADICTED` |
| contradiction_score ≥ 0.4 | `UNSUPPORTED` |
| total_evidence_count < 2 OR retrieval_coverage < 0.2 | `INSUFFICIENT_EVIDENCE` |

### Image Verdict

| Condition | Verdict |
|---|---|
| ai_score < 0.2 AND ela < 0.2 AND noise < 0.2 | `LIKELY_AUTHENTIC` |
| ai_score < 0.35 AND manipulation_indicators_max < 0.35 | `POSSIBLY_AUTHENTIC` |
| one indicator ≥ 0.6 | `POSSIBLY_MANIPULATED` |
| two+ indicators ≥ 0.6 OR ela ≥ 0.75 | `LIKELY_MANIPULATED` |
| ai_score ≥ 0.55 AND frequency_anomaly high | `POSSIBLY_AI_GENERATED` |
| ai_score ≥ 0.75 | `LIKELY_AI_GENERATED` |
| cannot determine | `INCONCLUSIVE` |

---

## 8. Source Quality Scoring (Transparent Components)

```python
quality_score = (
    0.30 * domain_authority_score +   # government/academic=1.0, blog=0.2
    0.20 * source_type_score +         # news=0.7, social=0.1
    0.15 * recency_score +             # exponential decay on date
    0.20 * directness_score +          # primary vs secondary source
    0.15 * corroboration_score         # appears in multiple independent docs
)
```
All individual components stored separately in `quality_components_json`.

---

## 9. Development Roadmap

| Phase | Focus | Est. Complexity |
|---|---|---|
| **1** | Project skeleton: FastAPI + React + DB schema + Alembic | Medium |
| **2** | NLP pipeline: tokenizer, NER, claim extractor, query gen | High |
| **3** | Search provider abstraction + cache + DuckDuckGo default | Medium |
| **4** | Document fetcher: article extraction, cleaning, hashing | Medium |
| **5** | TF-IDF retriever + cosine similarity | Low |
| **6** | BM25 retriever + ranker | Low |
| **7** | Evidence extraction (sentence-level) | Medium |
| **8** | Support/contradiction engine (negation, entity conflict) | High |
| **9** | Source quality scoring + independent source corroboration | Medium |
| **10** | Deterministic verdict engine + template report | High |
| **11** | Image preprocessing: EXIF, hashes, metadata | Medium |
| **12** | Image forensics: ELA, frequency, noise, copy-move | High |
| **13** | OCR integration (Tesseract) | Low |
| **14** | pHash web matching + old image detection | Medium |
| **15** | Multimodal consistency analysis | Medium |
| **16** | Evidence graph (DB edges + API) | Medium |
| **17** | Frontend: all pages + evidence graph viz | High |
| **18** | Security hardening (rate limit, validation, SSRF) | Medium |
| **19** | Tests: unit + integration | High |
| **20** | Benchmark: Fakeddit dataset eval, metrics dashboard | Medium |

---

## 10. Testing Strategy

### Unit Tests (pytest)
| Module | What to test |
|---|---|
| `tokenizer` | Unicode edge cases, empty input, special chars |
| `claim_extractor` | Atomic claim splitting, entity preservation |
| `query_generator` | Query diversity, no stopwords, entity inclusion |
| `tfidf_retriever` | Correct top-k ranking on fixture corpus |
| `bm25_retriever` | Correct scoring on fixture corpus |
| `contradiction_detector` | Negation patterns, numeric conflicts |
| `source_scorer` | Correct weight application, component storage |
| `phash` | Known similar/dissimilar image pairs |
| `ela_analyzer` | Known authentic vs manipulated fixture images |
| `verdict_engine` | All 7 verdict levels hit correct branches |

### Integration Tests
| Scenario | What to verify |
|---|---|
| Text-only pipeline | End-to-end: text → claim → search → verdict → report JSON |
| Image-only pipeline | End-to-end: image → forensics → verdict |
| Multimodal pipeline | Both inputs → combined consistency → combined verdict |

### Benchmark (Phase 20)
Reuse existing `Fakeddit datasetv2.0/` + `stage1_extracted_features.json` for ground-truth labels.
Track: accuracy, precision, recall, F1, FPR, FNR, retrieval precision, evidence coverage.

---

## Open Questions

> [!IMPORTANT]
> **Search Provider**: Which search API do you want to use?
> - DuckDuckGo Instant Answers (free, no key, limited results)
> - SerpAPI (paid, key required, full Google results)
> - Bing Search API (Azure key)
> - Brave Search API (free tier available)
>
> The abstraction will support all — just clarify the default for Phase 1.

> [!IMPORTANT]
> **Deployment Target**: Is the initial deployment local-only (Windows dev machine) or containerized (Docker)?
> This affects how Tesseract OCR and spaCy models are installed.

> [!NOTE]
> **Existing Notebook**: `Political_Lens_NLP_Pipeline.ipynb` appears to have tokenizer and sentiment work. Should I extract this into the backend `nlp/` modules as the starting point for Phase 2, or start fresh with a cleaner spaCy-based implementation?

> [!NOTE]
> **ChromaDB**: Included as optional secondary retrieval layer (Phase 6+). Should it be enabled from the start or added only if TF-IDF + BM25 prove insufficient?
