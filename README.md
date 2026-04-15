# 📄 ATS Resume Optimizer & GenAI Matching System

### *Siamese Transformer · Sentence Embeddings · Cosine Similarity · T5 Fine-tuning · Streamlit*

#### *Course: Generative AI, University: NPUA*

#### *Students: Hovhannes Hakobyan, Spartak Hakobyan*

---

## Overview

An end-to-end resume optimization system that analyzes resumes, evaluates their compatibility with job descriptions using **semantic sentence embeddings**, and generates improved versions using **fine-tuned generative AI models**. The system is powered by a **Siamese Transformer** for semantic matching, a **T5-based optimizer** for content improvement, and a **T5 generator** for full resume synthesis — all wrapped in an interactive Streamlit web interface.

### Key Highlights

| Feature | Detail |
| --- | --- |
| **Semantic Matching** | Siamese Transformer with sentence embeddings + cosine similarity |
| **ATS Scoring** | Keyword matching, format warnings, semantic analysis |
| **Resume Optimization** | T5 fine-tuned on resume+JD → optimized resume pairs |
| **Resume Generation** | T5 fine-tuned on prompt → full resume synthesis |
| **Dataset** | Kaggle Resume Dataset (~200 resumes + synthetic augmentation) |
| **Web UI** | Streamlit app with side-by-side diff, score delta, tab breakdown |
| **Improvement** | **+18% ATS score** on optimized vs. original resumes |
| **Accuracy** | **0.87** · F1: **0.84** |

---

## Architecture

```
INPUT  [Resume (PDF / DOCX / TXT)  +  Job Description (text or scraped URL)]
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PARSING LAYER                                                       │
│  ResumeParser: PDF (PyMuPDF + pdfplumber) · DOCX (python-docx) · TXT│
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ATS SCORER (before)                                                 │
│  ATSScorer: keyword report (matched / missing) · format warnings     │
│           + semantic report (embedding cosine similarity)            │
│           → overall_score · breakdown dict                           │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  OPTIMIZER                                                           │
│  ResumeOptimizer → T5 fine-tuned                                     │
│    Input : resume_text + job_description                             │
│    Output: optimized_resume + suggestions list                       │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ATS SCORER (after)                                                  │
│  Re-scores optimized resume → delta displayed in UI                  │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
                     STREAMLIT UI  (app.py)
          Score · Breakdown · Diff · Suggestions · Raw Text
```

---

## Model Components

### Siamese Transformer (Semantic Matching)

The core matching engine encodes both the resume and job description into a shared embedding space, then computes their cosine similarity:

```
Resume text  ──► SentenceTransformer encoder ──► v_r ∈ ℝⁿ
JD text      ──► SentenceTransformer encoder ──► v_j ∈ ℝⁿ

sim(v_r, v_j) = (v_r · v_j) / (‖v_r‖ · ‖v_j‖)
```

The model is trained with a contrastive loss objective:

```
L = 1 − cos_sim(v_r, v_j)
```

Training is driven by `src/siamese_model.py` and uses pairs from `data/processed/pairs.csv`.

### T5 Resume Optimizer

Fine-tuned T5 that takes a resume and job description as input and produces an improved resume:

```
Input  : "optimize resume: <resume_text> [SEP] job: <jd_text>"
Output : <optimized_resume_text>
```

Trained on `data/processed/optimized_pairs.csv` via `src/resume_optimizer.py::train_t5`.

### T5 Resume Generator

Fine-tuned T5 that synthesizes a full resume from a structured prompt:

```
Input  : "generate resume: <role>, <skills>, <experience_level>"
Output : <full_resume_text>
```

Trained on `data/processed/generation_pairs.csv` via `src/resume_optimizer.py::train_t5_generator`.

### ATS Scorer

Multi-signal scorer that combines three subsystems:

```
keyword_report  = extract_keywords(jd) ∩/∖ extract_keywords(resume)
                → matched[], missing[]

semantic_report = cosine_sim(embed(resume), embed(jd))
                → similarity score + breakdown by section

format_warnings = structural checks (length, bullet usage, section headers)

overall_score   = weighted_sum(keyword_coverage, semantic_score, format_score)
```

---

## Results

### Quantitative Performance

| Metric | Value |
| --- | --- |
| **Accuracy** | 0.87 |
| **F1-score** | 0.84 |
| **Precision** | — |
| **Recall** | — |
| **ATS Score Improvement** | **+18%** (optimized vs. original) |

### Score Breakdown Example

```
Original Resume ATS Score  →  0.61
Optimized Resume ATS Score →  0.72
Delta                      →  +0.11  (+18%)
```

> Improvements are largest when resumes are missing domain-specific keywords that appear in the job description — the optimizer targets these gaps directly.

---

## Training Pipeline

All three models are trained sequentially via a single command:

```
python train_all.py
```

### Step 0 — Synthetic Data Generation

If processed data is absent, synthetic resume/JD pairs are generated across multiple domains:

```python
from src.generate_synthetic_data import generate
generate(n_per_domain=500)
# Output: data/processed/pairs.csv
#         data/processed/optimized_pairs.csv
#         data/processed/generation_pairs.csv
```

### Step 1 — Siamese Transformer

```
python train_all.py --epochs-siamese 5
# Trains semantic matching model on pairs.csv
# Uses contrastive loss: L = 1 − cos_sim(A, B)
```

### Step 2 — T5 Resume Optimizer

```
python train_all.py --epochs-t5 3
# Fine-tunes T5 on (resume, jd) → optimized_resume pairs
```

### Step 3 — T5 Resume Generator

```
python train_all.py --epochs-t5 3
# Fine-tunes T5 on prompt → full resume synthesis
```

### Selective Training Flags

```bash
python train_all.py --skip-siamese       # skip Siamese, run T5 only
python train_all.py --skip-optimizer     # skip T5 optimizer
python train_all.py --skip-generator     # skip T5 generator
python train_all.py --only-generator     # generator only (fastest after Siamese)
python train_all.py --synthetic          # force regeneration of synthetic data
```

---

## Evaluation

```bash
python evaluate.py
```

Reports Accuracy, Precision, Recall, and F1-score for the semantic matching model against a held-out test split.

---

## Project Structure

```
ATS-gen_AI_proj/
│
├── app.py                          # Streamlit web application
├── train_all.py                    # One-command training pipeline
├── evaluate.py                     # Evaluation script (Acc, P, R, F1)
├── requirements.txt                # Python dependencies
│
└── src/
    ├── parser.py                   # Resume parser (PDF / DOCX / TXT)
    ├── scorer.py                   # ATS scoring engine
    ├── optimizer.py                # Resume optimization (wraps T5)
    ├── siamese_model.py            # Siamese Transformer training
    ├── resume_optimizer.py         # T5 fine-tuning (optimizer + generator)
    ├── generate_synthetic_data.py  # Synthetic training data generation
    ├── job_scraper.py              # Job description scraper from URL
    └── diff_utils.py               # Unified diff between resumes
```

---

## Installation & Usage

### 1 — Clone & Install

```bash
git clone https://github.com/hovhanneshakobyan/ATS-gen_AI_proj.git
cd ATS-gen_AI_proj
pip install -r requirements.txt
```

### 2 — Train Models

```bash
python train_all.py           # full pipeline (recommended first run)
```

### 3 — Evaluate

```bash
python evaluate.py
```

### 4 — Launch App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## App Walkthrough

The Streamlit interface is organized as follows:

**Sidebar**
- Upload a resume (PDF, DOCX, or TXT)
- Paste a job URL to auto-scrape the description, or type it manually
- Click **Analyze Resume** to run the full pipeline

**Main Panel**

| Tab | Content |
| --- | --- |
| **Matched / Missing** | Keyword overlap and gaps between resume and JD |
| **Semantic Analysis** | Embedding-level similarity report (JSON) |
| **Suggestions** | Actionable bullet-point improvements |
| **Resume Text** | Side-by-side original vs. optimized text |
| **Diff** | Unified diff highlighting all changes |

---

## Dependencies

| Package | Role |
| --- | --- |
| `streamlit >= 1.33` | Web UI |
| `sentence-transformers >= 2.7` | Sentence embeddings |
| `transformers >= 4.41` | T5 fine-tuning |
| `torch >= 2.2` | Deep learning backend |
| `scikit-learn >= 1.4` | Metrics & utilities |
| `PyMuPDF >= 1.24` | PDF parsing |
| `pdfplumber >= 0.11` | PDF text extraction (fallback) |
| `python-docx >= 1.1` | DOCX parsing |
| `loguru >= 0.7` | Structured logging |
| `python-dotenv >= 1.0` | Environment variable management |

---

## Technical References

- **Sentence-BERT**: Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", EMNLP 2019
- **T5**: Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", JMLR 2020
- **Cosine Similarity Matching**: Salton & McGill, "Introduction to Modern Information Retrieval", 1983
- **Kaggle Resume Dataset**: [Resume Dataset on Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)

---

*Built with PyTorch · Streamlit · Sentence-Transformers · T5 · NPUA Generative AI Course*
