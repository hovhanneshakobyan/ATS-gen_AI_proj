# Resume2Vec — ATS Resume Optimizer

Generative AI project based on **Siamese Transformer Network + T5 Resume Optimizer**.  
Optimizes resumes for Applicant Tracking Systems (ATS).

---

## Architecture

```
Resume (PDF/DOCX/text)
        │
        ▼
[ Resume Parser ]
        │
        ├──────────────────────┐
        ▼                      ▼
[ Siamese Transformer ]   [ T5 Optimizer ]
  (semantic matching)     (text rewriting)
        │                      │
        ▼                      ▼
[ ATS Checker ] ◄─────── Optimized Resume
        │
        ▼
  ATS Score [0–100]
```

### Models
| Model | Purpose | Base |
|-------|---------|------|
| Siamese Transformer | Resume ↔ JD semantic matching | `all-MiniLM-L6-v2` |
| T5 Optimizer | Resume rewriting for ATS | `t5-base` |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Download datasets
Place in `data/raw/`:
- [Resume Dataset](https://www.kaggle.com/datasets/palaksood97/resume-dataset) → `data/raw/resumes.csv`
- [Job Descriptions](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) → `data/raw/job_descriptions.csv`

### 3. Prepare data & train models
```bash
python train_all.py
```
This runs:
- `src/data_prep.py`        → builds training pairs
- `src/siamese_model.py`    → trains Siamese Transformer
- `src/resume_optimizer.py` → fine-tunes T5

### 4. Launch web app
```bash
streamlit run app.py
```

---

## Project Structure

```
GEN AI proj/
├── app.py                  # Streamlit web UI
├── train_all.py            # One-command training
├── requirements.txt
├── data/
│   ├── raw/                # Place Kaggle CSVs here
│   └── processed/          # Generated training pairs
├── models/
│   ├── siamese_checkpoint.pt
│   └── optimizer_t5/
├── src/
│   ├── resume_parser.py    # PDF / DOCX / text parsing
│   ├── siamese_model.py    # Siamese Transformer (train + inference)
│   ├── resume_optimizer.py # T5 generative optimizer (train + inference)
│   ├── ats_checker.py      # ATS scoring engine
│   └── data_prep.py        # Dataset preparation
└── notebooks/              # Exploration notebooks
```

---

## References
- Resume2Vec: https://www.mdpi.com/2079-9292/14/4/794
- ResumeFlow: https://arxiv.org/abs/2402.06221
- Bommasani et al. (2021): https://arxiv.org/abs/2108.07258
