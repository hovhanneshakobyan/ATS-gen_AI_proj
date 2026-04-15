"""
data_prep.py
Prepares training data for both models from Kaggle datasets.

Sources (download manually and place in data/raw/):
  - Resume Dataset   : https://www.kaggle.com/datasets/palaksood97/resume-dataset
  - Job Descriptions : https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset

Outputs:
  data/processed/pairs.csv          — for Siamese model  (resume_text, jd_text, label)
  data/processed/optimized_pairs.csv — for T5 optimizer  (resume_text, jd_text, optimized_resume)
"""

import os
import pandas as pd
import random
from loguru import logger

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def load_resumes(path: str) -> pd.DataFrame:
    """Load and normalize resume CSV."""
    df = pd.read_csv(path)
    # Normalize column names (dataset-specific)
    df.columns = [c.lower().strip() for c in df.columns]
    text_col = next((c for c in df.columns if "resume" in c or "text" in c), df.columns[0])
    return df[[text_col]].rename(columns={text_col: "resume_text"}).dropna()


def load_jds(path: str) -> pd.DataFrame:
    """Load and normalize job description CSV."""
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    text_col = next((c for c in df.columns if "description" in c or "job" in c), df.columns[0])
    return df[[text_col]].rename(columns={text_col: "jd_text"}).dropna()


def build_siamese_pairs(resumes: pd.DataFrame, jds: pd.DataFrame,
                        n_positive: int = 2000, n_negative: int = 2000) -> pd.DataFrame:
    """
    Positive pairs  (label=1): resume and JD from same category/domain
    Negative pairs  (label=0): random mismatched resume + JD
    """
    res_list = resumes["resume_text"].tolist()
    jd_list  = jds["jd_text"].tolist()

    # Positive: sequential pairing (assumes datasets are category-aligned)
    pos_n    = min(n_positive, len(res_list), len(jd_list))
    positive = pd.DataFrame({
        "resume_text": res_list[:pos_n],
        "jd_text":     jd_list[:pos_n],
        "label":       [1] * pos_n,
    })

    # Negative: random shuffle
    shuffled_jd = jd_list.copy()
    random.shuffle(shuffled_jd)
    neg_n    = min(n_negative, len(res_list), len(shuffled_jd))
    negative = pd.DataFrame({
        "resume_text": res_list[:neg_n],
        "jd_text":     shuffled_jd[:neg_n],
        "label":       [0] * neg_n,
    })

    pairs = pd.concat([positive, negative]).sample(frac=1, random_state=RANDOM_SEED)
    return pairs.reset_index(drop=True)


def build_optimizer_pairs(resumes: pd.DataFrame, jds: pd.DataFrame,
                          n: int = 1000) -> pd.DataFrame:
    """
    For T5 training we pair each resume with its matching JD.
    The 'optimized_resume' column is the gold label.

    NOTE: If you have human-curated optimized resumes, replace the
    placeholder column below with the real data.
    As a bootstrap, we use the original resume as the target (self-supervised),
    which at minimum teaches the model the format.  Replace with better data
    once available.
    """
    res_list = resumes["resume_text"].tolist()
    jd_list  = jds["jd_text"].tolist()
    n        = min(n, len(res_list), len(jd_list))

    return pd.DataFrame({
        "resume_text":      res_list[:n],
        "jd_text":          jd_list[:n],
        "optimized_resume": res_list[:n],   # ← replace with curated labels
    })


def run(resume_csv: str = "data/raw/resumes.csv",
        jd_csv:     str = "data/raw/job_descriptions.csv"):

    logger.info("Loading raw data …")
    resumes = load_resumes(resume_csv)
    jds     = load_jds(jd_csv)
    logger.info(f"  Resumes: {len(resumes)}  |  JDs: {len(jds)}")

    os.makedirs("data/processed", exist_ok=True)

    logger.info("Building Siamese pairs …")
    pairs = build_siamese_pairs(resumes, jds)
    pairs.to_csv("data/processed/pairs.csv", index=False)
    logger.success(f"  Saved {len(pairs)} pairs → data/processed/pairs.csv")

    logger.info("Building T5 optimizer pairs …")
    opt_pairs = build_optimizer_pairs(resumes, jds)
    opt_pairs.to_csv("data/processed/optimized_pairs.csv", index=False)
    logger.success(f"  Saved {len(opt_pairs)} pairs → data/processed/optimized_pairs.csv")


if __name__ == "__main__":
    run()
