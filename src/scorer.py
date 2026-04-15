from __future__ import annotations

import re
from typing import Dict, List
import numpy as np

from src.jd_rules import build_requirement_table
from src.sections import SectionExtractor
from src.semantic_model import SemanticResumeMatcher
from src.utils import ACTION_VERBS, count_numeric_impact


# =========================
# SAFE COSINE
# =========================
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0

    a = np.asarray(a)
    b = np.asarray(b)

    if a.size == 0 or b.size == 0:
        return 0.0

    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    if a.shape[1] != b.shape[1]:
        return 0.0

    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)

    sim = np.dot(a, b.T)
    return float(np.mean(np.max(sim, axis=1)))


class ATSScorer:
    def __init__(self):
        self.sections = SectionExtractor()
        self.semantic = SemanticResumeMatcher()

    # -------------------------
    def normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower())

    # -------------------------
    def extract_skills(self, text: str) -> List[str]:
        low = self.normalize(text)
        skills = []

        keywords = [
            "c#", ".net", "asp.net", "sql", "javascript",
            "python", "java", "c++", "maui", "wpf"
        ]

        for k in keywords:
            if k in low:
                skills.append(k)

        return list(set(skills))

    # -------------------------
    def extract_bullets(self, text: str) -> List[str]:
        if not text:
            return []

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return [l for l in lines if len(l.split()) > 6]

    # -------------------------
    def safe_encode(self, texts: List[str]):
        if not texts:
            return np.zeros((1, 384))

        emb = self.semantic.encode(texts)
        if emb is None or len(emb) == 0:
            return np.zeros((1, 384))

        return np.array(emb)

    # -------------------------
    def semantic_score(self, resume_text, jd_text, bullets, edu):

        jd_rows = build_requirement_table(jd_text, self.extract_skills)

        jd_resp, jd_skill, jd_edu = [], [], []

        for r in jd_rows:
            t = r.get("text", "").lower()

            if any(x in t for x in ["degree", "education", "bachelor", "master"]):
                jd_edu.append(r["text"])
            elif r.get("skills"):
                jd_skill.append(r["text"])
            else:
                jd_resp.append(r["text"])

        resp = cosine(
            self.safe_encode(jd_resp),
            self.safe_encode(bullets)
        )

        skill = cosine(
            self.safe_encode(jd_skill),
            self.safe_encode(self.extract_skills(resume_text))
        )

        edu = cosine(
            self.safe_encode(jd_edu),
            self.safe_encode(edu)
        )

        # 🔥 IMPORTANT FIX: avoid overly harsh scoring
        overall = (
            0.45 * resp +
            0.35 * skill +
            0.20 * edu
        )

        return {
            "overall_similarity": round(min(overall + 0.15, 1.0), 4),  # BOOST FIX
            "responsibility_score": round(resp, 4),
            "skill_score": round(skill, 4),
            "education_score": round(edu, 4),
        }

    # -------------------------
    def achievement_score(self, bullets: List[str]) -> float:
        if not bullets:
            return 0.25

        v = sum(1 for b in bullets if b.split()[0].lower() in ACTION_VERBS)
        n = sum(1 for b in bullets if count_numeric_impact(b) > 0)

        return min(1.0, (v + n) / (2 * len(bullets)))

    # -------------------------
    def keyword_report(self, resume_text: str, jd_text: str):
        resume_skills = set(self.extract_skills(resume_text))
        jd_skills = set(self.extract_skills(jd_text))

        matched = list(resume_skills & jd_skills)
        missing = list(jd_skills - resume_skills)

        return {
            "matched": matched,
            "missing": missing,
            "coverage_pct": len(matched) / max(len(jd_skills), 1) * 100,
        }

    # -------------------------
    def score(self, resume_text: str, jd_text: str) -> Dict:

        exp = self.sections.extract_experience_block(resume_text)
        bullets = self.extract_bullets(exp)
        edu = self.sections.extract_education_lines(resume_text)

        keyword = self.keyword_report(resume_text, jd_text)
        semantic = self.semantic_score(resume_text, jd_text, bullets, edu)
        achievement = self.achievement_score(bullets)

        section_score = sum(
            1 for s in ["experience", "education", "skills"]
            if self.sections.detect_sections(resume_text).get(s, False)
        ) / 3

        overall = (
            0.40 * (keyword["coverage_pct"] / 100) +
            0.30 * semantic["overall_similarity"] +
            0.15 * section_score +
            0.15 * achievement
        ) * 100

        return {
            "overall_score": round(overall, 1),
            "breakdown": {
                "keyword": round(keyword["coverage_pct"], 1),
                "semantic": round(semantic["overall_similarity"] * 100, 1),
                "sections": round(section_score * 100, 1),
                "achievement": round(achievement * 100, 1),
            },
            "keyword_report": keyword,
            "semantic_report": semantic,

            # ✅ FIX FOR STREAMLIT ERROR
            "format_warnings": []
        }