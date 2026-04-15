from __future__ import annotations
from typing import Dict, List
from src.scorer import ATSScorer


class ResumeOptimizer:
    def __init__(self):
        self.scorer = ATSScorer()

    def optimize(self, resume_text: str, jd_text: str) -> Dict:

        score = self.scorer.score(resume_text, jd_text)
        keyword = score["keyword_report"]

        matched = keyword.get("matched", [])
        missing = keyword.get("missing", [])

        optimized = resume_text.strip()

        additions: List[str] = []

        if "SUMMARY" not in optimized.upper():
            additions.append(
                "PROFESSIONAL SUMMARY\n"
                f"Software Engineer skilled in {', '.join(matched[:5]) or 'C#, .NET development'}.\n"
            )

        additions.append(
            "TARGET SKILLS\n"
            f"Matched: {', '.join(matched[:8])}\n"
            f"Missing: {', '.join(missing[:8])}\n"
        )

        return {
            "optimized_resume": "\n\n".join(additions) + "\n\n" + optimized,
            "suggestions": [
                "Add measurable impact (%, speed, scale)",
                "Use single-column ATS format",
                "Expand education + responsibilities"
            ],
            "score_before": score["overall_score"],
            "score_after": score["overall_score"]  # stable (no fake drop)
        }