from __future__ import annotations

import re
from typing import Dict, List, Set


class SkillExtractor:
    """
    General-purpose skill extractor with:
    - normalization
    - alias mapping
    - phrase-first matching
    - light typo repair
    - concept grouping for related technologies
    """

    def __init__(self):
        self.skill_aliases: Dict[str, List[str]] = {
            "c#": ["c#", "c sharp", "c-sharp"],
            ".net": [".net", "dotnet", "dot net"],
            "asp.net": ["asp.net", "asp net", "aspnet"],
            "sql": ["sql", "sql server", "postgresql", "mysql"],
            "javascript": ["javascript", "java script", "js", "javasript"],
            "html5": ["html5", "html"],
            "css3": ["css3", "css"],
            "wpf": ["wpf", "windows presentation foundation"],
            "maui": ["maui", ".net maui", "dotnet maui"],
            "xamarin": ["xamarin"],
            "android": ["android", "android development", "android developer"],
            "azure devops": ["azure devops", "azuredevops", "devops"],
            "rest api": ["rest api", "restful api", "restful apis", "api development"],
            "entity framework": ["entity framework", "ef core", "entityframework"],
            "unit testing": ["unit testing", "automated testing", "testing"],
            "oop": [
                "object-oriented",
                "object oriented",
                "object-oriented design",
                "oop",
            ],
            "design patterns": ["design patterns", "software design patterns"],
            "full stack": ["full stack", "full-stack", "fullstack"],
            "desktop ui": ["desktop applications", "desktop ui", "desktop development"],
            "web ui": ["web applications", "web ui", "frontend", "front-end"],
            "git": ["git", "github", "gitlab"],
            "python": ["python"],
            "java": ["java"],
            "c++": ["c++", "cpp"],
            "kotlin": ["kotlin"],
            "flask": ["flask"],
            "django": ["django"],
            "selenium": ["selenium"],
            "tensorflow": ["tensorflow"],
            "pandas": ["pandas"],
            "firebase": ["firebase", "fire base"],
            "nosql": ["nosql", "no sql"],
            "postgresql": ["postgresql", "postgres"],
            "sql server": ["sql server"],
        }

        # broader concept relationships for partial semantic credit
        self.related_concepts: Dict[str, Set[str]] = {
            "android": {"maui", "xamarin", "kotlin"},
            "maui": {"android", "xamarin", ".net"},
            "xamarin": {"android", "maui", ".net"},
            "desktop ui": {"wpf", "maui"},
            "web ui": {"html5", "css3", "javascript", "asp.net"},
            "asp.net": {".net", "c#", "sql", "web ui"},
            ".net": {"c#", "asp.net", "maui", "wpf"},
            "oop": {"c#", "java", "c++", "design patterns"},
            "design patterns": {"oop", "c#", "java", "c++"},
            "rest api": {"asp.net", "c#", ".net", "python"},
            "full stack": {"html5", "css3", "javascript", "asp.net", "sql"},
            "sql": {"sql server", "postgresql", "mysql"},
        }

    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = text.replace("javasript", "javascript")
        text = text.replace("java script", "javascript")
        text = text.replace("asp net", "asp.net")
        text = text.replace("dotnet", ".net")
        text = text.replace("dot net", ".net")
        text = text.replace("entityframework", "entity framework")
        text = text.replace("restful apis", "rest api")
        text = text.replace("restful api", "rest api")
        text = text.replace("object oriented", "object-oriented")
        text = re.sub(r"[^a-z0-9+#./\-\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_skills(self, text: str) -> List[str]:
        norm = self.normalize_text(text)
        found = []

        # phrase-first matching
        for canonical, aliases in sorted(
            self.skill_aliases.items(),
            key=lambda x: max(len(a) for a in x[1]),
            reverse=True,
        ):
            if any(alias in norm for alias in aliases):
                found.append(canonical)

        # dedupe while preserving order
        seen = set()
        result = []
        for s in found:
            if s not in seen:
                seen.add(s)
                result.append(s)

        # remove weaker duplicates if stronger one exists
        result = self._dedupe_overlaps(result)
        return result

    def jd_skill_report(self, resume_text: str, jd_text: str) -> Dict[str, List[str] | float]:
        resume_skills = set(self.extract_skills(resume_text))
        jd_skills = set(self.extract_skills(jd_text))

        matched = sorted(resume_skills.intersection(jd_skills))
        missing = sorted(jd_skills.difference(resume_skills))
        coverage = (len(matched) / max(len(jd_skills), 1)) * 100.0

        related_matches = []
        for jd_skill in sorted(missing):
            related = self.find_related_resume_skills(jd_skill, resume_skills)
            if related:
                related_matches.append({
                    "jd_skill": jd_skill,
                    "related_resume_skills": related,
                })

        return {
            "resume_skills": sorted(resume_skills),
            "jd_skills": sorted(jd_skills),
            "matched": matched,
            "missing": missing,
            "related_matches": related_matches,
            "coverage_pct": round(coverage, 1),
        }

    def find_related_resume_skills(self, jd_skill: str, resume_skills: Set[str]) -> List[str]:
        related = self.related_concepts.get(jd_skill, set())
        hits = sorted(s for s in resume_skills if s in related)
        return hits

    def relatedness_score(self, left: str, right: str) -> float:
        """
        Skill-level relatedness, not full sentence semantic similarity.
        Returns a small structured score:
        - 1.0 exact canonical match
        - 0.7 directly related concept
        - 0.0 otherwise
        """
        left_norm = self.extract_skills(left)
        right_norm = self.extract_skills(right)

        if not left_norm or not right_norm:
            return 0.0

        if set(left_norm).intersection(set(right_norm)):
            return 1.0

        for l in left_norm:
            related = self.related_concepts.get(l, set())
            if any(r in related for r in right_norm):
                return 0.7

        for r in right_norm:
            related = self.related_concepts.get(r, set())
            if any(l in related for l in left_norm):
                return 0.7

        return 0.0

    def _dedupe_overlaps(self, skills: List[str]) -> List[str]:
        skills_set = set(skills)

        # If sql server exists, keep both sql and sql server? yes, because sql is broader.
        # If c# exists, do not keep plain c from accidental tokenization. Here we don't extract plain c.
        # Keep this function for future overlap cleanup.
        ordered = []
        seen = set()
        for s in skills:
            if s not in seen:
                seen.add(s)
                ordered.append(s)
        return ordered
