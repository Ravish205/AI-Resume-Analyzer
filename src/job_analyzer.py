"""
Job Description Analyzer Module for AI Resume Analyzer.

Analyzes job descriptions to extract:
- Title and core domain
- Required technical & soft skills
- Preferred / nice-to-have skills
- Minimum experience requirements (years)
- Minimum educational qualification requirements
- Key responsibilities and action bullet points
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from config import DEGREE_HIERARCHY
from src.skill_extractor import SkillExtractor


@dataclass
class JobDescriptionData:
    """Structured representation of parsed job description."""
    raw_text: str = ""
    clean_text: str = ""
    job_title: Optional[str] = None
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    all_skills: List[str] = field(default_factory=list)
    experience_required_years: float = 0.0
    required_education_level: int = 0
    required_education_name: Optional[str] = None
    responsibilities: List[str] = field(default_factory=list)
    domain_terms: List[str] = field(default_factory=list)


class JobAnalyzer:
    """Analyzes and structures candidate job descriptions."""

    def __init__(self, skill_extractor: Optional[SkillExtractor] = None) -> None:
        self.skill_extractor = skill_extractor or SkillExtractor()

    def analyze(self, text: str) -> JobDescriptionData:
        """
        Analyze a raw job description string and extract structured requirements.
        
        Args:
            text: Raw job description text.
            
        Returns:
            JobDescriptionData populated with requirements and skills.
        """
        clean_text = self._clean_text(text)
        if not clean_text:
            return JobDescriptionData()

        job_title = self._extract_job_title(clean_text)
        req_skills, pref_skills, all_skills = self._partition_skills(clean_text)
        exp_years = self._extract_experience_requirement(clean_text)
        edu_level, edu_name = self._extract_education_requirement(clean_text)
        responsibilities = self._extract_responsibilities(clean_text)
        domain_terms = self._extract_domain_terms(clean_text, all_skills)

        return JobDescriptionData(
            raw_text=text,
            clean_text=clean_text,
            job_title=job_title,
            required_skills=sorted(list(req_skills)),
            preferred_skills=sorted(list(pref_skills)),
            all_skills=sorted(list(all_skills)),
            experience_required_years=exp_years,
            required_education_level=edu_level,
            required_education_name=edu_name,
            responsibilities=responsibilities,
            domain_terms=domain_terms
        )

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace and bullet characters."""
        if not text:
            return ""
        text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219\u00B7\u25AA\u25CF]", "\n* ", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _extract_job_title(self, text: str) -> Optional[str]:
        """Detect the job title from the first lines of the posting."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return None

        # Look for explicit label 'Job Title: Software Engineer'
        for line in lines[:5]:
            match = re.search(r"(?:job\s+title|position|role)\s*[:\-]\s*(.+)", line, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Heuristic: the first short line mentioning typical engineering/analyst roles
        role_keywords = ["developer", "engineer", "scientist", "analyst", "architect", "manager", "specialist"]
        for line in lines[:3]:
            lower = line.lower()
            if any(k in lower for k in role_keywords) and len(line.split()) <= 7:
                return line.strip()

        return lines[0] if len(lines[0].split()) <= 6 else None

    def _partition_skills(self, text: str) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Partition skills into Required vs Preferred based on job description sections.
        """
        all_extracted = self.skill_extractor.extract_skill_names(text)
        if not all_extracted:
            return set(), set(), set()

        # Split text into sections based on keywords
        lines = text.split("\n")
        required_text_parts: List[str] = []
        preferred_text_parts: List[str] = []
        
        current_bucket = "general"  # "required", "preferred", "general"

        req_headers = [
            "requirements", "required skills", "what you need", "qualifications",
            "must have", "minimum qualifications", "basic qualifications", "core skills"
        ]
        pref_headers = [
            "preferred", "nice to have", "bonus", "plus", "good to have",
            "preferred qualifications", "desired skills", "bonus points"
        ]

        for line in lines:
            line_str = line.strip().lower()
            clean_h = re.sub(r"[:\-_|*#]", "", line_str).strip()

            if any(clean_h.startswith(h) or clean_h == h for h in req_headers):
                current_bucket = "required"
                continue
            elif any(clean_h.startswith(h) or clean_h == h for h in pref_headers):
                current_bucket = "preferred"
                continue
            elif any(clean_h.startswith(h) for h in ["responsibilities", "about the role", "benefits", "what we offer"]):
                current_bucket = "general"
                continue

            if current_bucket == "required":
                required_text_parts.append(line)
            elif current_bucket == "preferred":
                preferred_text_parts.append(line)

        req_skills = self.skill_extractor.extract_skill_names("\n".join(required_text_parts))
        pref_skills = self.skill_extractor.extract_skill_names("\n".join(preferred_text_parts))

        # If no explicit partitioning was identified in headings, treat all skills as required baseline
        if not req_skills and not pref_skills:
            req_skills = all_extracted.copy()
        elif not req_skills and pref_skills:
            # Remaining skills become required
            req_skills = all_extracted - pref_skills
        else:
            # Skills not explicitly marked preferred default to required
            unassigned = all_extracted - (req_skills | pref_skills)
            req_skills.update(unassigned)

        return req_skills, pref_skills, all_extracted

    def _extract_experience_requirement(self, text: str) -> float:
        """Extract minimum years of experience required by the job."""
        patterns = [
            r"(\d+(?:\.\d+)?)\+?\s*(?:to\s*\d+\s*)?years?(?:\s+of)?(?:\s+relevant)?\s+experience",
            r"minimum\s*(?:of)?\s*(\d+)\+?\s*years",
            r"at\s+least\s*(\d+)\+?\s*years"
        ]
        candidates: List[float] = []
        for pat in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                try:
                    candidates.append(float(match.group(1)))
                except (ValueError, IndexError):
                    continue

        return min(candidates) if candidates else 0.0

    def _extract_education_requirement(self, text: str) -> Tuple[int, Optional[str]]:
        """Extract required education level from job description."""
        text_lower = text.lower()
        if re.search(r"\b(ph\.?d|doctorate)\b", text_lower):
            return 4, "PhD / Doctorate"
        if re.search(r"\b(master'?s?|ms|m\.tech|mba)\b", text_lower):
            return 3, "Master's Degree"
        if re.search(r"\b(bachelor'?s?|bs|b\.tech|b\.e|undergraduate degree)\b", text_lower):
            return 2, "Bachelor's Degree"
        if re.search(r"\b(associate'?s?|diploma)\b", text_lower):
            return 1, "Associate / Diploma"
        return 0, None

    def _extract_responsibilities(self, text: str) -> List[str]:
        """Extract bullet points that specify responsibilities."""
        bullets: List[str] = []
        in_resp = False
        lines = text.split("\n")

        for line in lines:
            line_str = line.strip()
            clean_h = re.sub(r"[:\-_|*#]", "", line_str).strip().lower()
            
            if any(clean_h.startswith(h) for h in ["responsibilities", "what you will do", "duties", "day-to-day"]):
                in_resp = True
                continue
            elif in_resp and any(clean_h.startswith(h) for h in ["requirements", "qualifications", "benefits"]):
                in_resp = False
                break

            if in_resp and (line_str.startswith(("*", "-", "•")) or re.match(r"^\d+\.", line_str)):
                clean_bullet = re.sub(r"^[\*\-•\d.]+\s*", "", line_str).strip()
                if len(clean_bullet) > 10:
                    bullets.append(clean_bullet)

        return bullets[:10]

    def _extract_domain_terms(self, text: str, known_skills: Set[str]) -> List[str]:
        """Extract recurring domain-specific terms that aren't already captured as skills."""
        domain_patterns = [
            "distributed systems", "high throughput", "low latency", "scalability",
            "concurrency", "fault tolerance", "event-driven", "system design",
            "code review", "cross-functional", "end-to-end", "data pipeline",
            "production environment", "security best practices", "performance optimization"
        ]
        text_lower = text.lower()
        found = []
        for term in domain_patterns:
            if term in text_lower and term not in [s.lower() for s in known_skills]:
                found.append(term.title())
        return found

