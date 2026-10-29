"""
Application Configuration for AI Resume Analyzer.

Defines default scoring weights, thresholds, file parsing settings,
section aliases, action verb taxonomies, and filesystem paths.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# Base directory paths
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
SAMPLE_DATA_DIR: Path = BASE_DIR / "sample_data"
DOCS_DIR: Path = BASE_DIR / "docs"

# Skills database file path
SKILLS_DB_PATH: Path = DATA_DIR / "skills_database.json"

# File constraints
MAX_FILE_SIZE_MB: int = 10
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt"]


@dataclass(frozen=True)
class ScoringWeights:
    """Weights used for overall resume-to-job compatibility score calculation."""
    required_skills: float = 0.50     # 50% Required skill coverage
    semantic_similarity: float = 0.25 # 25% Semantic & contextual text similarity
    experience: float = 0.15          # 15% Experience years & quantifiable achievements
    education: float = 0.10           # 10% Education level & certifications relevance

    def validate(self) -> bool:
        """Verify weights sum approximately to 1.0."""
        total = self.required_skills + self.semantic_similarity + self.experience + self.education
        return abs(total - 1.0) < 1e-4


# Standard score tiers
SCORE_TIERS: Dict[str, Dict[str, str]] = {
    "Excellent Match": {
        "min": 80,
        "color": "#10B981",  # Green
        "description": "Strong alignment across required skills, experience, and background."
    },
    "Good Match": {
        "min": 65,
        "color": "#3B82F6",  # Blue
        "description": "Solid match with minor skill or experience gaps."
    },
    "Moderate Match": {
        "min": 50,
        "color": "#F59E0B",  # Amber
        "description": "Fair baseline, but missing key required skills or context."
    },
    "Low Match": {
        "min": 0,
        "color": "#EF4444",  # Red
        "description": "Significant gaps in required skills, keywords, or experience."
    }
}

# Resume section header aliases for section segmentation
SECTION_PATTERNS: Dict[str, List[str]] = {
    "summary": [
        "summary", "professional summary", "profile", "about me",
        "career objective", "objective", "executive summary", "personal profile"
    ],
    "skills": [
        "skills", "technical skills", "core competencies", "key skills",
        "technologies", "expertise", "proficiencies", "areas of expertise",
        "tools & technologies", "technical proficiencies"
    ],
    "experience": [
        "experience", "work experience", "employment history",
        "professional experience", "work history", "career history",
        "internships", "professional background"
    ],
    "education": [
        "education", "academic background", "academic history",
        "qualifications", "degrees", "educational qualifications",
        "academic credentials"
    ],
    "projects": [
        "projects", "personal projects", "key projects", "academic projects",
        "selected projects", "open source contributions"
    ],
    "certifications": [
        "certifications", "certificates", "licenses", "credentials",
        "professional certifications", "accreditations", "courses"
    ]
}

# Degree level keywords mapped to rank hierarchy
DEGREE_HIERARCHY: Dict[str, int] = {
    "phd": 4,
    "doctorate": 4,
    "master": 3,
    "ms": 3,
    "msc": 3,
    "m.tech": 3,
    "mba": 3,
    "bachelor": 2,
    "bs": 2,
    "bsc": 2,
    "b.tech": 2,
    "be": 2,
    "associate": 1,
    "diploma": 1
}

# Action verbs taxonomy for resume recommendations
STRONG_ACTION_VERBS: List[str] = [
    "accelerated", "accomplished", "achieved", "administered", "architected",
    "automated", "boosted", "built", "centralized", "championed", "collaborated",
    "conceived", "consolidated", "constructed", "crafted", "delivered", "deployed",
    "designed", "developed", "directed", "eliminated", "enabled", "engineered",
    "established", "executed", "expanded", "expedited", "formulated", "founded",
    "generated", "guided", "implemented", "improved", "increased", "initiated",
    "innovated", "inspected", "instituted", "integrated", "launched", "led",
    "maintained", "managed", "maximized", "mentored", "migrated", "minimized",
    "modernized", "negotiated", "optimized", "orchestrated", "overhauled",
    "pioneered", "produced", "reduced", "refactored", "resolved", "revamped",
    "scaled", "secured", "simplified", "spearheaded", "standardized",
    "streamlined", "strengthened", "surpassed", "transformed", "unified", "upgraded"
]

WEAK_ACTION_VERBS: List[str] = [
    "worked on", "helped", "assisted", "responsible for", "duties included",
    "handled", "participated in", "involved in", "did", "tried to", "supported"
]

