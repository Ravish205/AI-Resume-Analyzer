"""
Skill Extractor Module for AI Resume Analyzer.

Provides rule-based and taxonomy-driven skill extraction with:
- Word-boundary and symbol-aware alias matching (e.g. C++, CI/CD, Node.js)
- Canonical skill name resolution
- Skill categorization (Programming Languages, Frameworks, Cloud, Databases, Soft Skills)
- Contextual match detection and occurrence frequency calculation
"""

import json
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from config import SKILLS_DB_PATH

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedSkill:
    """Represents an extracted skill with category and canonical form."""
    name: str               # Canonical name e.g., "Python", "Kubernetes"
    category: str           # e.g., "Programming Languages", "Cloud & DevOps"
    alias_matched: str      # e.g., "k8s" matched for "Kubernetes"
    frequency: int = 1      # Number of mentions in text


class SkillExtractor:
    """Extracts skills from text using a canonical taxonomy and alias mapping."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or SKILLS_DB_PATH
        self.taxonomy: Dict[str, Dict[str, List[str]]] = {}
        self.skill_to_category: Dict[str, str] = {}
        self.alias_patterns: List[Tuple[re.Pattern, str, str]] = []  # (compiled_pattern, canonical_name, alias)
        self._load_taxonomy()

    def _load_taxonomy(self) -> None:
        """Load the skills database and compile search regexes with symbol handling."""
        if not self.db_path.exists():
            logger.warning(f"Skills database not found at {self.db_path}. Using built-in minimal defaults.")
            data = self._get_fallback_taxonomy()
        else:
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Error reading {self.db_path}: {e}")
                data = self._get_fallback_taxonomy()

        categories = data.get("categories", {})
        self.taxonomy = categories

        compiled_patterns = []
        for category, skills in categories.items():
            for canonical_name, aliases in skills.items():
                self.skill_to_category[canonical_name] = category
                # Ensure canonical name itself is tested
                all_variants = set(aliases + [canonical_name.lower()])
                for alias in sorted(all_variants, key=len, reverse=True):
                    escaped = re.escape(alias)
                    is_start_alnum = bool(alias and alias[0].isalnum())
                    is_end_alnum = bool(alias and alias[-1].isalnum())
                    prefix = r"\b" if is_start_alnum else r"(?:^|[\s,;/()\-–])"
                    suffix = r"\b" if is_end_alnum else r"(?:$|[\s,;/.()!?:])"
                    pattern_str = f"{prefix}{escaped}{suffix}"
                    try:
                        pat = re.compile(pattern_str, re.IGNORECASE)
                        compiled_patterns.append((pat, canonical_name, alias))
                    except re.error as err:
                        logger.debug(f"Failed to compile pattern '{pattern_str}': {err}")

        self.alias_patterns = compiled_patterns

    def extract_skills(self, text: str) -> List[ExtractedSkill]:
        """
        Extract canonical skills from text with occurrence count.
        
        Args:
            text: Resume or Job Description text.
            
        Returns:
            List of ExtractedSkill objects sorted alphabetically by name.
        """
        if not text or not text.strip():
            return []

        matched_skills: Dict[str, ExtractedSkill] = {}
        text_lower = text.lower()

        for pattern, canonical_name, alias in self.alias_patterns:
            matches = list(pattern.finditer(text_lower))
            if matches:
                freq = len(matches)
                category = self.skill_to_category.get(canonical_name, "General")
                
                if canonical_name in matched_skills:
                    # Increment frequency if another alias or repeat found
                    prev = matched_skills[canonical_name]
                    matched_skills[canonical_name] = ExtractedSkill(
                        name=canonical_name,
                        category=category,
                        alias_matched=prev.alias_matched,
                        frequency=prev.frequency + freq
                    )
                else:
                    matched_skills[canonical_name] = ExtractedSkill(
                        name=canonical_name,
                        category=category,
                        alias_matched=alias,
                        frequency=freq
                    )

        return sorted(matched_skills.values(), key=lambda s: s.name)

    def extract_skill_names(self, text: str) -> Set[str]:
        """Convenience method returning a set of canonical skill names."""
        extracted = self.extract_skills(text)
        return {s.name for s in extracted}

    def categorize_skills(self, skills: List[ExtractedSkill]) -> Dict[str, List[str]]:
        """Group a list of extracted skills by category."""
        grouped: Dict[str, List[str]] = {}
        for skill in skills:
            grouped.setdefault(skill.category, []).append(skill.name)
        for cat in grouped:
            grouped[cat].sort()
        return grouped

    def compare_skills(
        self,
        resume_skills: Set[str],
        job_skills: Set[str]
    ) -> Dict[str, List[str]]:
        """
        Perform 3-way skill gap analysis between resume and job description.
        
        Returns:
            Dict containing:
            - 'matching': Skills found in both
            - 'missing': Required skills in JD but absent in resume
            - 'additional': Skills present in resume but not in JD
        """
        resume_norm = {s.strip() for s in resume_skills if s and s.strip()}
        job_norm = {s.strip() for s in job_skills if s and s.strip()}

        matching = sorted(list(resume_norm.intersection(job_norm)))
        missing = sorted(list(job_norm - resume_norm))
        additional = sorted(list(resume_norm - job_norm))

        return {
            "matching": matching,
            "missing": missing,
            "additional": additional,
        }

    def _get_fallback_taxonomy(self) -> Dict[str, Any]:
        """Emergency built-in fallback taxonomy if json file is missing."""
        return {
            "categories": {
                "Programming Languages": {
                    "Python": ["python", "python3"],
                    "SQL": ["sql"],
                    "JavaScript": ["javascript", "js"],
                    "C++": ["c++", "cpp"],
                    "Java": ["java"]
                },
                "Frameworks & Libraries": {
                    "FastAPI": ["fastapi"],
                    "Django": ["django"],
                    "Flask": ["flask"],
                    "React": ["react"],
                    "Scikit-learn": ["scikit-learn", "sklearn"]
                },
                "Tools & Methodologies": {
                    "Git": ["git", "github"],
                    "Docker": ["docker"],
                    "REST APIs": ["rest api", "restful api", "rest apis"]
                },
                "Machine Learning & Data": {
                    "Machine Learning": ["machine learning", "ml"],
                    "Natural Language Processing": ["nlp", "natural language processing"]
                },
                "Soft Skills": {
                    "Communication": ["communication"],
                    "Leadership": ["leadership"],
                    "Problem Solving": ["problem solving"]
                }
            }
        }
