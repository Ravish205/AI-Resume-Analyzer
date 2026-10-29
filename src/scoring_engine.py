"""
Scoring Engine Module for AI Resume Analyzer.

Implements transparent, explainable compatibility score computation:
- Configurable component weights (Skills, Semantics, Experience, Education)
- Normalized sub-scores strictly bounded to [0.0, 100.0]
- Graceful handling of missing or partial information
- Clear ATS disclaimer and tier categorization
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from config import SCORE_TIERS, ScoringWeights
from src.job_analyzer import JobDescriptionData
from src.resume_parser import ResumeData
from src.similarity_engine import SimilarityEngine
from src.skill_extractor import SkillExtractor

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Comprehensive output of resume-to-job match scoring."""
    overall_score: float
    tier: str
    tier_color: str
    tier_description: str
    skills_score: float
    semantic_score: float
    experience_score: float
    education_score: float
    weights_used: ScoringWeights
    matching_skills: List[str]
    missing_skills: List[str]
    additional_skills: List[str]
    skill_coverage_ratio: float
    section_similarities: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    disclaimer: str = (
        "Estimated resume-to-job compatibility score based on objective text analysis. "
        "This is an indicative alignment metric, not a proprietary ATS prediction or hiring guarantee."
    )


class ScoringEngine:
    """Computes multi-dimensional alignment scores between resume and job description."""

    def __init__(
        self,
        similarity_engine: Optional[SimilarityEngine] = None,
        skill_extractor: Optional[SkillExtractor] = None,
        default_weights: Optional[ScoringWeights] = None
    ) -> None:
        self.similarity_engine = similarity_engine or SimilarityEngine()
        self.skill_extractor = skill_extractor or SkillExtractor()
        self.weights = default_weights or ScoringWeights()

    def calculate_match(
        self,
        resume: ResumeData,
        job: JobDescriptionData,
        weights: Optional[ScoringWeights] = None,
        use_transformer: bool = False
    ) -> MatchResult:
        """
        Calculate complete compatibility score and diagnostic breakdown.
        
        Args:
            resume: Parsed ResumeData.
            job: Parsed JobDescriptionData.
            weights: Optional custom scoring weights override.
            use_transformer: Whether to use dense semantic embeddings.
            
        Returns:
            MatchResult containing overall score, component scores, and details.
        """
        w = weights or self.weights
        notes: List[str] = []

        # Extract resume skills
        resume_skills_set = self.skill_extractor.extract_skill_names(resume.clean_text)
        job_req_skills = set(job.required_skills)
        job_pref_skills = set(job.preferred_skills)
        all_job_skills = set(job.all_skills) if job.all_skills else (job_req_skills | job_pref_skills)

        # 1. Skill Gap Analysis
        skill_comparison = self.skill_extractor.compare_skills(resume_skills_set, all_job_skills)
        matching_skills = skill_comparison["matching"]
        missing_skills = skill_comparison["missing"]
        additional_skills = skill_comparison["additional"]

        # 2. Skills Coverage Score (0 - 100)
        skills_score, coverage_ratio = self._calculate_skills_score(
            resume_skills=resume_skills_set,
            required_skills=job_req_skills,
            preferred_skills=job_pref_skills,
            matching_skills=matching_skills
        )

        # 3. Semantic Similarity Score (0 - 100)
        semantic_sim = self.similarity_engine.compute_similarity(
            resume_text=resume.clean_text,
            job_text=job.clean_text,
            use_transformer=use_transformer
        )
        semantic_score = round(semantic_sim * 100.0, 1)

        # Section-specific similarities
        section_similarities = self.similarity_engine.compute_section_breakdown(
            sections=resume.sections,
            job_text=job.clean_text
        )

        # 4. Experience Relevance Score (0 - 100)
        exp_score, exp_note = self._calculate_experience_score(resume, job)
        if exp_note:
            notes.append(exp_note)

        # 5. Education & Certification Score (0 - 100)
        edu_score, edu_note = self._calculate_education_score(resume, job)
        if edu_note:
            notes.append(edu_note)

        # 6. Overall Weighted Score Calculation
        overall = (
            (w.required_skills * skills_score) +
            (w.semantic_similarity * semantic_score) +
            (w.experience * exp_score) +
            (w.education * edu_score)
        )
        overall_score = round(max(0.0, min(100.0, overall)), 1)

        # Tier assignment
        tier, tier_color, tier_desc = self._get_tier(overall_score)

        return MatchResult(
            overall_score=overall_score,
            tier=tier,
            tier_color=tier_color,
            tier_description=tier_desc,
            skills_score=round(skills_score, 1),
            semantic_score=round(semantic_score, 1),
            experience_score=round(exp_score, 1),
            education_score=round(edu_score, 1),
            weights_used=w,
            matching_skills=matching_skills,
            missing_skills=missing_skills,
            additional_skills=additional_skills,
            skill_coverage_ratio=round(coverage_ratio, 2),
            section_similarities=section_similarities,
            notes=notes
        )

    def _calculate_skills_score(
        self,
        resume_skills: Set[str],
        required_skills: Set[str],
        preferred_skills: Set[str],
        matching_skills: List[str]
    ) -> (float, float):
        """Calculate normalized skills coverage score and ratio."""
        if not required_skills and not preferred_skills:
            if not matching_skills:
                return 50.0, 0.5  # Neutral baseline when JD lists no explicit tech skills
            # Scale based on relevant skills present
            ratio = min(len(matching_skills) / 5.0, 1.0)
            return ratio * 100.0, ratio

        # Required skills coverage (primary)
        req_matched = [s for s in required_skills if s in resume_skills]
        req_ratio = len(req_matched) / len(required_skills) if required_skills else 1.0

        if preferred_skills:
            # Preferred skills coverage (bonus)
            pref_matched = [s for s in preferred_skills if s in resume_skills]
            pref_ratio = len(pref_matched) / len(preferred_skills)
            score = (req_ratio * 85.0) + (pref_ratio * 15.0)
            overall_ratio = (len(req_matched) + len(pref_matched)) / (len(required_skills) + len(preferred_skills))
        else:
            score = req_ratio * 100.0
            overall_ratio = req_ratio

        return max(0.0, min(100.0, score)), overall_ratio

    def _calculate_experience_score(
        self,
        resume: ResumeData,
        job: JobDescriptionData
    ) -> (float, Optional[str]):
        """Calculate experience alignment based on years and quantifiable impact."""
        cand_exp = resume.experience_years
        req_exp = job.experience_required_years

        base_score = 60.0
        note = None

        if req_exp > 0:
            if cand_exp >= req_exp:
                base_score = 80.0 + min((cand_exp - req_exp) * 2.0, 10.0)
                note = f"Candidate experience ({cand_exp:.1f} yrs) meets or exceeds job requirement ({req_exp:.1f} yrs)."
            elif cand_exp > 0:
                base_score = (cand_exp / req_exp) * 75.0
                note = f"Candidate experience ({cand_exp:.1f} yrs) is below stated job requirement ({req_exp:.1f} yrs)."
            else:
                base_score = 45.0
                note = "Candidate experience duration could not be reliably verified from date ranges."
        else:
            # Job didn't specify required years
            base_score = min(70.0 + (cand_exp * 3.0), 90.0)

        # Quantifiable metrics bonus (up to 10 points)
        metrics_bonus = min(resume.metrics_count * 2.0, 10.0)
        total_exp_score = min(100.0, base_score + metrics_bonus)

        return total_exp_score, note

    def _calculate_education_score(
        self,
        resume: ResumeData,
        job: JobDescriptionData
    ) -> (float, Optional[str]):
        """Calculate education and certification relevance."""
        cand_level = resume.highest_degree_level
        req_level = job.required_education_level
        note = None

        if req_level > 0:
            if cand_level >= req_level:
                score = 90.0
                note = f"Candidate education meets requirement ({job.required_education_name})."
            elif cand_level > 0:
                score = 65.0
                note = f"Candidate education ({resume.highest_degree_name}) is below job requirement ({job.required_education_name})."
            else:
                score = 50.0
                note = "No formal degree credentials explicitly identified."
        else:
            score = 85.0 if cand_level > 0 else 70.0

        # Certifications bonus (up to +10 points)
        if "certifications" in resume.sections and len(resume.sections["certifications"].strip()) > 15:
            score = min(100.0, score + 10.0)

        return score, note

    def _get_tier(self, score: float) -> (str, str, str):
        """Retrieve classification tier, color, and description."""
        for tier_name, meta in SCORE_TIERS.items():
            if score >= meta["min"]:
                return tier_name, meta["color"], meta["description"]
        return "Low Match", "#EF4444", "Significant gaps in required skills, keywords, or experience."
