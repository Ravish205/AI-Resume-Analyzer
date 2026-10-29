"""
Recommendation Engine Module for AI Resume Analyzer.

Generates evidence-based, actionable resume improvement suggestions:
- Missing critical skills and domain keywords
- Action verb enhancements (replacing passive phrasing with impactful verbs)
- Quantifiable metrics integration (XYZ resume formula)
- Resume structure and ATS formatting completeness
- Section-by-section health checks
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import STRONG_ACTION_VERBS, WEAK_ACTION_VERBS
from src.job_analyzer import JobDescriptionData
from src.resume_parser import ResumeData
from src.scoring_engine import MatchResult


@dataclass
class Recommendation:
    """Actionable improvement recommendation."""
    category: str        # e.g., "Skills Gap", "Action Verbs", "Quantifiable Metrics", "Structure & ATS"
    priority: str        # "High", "Medium", "Low"
    title: str
    message: str
    evidence: Optional[str] = None
    examples: List[str] = field(default_factory=list)


class RecommendationEngine:
    """Generates context-aware, non-hallucinatory recommendations for resume improvement."""

    def __init__(self) -> None:
        self.strong_verbs = STRONG_ACTION_VERBS
        self.weak_verbs = WEAK_ACTION_VERBS

    def generate_recommendations(
        self,
        resume: ResumeData,
        job: JobDescriptionData,
        match_result: MatchResult
    ) -> List[Recommendation]:
        """
        Produce prioritized recommendations based on parsed resume and job match results.
        
        Args:
            resume: Parsed ResumeData.
            job: Parsed JobDescriptionData.
            match_result: MatchResult from scoring engine.
            
        Returns:
            List of Recommendation objects ordered by priority.
        """
        recs: List[Recommendation] = []

        # 1. Missing Skills Analysis
        missing_skills = match_result.missing_skills
        if missing_skills:
            top_missing = missing_skills[:6]
            recs.append(
                Recommendation(
                    category="Skills Gap",
                    priority="High",
                    title="Incorporate Missing Target Skills",
                    message=(
                        f"The job posting explicitly emphasizes: {', '.join(top_missing)}. "
                        "If you have academic, project, or professional experience with these technologies, "
                        "ensure they are explicitly listed in your Skills section and referenced in your project bullets. "
                        "Never fabricate experience you do not possess."
                    ),
                    evidence=f"Missing required keywords: {', '.join(top_missing)}",
                    examples=[
                        f"Example: 'Built RESTful endpoints using {top_missing[0] if top_missing else 'FastAPI'} to serve real-time predictions.'"
                    ]
                )
            )

        # 2. Action Verbs & Impact
        weak_verbs_found = self._find_weak_verbs(resume.clean_text)
        if weak_verbs_found:
            recs.append(
                Recommendation(
                    category="Action Verbs",
                    priority="High",
                    title="Replace Passive Phrasing with Dynamic Action Verbs",
                    message=(
                        f"Detected passive or weak phrases such as: {', '.join(set(weak_verbs_found))}. "
                        "Lead your bullet points with high-impact action verbs that highlight ownership and execution."
                    ),
                    evidence=f"Weak phrases detected: {', '.join(set(weak_verbs_found))}",
                    examples=[
                        "Instead of: 'Responsible for database queries and reports'",
                        "Write: 'Optimized PostgreSQL queries, reducing execution latency by 45%'"
                    ]
                )
            )
        else:
            recs.append(
                Recommendation(
                    category="Action Verbs",
                    priority="Low",
                    title="Strong Action Verbs Maintained",
                    message="Great job! Your resume demonstrates strong action-oriented language without excessive passive phrasing."
                )
            )

        # 3. Quantifiable Impact & Metrics
        if resume.metrics_count < 3:
            recs.append(
                Recommendation(
                    category="Quantifiable Metrics",
                    priority="High",
                    title="Incorporate Quantifiable Achievements",
                    message=(
                        f"Only {resume.metrics_count} quantifiable metric(s) were detected. "
                        "Recruiters and hiring managers favor resumes with concrete numbers, percentages, and scale indicators. "
                        "Adopt Google's X-Y-Z formula: 'Accomplished [X] as measured by [Y], by doing [Z]'."
                    ),
                    evidence="Low density of numerical metrics (percentages, dollar amounts, scale counts).",
                    examples=[
                        "Example: 'Decreased API response time by 38% by caching hot keys with Redis.'",
                        "Example: 'Led a team of 4 engineers to deliver the product 2 weeks ahead of schedule.'"
                    ]
                )
            )

        # 4. Resume Section Health & Completeness
        missing_sections = []
        for sec in ["summary", "skills", "experience", "education", "projects"]:
            if sec not in resume.sections or len(resume.sections[sec].strip()) < 20:
                missing_sections.append(sec.title())

        if missing_sections:
            recs.append(
                Recommendation(
                    category="Structure & ATS",
                    priority="Medium",
                    title="Add Missing Essential Resume Sections",
                    message=(
                        f"The following standard sections appear missing or minimal: {', '.join(missing_sections)}. "
                        "Applicant Tracking Systems rely on clear, conventional section headings to parse your history."
                    ),
                    evidence=f"Missing sections: {', '.join(missing_sections)}",
                    examples=["Use standard headers: 'Professional Experience', 'Technical Skills', 'Education', 'Projects'."]
                )
            )

        # 5. Contact Information Checks
        contact_issues = []
        if not resume.email:
            contact_issues.append("Email address not found")
        if not resume.phone:
            contact_issues.append("Phone number not found")
        if "linkedin" not in resume.links:
            contact_issues.append("LinkedIn profile link not detected")
        if "github" not in resume.links and any(k in job.clean_text.lower() for k in ["git", "github", "code", "software"]):
            contact_issues.append("GitHub profile link not detected for software role")

        if contact_issues:
            recs.append(
                Recommendation(
                    category="Structure & ATS",
                    priority="Medium",
                    title="Ensure Complete Contact Details & Professional Links",
                    message=f"Missing contact items: {', '.join(contact_issues)}. Place these prominently at the very top.",
                    evidence=f"Issues: {', '.join(contact_issues)}"
                )
            )

        # 6. Length & Density Guidance
        word_count = resume.word_count
        if word_count < 250:
            recs.append(
                Recommendation(
                    category="Structure & ATS",
                    priority="Medium",
                    title="Expand Resume Depth and Details",
                    message=f"Your resume contains approximately {word_count} words. Consider elaborating on technical responsibilities and project architecture."
                )
            )
        elif word_count > 900:
            recs.append(
                Recommendation(
                    category="Structure & ATS",
                    priority="Low",
                    title="Review Document Length and Brevity",
                    message=f"Your resume is {word_count} words long. Aim for a concise 1-2 page format highlighting the most relevant achievements for this specific role."
                )
            )

        # Sort recommendations by priority (High -> Medium -> Low)
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        return sorted(recs, key=lambda r: priority_order.get(r.priority, 3))

    def _find_weak_verbs(self, text: str) -> List[str]:
        """Scan text for passive or weak action phrasing."""
        found = []
        text_lower = text.lower()
        for weak in self.weak_verbs:
            if re.search(rf"\b{re.escape(weak)}\b", text_lower):
                found.append(weak)
        return found

