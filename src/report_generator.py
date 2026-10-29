"""
Report Generator Module for AI Resume Analyzer.

Generates professional downloadable analysis reports in:
- Clean PDF format using fpdf2
- Formatted Text / Markdown format

Protects candidate privacy by sanitizing personal information
and includes comprehensive scoring diagnostics and recommendations.
"""

import datetime
from typing import List, Optional

# Attempt fpdf2 import
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

from src.job_analyzer import JobDescriptionData
from src.recommendation_engine import Recommendation
from src.resume_parser import ResumeData
from src.scoring_engine import MatchResult


class ReportGenerator:
    """Generates sanitized, professional resume match reports."""

    def __init__(self) -> None:
        pass

    def generate_txt_report(
        self,
        resume: ResumeData,
        job: JobDescriptionData,
        match: MatchResult,
        recommendations: List[Recommendation]
    ) -> str:
        """
        Generate a structured plain-text report.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sanitized_email = self._sanitize_email(resume.email)
        sanitized_name = resume.name or "Candidate"

        lines = [
            "=" * 78,
            "               AI RESUME ANALYZER - JOB MATCH REPORT",
            "=" * 78,
            f"Generated on      : {timestamp}",
            f"Candidate Name    : {sanitized_name}",
            f"Candidate Contact : {sanitized_email}",
            f"Target Role/Title : {job.job_title or 'Target Job Description'}",
            "-" * 78,
            "1. EXECUTIVE SUMMARY & COMPATIBILITY SCORE",
            "-" * 78,
            f"Overall Match Score: {match.overall_score} / 100  [{match.tier.upper()}]",
            f"Tier Assessment    : {match.tier_description}",
            "",
            "Component Breakdown:",
            f"  - Skills Coverage (50% wt)       : {match.skills_score:.1f}%",
            f"  - Semantic Similarity (25% wt)   : {match.semantic_score:.1f}%",
            f"  - Experience Alignment (15% wt)  : {match.experience_score:.1f}%",
            f"  - Education & Certs (10% wt)     : {match.education_score:.1f}%",
            "",
            "-" * 78,
            "2. SKILLS GAP ANALYSIS",
            "-" * 78,
            f"Matching Skills ({len(match.matching_skills)}):",
            f"  {', '.join(match.matching_skills) if match.matching_skills else 'None identified'}",
            "",
            f"Missing Required Skills ({len(match.missing_skills)}):",
            f"  {', '.join(match.missing_skills) if match.missing_skills else 'None - Full coverage!'}",
            "",
            f"Additional Resume Skills ({len(match.additional_skills)}):",
            f"  {', '.join(match.additional_skills) if match.additional_skills else 'None identified'}",
            "",
            "-" * 78,
            "3. ACTIONABLE IMPROVEMENT RECOMMENDATIONS",
            "-" * 78
        ]

        if not recommendations:
            lines.append("No critical improvement recommendations identified.")
        else:
            for idx, rec in enumerate(recommendations, 1):
                lines.append(f"[{rec.priority.upper()} PRIORITY] {idx}. {rec.title} ({rec.category})")
                lines.append(f"  Guidance: {rec.message}")
                if rec.examples:
                    for ex in rec.examples:
                        lines.append(f"  * {ex}")
                lines.append("")

        lines.extend([
            "-" * 78,
            "4. METHODOLOGY & LIMITATIONS DISCLAIMER",
            "-" * 78,
            match.disclaimer,
            "",
            "Limitations:",
            "- This automated screening tool utilizes local NLP, TF-IDF vectorization, and rule taxonomies.",
            "- It evaluates keyword coverage, phrasing impact, and semantic alignment.",
            "- It is NOT an official guarantee of hiring and does not reproduce proprietary commercial ATS scoring.",
            "- Genuine human evaluation and portfolio demonstration remain paramount.",
            "=" * 78,
        ])

        return "\n".join(lines)

    def generate_pdf_report(
        self,
        resume: ResumeData,
        job: JobDescriptionData,
        match: MatchResult,
        recommendations: List[Recommendation]
    ) -> bytes:
        """
        Generate a professional PDF report.
        """
        if not HAS_FPDF:
            txt = self.generate_txt_report(resume, job, match, recommendations)
            return txt.encode("utf-8")

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_margins(15, 15, 15)

        # Title Banner
        pdf.set_fill_color(30, 41, 59)  # Dark slate
        pdf.rect(0, 0, 210, 26, "F")
        pdf.set_xy(15, 6)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(pdf.epw, 7, "AI RESUME ANALYZER REPORT", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=9)
        timestamp = datetime.datetime.now().strftime("%B %d, %Y - %H:%M UTC")
        pdf.cell(pdf.epw, 5, self._clean_pdf_text(f"Automated Screening & Job Alignment Audit | {timestamp}"), new_x="LMARGIN", new_y="NEXT")

        pdf.set_xy(15, 32)
        pdf.set_text_color(30, 41, 59)

        # Overview Section
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(pdf.epw, 7, "Candidate & Target Overview", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        sanitized_name = self._clean_pdf_text(resume.name or "Candidate")
        sanitized_email = self._clean_pdf_text(self._sanitize_email(resume.email))
        target_role = self._clean_pdf_text(job.job_title or "Job Description Target")

        pdf.cell(45, 6, "Candidate Name:", border=0)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, sanitized_name, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", size=10)
        pdf.cell(45, 6, "Contact Identifier:", border=0)
        pdf.cell(0, 6, sanitized_email, new_x="LMARGIN", new_y="NEXT")

        pdf.cell(45, 6, "Target Role / Job:", border=0)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, target_role, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        # Overall Score Callout Box
        pdf.set_fill_color(241, 245, 249)
        pdf.set_draw_color(203, 213, 225)
        y_box = pdf.get_y()
        pdf.rect(15, y_box, pdf.epw, 24, "DF")

        pdf.set_xy(20, y_box + 2)
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(40, 10, f"{match.overall_score}%", border=0)

        pdf.set_xy(65, y_box + 3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(pdf.epw - 55, 5, self._clean_pdf_text(f"Overall Compatibility: {match.tier.upper()}"), new_x="LEFT", new_y="NEXT")
        pdf.set_xy(65, y_box + 9)
        pdf.set_font("Helvetica", size=8)
        pdf.set_text_color(71, 85, 105)
        pdf.multi_cell(pdf.epw - 55, 4, self._clean_pdf_text(match.tier_description))

        pdf.set_xy(15, y_box + 28)

        # Score Breakdown
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(pdf.epw, 7, "Component Score Breakdown", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=9)

        breakdowns = [
            ("Required Skills Coverage (50%)", f"{match.skills_score:.1f}%"),
            ("Semantic Text Similarity (25%)", f"{match.semantic_score:.1f}%"),
            ("Experience Relevance (15%)", f"{match.experience_score:.1f}%"),
            ("Education & Certifications (10%)", f"{match.education_score:.1f}%")
        ]
        for name, val in breakdowns:
            pdf.cell(100, 6, name, border="B")
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(pdf.epw - 100, 6, val, border="B", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=9)
        pdf.ln(3)

        # Skills Gap Section
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(pdf.epw, 7, "Skills Gap Analysis", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(16, 185, 129)  # Green
        pdf.cell(pdf.epw, 5, f"[+] Matching Skills ({len(match.matching_skills)}):", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=8)
        pdf.set_text_color(51, 65, 85)
        match_text = ", ".join(match.matching_skills) if match.matching_skills else "None explicitly detected."
        pdf.multi_cell(pdf.epw, 4, self._clean_pdf_text(match_text))
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(225, 29, 72)  # Red
        pdf.cell(pdf.epw, 5, f"[-] Missing Required Skills ({len(match.missing_skills)}):", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=8)
        pdf.set_text_color(51, 65, 85)
        miss_text = ", ".join(match.missing_skills) if match.missing_skills else "None - 100% target skills present!"
        pdf.multi_cell(pdf.epw, 4, self._clean_pdf_text(miss_text))
        pdf.ln(3)

        # Recommendations Section
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(pdf.epw, 7, "Key Improvement Recommendations", new_x="LMARGIN", new_y="NEXT")

        for idx, rec in enumerate(recommendations[:4], 1):
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(30, 41, 59)
            pdf.cell(pdf.epw, 5, self._clean_pdf_text(f"{idx}. {rec.title} [{rec.priority.upper()}]"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=8)
            pdf.set_text_color(71, 85, 105)
            pdf.multi_cell(pdf.epw, 4, self._clean_pdf_text(rec.message))
            if rec.examples:
                pdf.set_font("Helvetica", "I", 8)
                pdf.multi_cell(pdf.epw, 4, self._clean_pdf_text(f"   {rec.examples[0]}"))
            pdf.ln(1)

        pdf.ln(2)
        # Disclaimer
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.multi_cell(pdf.epw, 4, self._clean_pdf_text(f"Disclaimer: {match.disclaimer}"))

        output = pdf.output()
        if isinstance(output, (bytes, bytearray)):
            return bytes(output)
        return output.encode("latin-1", errors="replace")

    def _sanitize_email(self, email: Optional[str]) -> str:
        """Sanitize email for privacy: e.g. 'john.doe@gmail.com' -> 'j***e@gmail.com'."""
        if not email:
            return "Not Provided"
        parts = email.split("@")
        if len(parts) != 2:
            return "***@***.com"
        user, domain = parts
        if len(user) <= 2:
            masked_user = user[0] + "***"
        else:
            masked_user = user[0] + "***" + user[-1]
        return f"{masked_user}@{domain}"

    def _clean_pdf_text(self, text: str) -> str:
        """Replace non-ASCII / Unicode characters with standard equivalents for standard Helvetica font."""
        if not text:
            return ""
        replacements = {
            "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u2022": "*", "\u25cf": "*",
            "\u2026": "...", "\u00a0": " "
        }
        for orig, rep in replacements.items():
            text = text.replace(orig, rep)
        return text.encode("latin-1", errors="replace").decode("latin-1")
