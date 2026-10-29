"""
AI Resume Analyzer — Streamlit Interactive Web Application.

Provides a modern dashboard for intelligent resume screening,
skill gap analysis, job alignment scoring, and PDF report generation.
"""

import io
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    DEGREE_HIERARCHY,
    SAMPLE_DATA_DIR,
    SCORE_TIERS,
    ScoringWeights,
)
from src.job_analyzer import JobAnalyzer, JobDescriptionData
from src.recommendation_engine import RecommendationEngine
from src.report_generator import ReportGenerator
from src.resume_parser import ResumeData, ResumeParser
from src.scoring_engine import MatchResult, ScoringEngine
from src.similarity_engine import SimilarityEngine
from src.skill_extractor import SkillExtractor

# Page configuration
st.set_page_config(
    page_title="AI Resume Analyzer — Intelligent Screening & Job Match",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 6px 0;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #94a3b8;
    }

    /* Skill Badge Styling */
    .skill-badge {
        display: inline-block;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .skill-match {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .skill-miss {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .skill-add {
        background-color: #e0f2fe;
        color: #075985;
        border: 1px solid #bae6fd;
    }

    /* Recommendation Card Styling */
    .rec-card {
        border-left: 4px solid #3b82f6;
        background-color: #ffffff;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .rec-high {
        border-left-color: #ef4444;
    }
    .rec-medium {
        border-left-color: #f59e0b;
    }
    .rec-low {
        border-left-color: #10b981;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_services():
    """Cache core analysis engines for responsive performance."""
    skill_extractor = SkillExtractor()
    similarity_engine = SimilarityEngine()
    resume_parser = ResumeParser()
    job_analyzer = JobAnalyzer(skill_extractor=skill_extractor)
    scoring_engine = ScoringEngine(
        similarity_engine=similarity_engine,
        skill_extractor=skill_extractor
    )
    recommendation_engine = RecommendationEngine()
    report_generator = ReportGenerator()

    return (
        resume_parser,
        job_analyzer,
        skill_extractor,
        similarity_engine,
        scoring_engine,
        recommendation_engine,
        report_generator,
    )


(
    resume_parser,
    job_analyzer,
    skill_extractor,
    similarity_engine,
    scoring_engine,
    recommendation_engine,
    report_generator,
) = get_services()


# Sidebar Navigation & Settings
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=64)
    st.title("Settings & Controls")
    st.markdown("Customize scoring weights and demonstration presets.")

    # Interview Demo Preset Button
    st.markdown("### 🎯 Quick Demo Loader")
    load_sample = st.button("Load Python Developer Demo", use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚖️ Scoring Weights")
    st.caption("Adjust weight distribution (must sum to 100%).")

    w_skills = st.slider("Required Skills Coverage", 0, 100, 50, 5, help="Weight given to exact matching technical skills.")
    w_semantic = st.slider("Semantic Similarity", 0, 100, 25, 5, help="Weight given to overall contextual and text alignment.")
    w_exp = st.slider("Experience Alignment", 0, 100, 15, 5, help="Weight given to years of experience and quantifiable metrics.")
    w_edu = st.slider("Education & Credentials", 0, 100, 10, 5, help="Weight given to matching degree level and certifications.")

    total_weight = w_skills + w_semantic + w_exp + w_edu
    if total_weight != 100:
        st.warning(f"⚠️ Total weight is {total_weight}%. Normalizing automatically to 100%.")

    weights = ScoringWeights(
        required_skills=w_skills / max(1, total_weight),
        semantic_similarity=w_semantic / max(1, total_weight),
        experience=w_exp / max(1, total_weight),
        education=w_edu / max(1, total_weight),
    )

    st.markdown("---")
    use_transformer = st.checkbox(
        "Enable Sentence-Transformers",
        value=False,
        help="Use dense neural embeddings for semantic similarity if installed."
    )

    st.markdown("---")
    st.markdown("### ℹ️ About ATS Screening")
    st.caption(
        "Commercial Applicant Tracking Systems parse resume sections, extract keywords, and check qualifications. "
        "This tool provides an estimated alignment score and actionable feedback, not a proprietary vendor score."
    )


# Session state initialization
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "job_text" not in st.session_state:
    st.session_state.job_text = ""
if "file_name" not in st.session_state:
    st.session_state.file_name = ""

# Handle Demo Loading
if load_sample:
    sample_resume_path = SAMPLE_DATA_DIR / "sample_resume.txt"
    sample_jd_path = SAMPLE_DATA_DIR / "sample_job_description.txt"

    if sample_resume_path.exists() and sample_jd_path.exists():
        st.session_state.resume_text = sample_resume_path.read_text(encoding="utf-8")
        st.session_state.job_text = sample_jd_path.read_text(encoding="utf-8")
        st.session_state.file_name = "sample_resume.txt"
        st.success("Loaded Python Developer resume and job description demo!")


# Main Application Header
st.title("AI Resume Analyzer")
st.markdown(
    "Intelligent resume screening, skill gap detection, and ATS optimization system. "
    "Compare your resume against any job description using local NLP and machine learning."
)

st.markdown("---")

# Step 1 & 2: Input Columns
col_resume, col_jd = st.columns(2)

with col_resume:
    st.subheader("1. Candidate Resume")
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        help="Maximum file size: 10MB."
    )

    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.getvalue()
            st.session_state.file_name = uploaded_file.name
            parsed_doc = resume_parser.parse(bytes_data, file_name=uploaded_file.name)
            st.session_state.resume_text = parsed_doc.clean_text
            st.success(f"Successfully parsed **{uploaded_file.name}** ({parsed_doc.word_count} words).")
            if parsed_doc.is_scanned_pdf:
                st.warning("⚠️ Scanned PDF detected: Text is image-based. OCR is recommended.")
        except Exception as e:
            st.error(f"Error reading resume: {e}")

    resume_text_input = st.text_area(
        "Or paste resume text directly:",
        value=st.session_state.resume_text,
        height=260,
        placeholder="Paste plain text resume here...",
        key="resume_area"
    )

with col_jd:
    st.subheader("2. Target Job Description")
    st.caption("Paste the target job description or requirements below.")
    job_text_input = st.text_area(
        "Job Description:",
        value=st.session_state.job_text,
        height=328,
        placeholder="Paste full job description including requirements and responsibilities...",
        key="job_area"
    )


# Update session state text
st.session_state.resume_text = resume_text_input
st.session_state.job_text = job_text_input


# Step 3: Interactive Skills Preview & Customization
if st.session_state.job_text.strip():
    with st.expander("🔍 Interactive Job Requirements & Skills Review", expanded=False):
        st.markdown("Review and refine extracted job requirements before running the full match analysis.")
        preview_job = job_analyzer.analyze(st.session_state.job_text)

        col_req, col_pref = st.columns(2)
        with col_req:
            st.markdown("**Required Skills Detected:**")
            selected_req_skills = st.multiselect(
                "Customize Required Skills:",
                options=sorted(list(set(preview_job.all_skills + preview_job.required_skills))),
                default=preview_job.required_skills,
                key="custom_req_skills"
            )
        with col_pref:
            st.markdown("**Preferred / Bonus Skills:**")
            selected_pref_skills = st.multiselect(
                "Customize Preferred Skills:",
                options=sorted(list(set(preview_job.all_skills + preview_job.preferred_skills))),
                default=preview_job.preferred_skills,
                key="custom_pref_skills"
            )

        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.caption(f"Required Experience: **{preview_job.experience_required_years:.1f} years**")
        with col_meta2:
            st.caption(f"Minimum Education: **{preview_job.required_education_name or 'Not explicitly required'}**")
else:
    selected_req_skills = None
    selected_pref_skills = None


# Step 4: Run Analysis Button
st.markdown("---")
run_analysis = st.button("🚀 Run Resume & Job Match Analysis", type="primary", use_container_width=True)

if run_analysis or (st.session_state.resume_text.strip() and st.session_state.job_text.strip()):
    if not st.session_state.resume_text.strip() or not st.session_state.job_text.strip():
        st.warning("Please provide both a Resume and a Job Description to proceed.")
        st.stop()

    with st.spinner("Analyzing resume content, extracting skills, and calculating match scores..."):
        # 1. Parse Resume
        file_name = st.session_state.file_name or "Resume.txt"
        resume_obj = resume_parser.parse(
            st.session_state.resume_text.encode("utf-8"),
            file_name=file_name
        )

        # 2. Parse Job Description
        job_obj = job_analyzer.analyze(st.session_state.job_text)
        if selected_req_skills is not None:
            job_obj.required_skills = selected_req_skills
        if selected_pref_skills is not None:
            job_obj.preferred_skills = selected_pref_skills

        # 3. Compute Match Score
        match_res: MatchResult = scoring_engine.calculate_match(
            resume=resume_obj,
            job=job_obj,
            weights=weights,
            use_transformer=use_transformer
        )

        # 4. Generate Recommendations
        recommendations = recommendation_engine.generate_recommendations(
            resume=resume_obj,
            job=job_obj,
            match_result=match_res
        )

    st.markdown("## 📊 Analysis Dashboard")

    # Big Score Banner
    score_col1, score_col2 = st.columns([1, 2])
    with score_col1:
        # Radial Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=match_res.overall_score,
            title={'text': "Compatibility Score", 'font': {'size': 20, 'color': '#1e293b'}},
            number={'suffix': "%", 'font': {'size': 44, 'color': '#0f172a'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                'bar': {'color': match_res.tier_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 50], 'color': '#fee2e2'},
                    {'range': [50, 65], 'color': '#fef3c7'},
                    {'range': [65, 80], 'color': '#dbeafe'},
                    {'range': [80, 100], 'color': '#dcfce7'}
                ],
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with score_col2:
        st.markdown(f"### Tier: <span style='color:{match_res.tier_color};'>{match_res.tier}</span>", unsafe_allow_html=True)
        st.markdown(f"**Assessment:** {match_res.tier_description}")
        st.markdown(f"**Target Role:** {job_obj.job_title or 'Target Job Description'}")
        if resume_obj.name:
            st.markdown(f"**Candidate:** {resume_obj.name}")

        # Diagnostic alerts
        if resume_obj.extraction_warnings:
            for w in resume_obj.extraction_warnings:
                st.info(f"ℹ️ {w}")

        st.caption(f"🔒 {match_res.disclaimer}")

    st.markdown("---")

    # 4 Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Skills Coverage ({int(weights.required_skills*100)}%)</div>
            <div class="metric-value">{match_res.skills_score:.1f}%</div>
            <div class="metric-sub">{len(match_res.matching_skills)} of {len(match_res.matching_skills) + len(match_res.missing_skills)} skills matched</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Semantic Match ({int(weights.semantic_similarity*100)}%)</div>
            <div class="metric-value">{match_res.semantic_score:.1f}%</div>
            <div class="metric-sub">Contextual TF-IDF & N-gram overlap</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Experience ({int(weights.experience*100)}%)</div>
            <div class="metric-value">{match_res.experience_score:.1f}%</div>
            <div class="metric-sub">{resume_obj.experience_years:.1f} yrs found (Req: {job_obj.experience_required_years:.1f} yrs)</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Education ({int(weights.education*100)}%)</div>
            <div class="metric-value">{match_res.education_score:.1f}%</div>
            <div class="metric-sub">{resume_obj.highest_degree_name or 'Degree verified'}</div>
        </div>
        """, unsafe_allow_html=True)

    # Visualizations Row
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        # Radar Chart of Components
        categories = ['Skills Coverage', 'Semantic Text', 'Experience', 'Education & Certs']
        values = [match_res.skills_score, match_res.semantic_score, match_res.experience_score, match_res.education_score]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.25)',
            line=dict(color='#3b82f6', width=2),
            name='Candidate Match'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title="Component Alignment Radar",
            height=300,
            margin=dict(l=40, r=40, t=40, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with v_col2:
        # Skill Counts Bar Chart
        counts_data = {
            "Category": ["Matching Skills", "Missing Required", "Additional Skills"],
            "Count": [len(match_res.matching_skills), len(match_res.missing_skills), len(match_res.additional_skills)]
        }
        df_counts = pd.DataFrame(counts_data)
        fig_bar = px.bar(
            df_counts,
            x="Count",
            y="Category",
            orientation="h",
            color="Category",
            color_discrete_map={
                "Matching Skills": "#10b981",
                "Missing Required": "#ef4444",
                "Additional Skills": "#3b82f6"
            },
            title="Skills Gap Distribution",
            text="Count"
        )
        fig_bar.update_layout(height=300, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Feature 4: Skills Gap Analysis Badges
    st.markdown("### 🧩 Skills Gap Analysis")

    tab_match, tab_miss, tab_add = st.tabs([
        f"✅ Matching Skills ({len(match_res.matching_skills)})",
        f"❌ Missing Required Skills ({len(match_res.missing_skills)})",
        f"➕ Additional Candidate Skills ({len(match_res.additional_skills)})"
    ])

    with tab_match:
        if match_res.matching_skills:
            st.markdown("Skills identified in **both** the candidate resume and the job description:")
            badges_html = "".join([f"<span class='skill-badge skill-match'>✓ {s}</span>" for s in match_res.matching_skills])
            st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.warning("No direct matching technical skills identified.")

    with tab_miss:
        if match_res.missing_skills:
            st.markdown("Required skills in the job description that were **not detected** in the resume:")
            badges_html = "".join([f"<span class='skill-badge skill-miss'>✕ {s}</span>" for s in match_res.missing_skills])
            st.markdown(badges_html, unsafe_allow_html=True)
            st.caption("💡 If you have genuine experience with these skills, add them to your resume and describe them in your project bullet points.")
        else:
            st.success("🎉 Excellent! 100% of required job skills were detected on your resume.")

    with tab_add:
        if match_res.additional_skills:
            st.markdown("Relevant skills present on your resume that were **not emphasized** in the job description:")
            badges_html = "".join([f"<span class='skill-badge skill-add'>+ {s}</span>" for s in match_res.additional_skills])
            st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.info("No additional skills detected.")

    st.markdown("---")

    # Section Completeness Checklist
    st.markdown("### 📋 Resume Section Completeness")
    sec_cols = st.columns(6)
    standard_sections = [
        ("Summary", "summary"),
        ("Skills", "skills"),
        ("Experience", "experience"),
        ("Education", "education"),
        ("Projects", "projects"),
        ("Certifications", "certifications")
    ]
    for idx, (label, key) in enumerate(standard_sections):
        with sec_cols[idx]:
            has_sec = key in resume_obj.sections and len(resume_obj.sections[key].strip()) > 15
            if has_sec:
                st.success(f"✓ {label}")
            else:
                st.warning(f"⚠️ {label}")

    st.markdown("---")

    # Feature 5: AI-Powered Recommendations
    st.markdown("### 💡 Actionable Improvement Recommendations")
    if recommendations:
        for rec in recommendations:
            priority_class = f"rec-{rec.priority.lower()}"
            st.markdown(f"""
            <div class="rec-card {priority_class}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong style="font-size: 1.05rem;">{rec.title}</strong>
                    <span style="font-size:0.75rem; font-weight:700; text-transform:uppercase; padding:2px 8px; border-radius:4px; background:#f1f5f9;">{rec.priority} Priority</span>
                </div>
                <p style="margin: 8px 0 4px 0; color: #334155; font-size: 0.92rem;">{rec.message}</p>
                {f'<p style="margin:0; font-size: 0.85rem; color:#64748b; font-style:italic;">{rec.examples[0]}</p>' if rec.examples else ''}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No major recommendations. Your resume aligns very well with this job posting.")

    st.markdown("---")

    # Feature 7: Downloadable Report
    st.markdown("### 📥 Download Analysis Report")
    col_dl_pdf, col_dl_txt = st.columns(2)

    with col_dl_pdf:
        pdf_bytes = report_generator.generate_pdf_report(
            resume=resume_obj,
            job=job_obj,
            match=match_res,
            recommendations=recommendations
        )
        st.download_button(
            label="📄 Download Full PDF Report",
            data=pdf_bytes,
            file_name=f"resume_match_report_{Path(file_name).stem}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with col_dl_txt:
        txt_content = report_generator.generate_txt_report(
            resume=resume_obj,
            job=job_obj,
            match=match_res,
            recommendations=recommendations
        )
        st.download_button(
            label="📝 Download Text Report (.txt)",
            data=txt_content,
            file_name=f"resume_match_report_{Path(file_name).stem}.txt",
            mime="text/plain",
            use_container_width=True
        )

