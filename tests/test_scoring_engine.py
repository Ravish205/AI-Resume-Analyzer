"""
Unit tests for ScoringEngine and RecommendationEngine.
"""

import pytest
from config import ScoringWeights
from src.job_analyzer import JobAnalyzer
from src.recommendation_engine import RecommendationEngine
from src.resume_parser import ResumeParser
from src.scoring_engine import ScoringEngine


@pytest.fixture
def parser():
    return ResumeParser()


@pytest.fixture
def analyzer():
    return JobAnalyzer()


@pytest.fixture
def scoring_engine():
    return ScoringEngine()


@pytest.fixture
def rec_engine():
    return RecommendationEngine()


@pytest.fixture
def high_match_resume(parser):
    text = """Alex Smith
Python Backend Developer
Email: alex.smith@example.com | Phone: (555) 123-4567

SUMMARY
Python developer with 4 years of experience building REST APIs and data processing systems.

SKILLS
Python, SQL, PostgreSQL, REST APIs, Git, Docker, Machine Learning, Communication

EXPERIENCE
TechCorp | 2021 - Present
- Architected REST APIs using Python and FastAPI, increasing throughput by 40%.
- Integrated SQL databases with machine learning classification models.

EDUCATION
Bachelor of Science in Computer Science | 2020
"""
    return parser.parse(text.encode("utf-8"), file_name="high_match.txt")


@pytest.fixture
def low_match_resume(parser):
    text = """Robert Green
Chef & Restaurant Manager
Email: robert.green@example.com | Phone: (555) 321-7654

SUMMARY
Experienced culinary professional with 8 years in fine dining kitchen management.

SKILLS
Menu Design, Culinary Arts, Food Safety, Inventory Management, Staff Leadership

EXPERIENCE
The Grand Bistro | 2018 - Present
- Managed kitchen staff of 15 cooks and coordinated inventory ordering.
"""
    return parser.parse(text.encode("utf-8"), file_name="low_match.txt")


@pytest.fixture
def sample_job(analyzer):
    jd_text = """Job Title: Python Developer
Requirements:
- Bachelor's degree in Computer Science
- 3+ years of experience with Python
- Strong expertise in SQL and REST APIs
- Version control with Git
- Familiarity with Machine Learning fundamentals
- Good Communication skills
"""
    return analyzer.analyze(jd_text)


def test_high_match_outscores_low_match(scoring_engine, high_match_resume, low_match_resume, sample_job):
    high_result = scoring_engine.calculate_match(high_match_resume, sample_job)
    low_result = scoring_engine.calculate_match(low_match_resume, sample_job)

    assert high_result.overall_score > low_result.overall_score
    assert high_result.overall_score >= 70.0
    assert low_result.overall_score < 40.0


def test_score_bounds_and_structure(scoring_engine, high_match_resume, sample_job):
    result = scoring_engine.calculate_match(high_match_resume, sample_job)

    assert 0.0 <= result.overall_score <= 100.0
    assert 0.0 <= result.skills_score <= 100.0
    assert 0.0 <= result.semantic_score <= 100.0
    assert 0.0 <= result.experience_score <= 100.0
    assert 0.0 <= result.education_score <= 100.0
    assert result.tier in ["Excellent Match", "Good Match", "Moderate Match", "Low Match"]
    assert len(result.disclaimer) > 10


def test_custom_weights(scoring_engine, high_match_resume, sample_job):
    custom_weights = ScoringWeights(
        required_skills=0.70,
        semantic_similarity=0.10,
        experience=0.10,
        education=0.10
    )
    result = scoring_engine.calculate_match(high_match_resume, sample_job, weights=custom_weights)
    assert 0.0 <= result.overall_score <= 100.0
    assert result.weights_used.required_skills == 0.70


def test_missing_data_handled_gracefully(scoring_engine, parser, analyzer):
    # Resume with zero education or experience mentioned
    minimal_resume = parser.parse(b"Skills: Python, SQL", file_name="minimal.txt")
    job = analyzer.analyze("Looking for Python, SQL developer. Must have 5 years experience and MS degree.")

    result = scoring_engine.calculate_match(minimal_resume, job)
    # Should not crash, score should be non-negative and bounded
    assert 0.0 <= result.overall_score <= 100.0
    assert result.experience_score >= 0.0
    assert result.education_score >= 0.0


def test_recommendation_generation(rec_engine, high_match_resume, sample_job, scoring_engine):
    match = scoring_engine.calculate_match(high_match_resume, sample_job)
    recs = rec_engine.generate_recommendations(high_match_resume, sample_job, match)

    assert isinstance(recs, list)
    assert len(recs) > 0
    # Every recommendation must have category, priority, title, and guidance message
    for r in recs:
        assert r.category in ["Skills Gap", "Action Verbs", "Quantifiable Metrics", "Structure & ATS"]
        assert r.priority in ["High", "Medium", "Low"]
        assert len(r.title) > 0
        assert len(r.message) > 0

