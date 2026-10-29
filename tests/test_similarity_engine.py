"""
Unit tests for SimilarityEngine.
"""

import pytest
from src.similarity_engine import SimilarityEngine


@pytest.fixture
def sim_engine():
    return SimilarityEngine()


def test_identical_texts_high_similarity(sim_engine):
    text = "Senior Python Developer with expertise in building scalable REST APIs, Docker, and PostgreSQL databases."
    score = sim_engine.compute_similarity(text, text)
    assert score >= 0.95


def test_disjoint_texts_low_similarity(sim_engine):
    text1 = "Python developer specializing in backend APIs, machine learning pipelines, and cloud databases."
    text2 = "Pastry chef expert in sourdough baking, chocolate souffles, and French patisserie decoration."
    score = sim_engine.compute_similarity(text1, text2)
    assert score < 0.15


def test_empty_input_returns_zero(sim_engine):
    assert sim_engine.compute_similarity("", "Python developer") == 0.0
    assert sim_engine.compute_similarity("Software Engineer", "") == 0.0
    assert sim_engine.compute_similarity("", "") == 0.0


def test_score_bounded_between_zero_and_one(sim_engine):
    score = sim_engine.compute_similarity("FastAPI engineer", "FastAPI developer")
    assert 0.0 <= score <= 1.0


def test_pure_python_fallback(sim_engine):
    text1 = "Docker Kubernetes cloud architecture"
    text2 = "Docker Kubernetes cloud microservices"
    score = sim_engine._pure_python_cosine_similarity(text1, text2)
    assert 0.5 <= score <= 1.0


def test_section_similarity_breakdown(sim_engine):
    sections = {
        "summary": "Experienced Python developer building REST APIs",
        "education": "BS in Biology"
    }
    job = "Looking for an experienced Python developer to build scalable REST APIs"
    breakdown = sim_engine.compute_section_breakdown(sections, job)

    assert "summary" in breakdown
    assert "education" in breakdown
    assert breakdown["summary"] > breakdown["education"]

