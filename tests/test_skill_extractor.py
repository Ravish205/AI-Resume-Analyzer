"""
Unit tests for SkillExtractor.
"""

import pytest
from src.skill_extractor import SkillExtractor, ExtractedSkill


@pytest.fixture
def extractor():
    return SkillExtractor()


def test_canonical_extraction(extractor):
    text = "We write code in Python3, manage deployments with K8s, and utilize C++ for fast modules."
    skills = extractor.extract_skills(text)
    names = {s.name for s in skills}

    assert "Python" in names
    assert "Kubernetes" in names
    assert "C++" in names


def test_multi_word_skill(extractor):
    text = "Developed REST APIs and applied Natural Language Processing techniques with Scikit-learn."
    skills = extractor.extract_skills(text)
    names = {s.name for s in skills}

    assert "REST APIs" in names
    assert "Natural Language Processing" in names
    assert "Scikit-learn" in names


def test_repeated_keywords_frequency(extractor):
    text = "Python is great. I love Python. Python microservices are fast."
    skills = extractor.extract_skills(text)
    py_skill = next((s for s in skills if s.name == "Python"), None)

    assert py_skill is not None
    assert py_skill.frequency >= 3


def test_categorize_skills(extractor):
    text = "Proficient in Python, PostgreSQL, Docker, and Agile communication."
    skills = extractor.extract_skills(text)
    categories = extractor.categorize_skills(skills)

    assert "Programming Languages" in categories
    assert "Databases & Storage" in categories
    assert "Cloud & DevOps" in categories
    assert "Python" in categories["Programming Languages"]
    assert "PostgreSQL" in categories["Databases & Storage"]


def test_compare_skills(extractor):
    resume_skills = {"Python", "SQL", "Docker", "Git"}
    job_skills = {"Python", "SQL", "Kubernetes", "REST APIs"}

    diff = extractor.compare_skills(resume_skills, job_skills)
    assert set(diff["matching"]) == {"Python", "SQL"}
    assert set(diff["missing"]) == {"Kubernetes", "REST APIs"}
    assert set(diff["additional"]) == {"Docker", "Git"}


def test_empty_text_returns_empty(extractor):
    assert extractor.extract_skills("") == []
    assert extractor.extract_skills("    ") == []

