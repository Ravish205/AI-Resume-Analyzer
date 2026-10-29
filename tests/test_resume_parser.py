"""
Unit tests for ResumeParser.
"""

import io
import pytest
from pathlib import Path
from src.resume_parser import ResumeParser, ResumeData


@pytest.fixture
def parser():
    return ResumeParser()


@pytest.fixture
def sample_text():
    return """Jane Doe
Senior Backend Engineer
Email: jane.doe@example.com | Phone: (555) 987-6543
LinkedIn: https://linkedin.com/in/janedoe | GitHub: https://github.com/janedoe

SUMMARY
Experienced software engineer with 5 years of experience in backend development.

TECHNICAL SKILLS
Python, SQL, PostgreSQL, Docker, Git, FastAPI, AWS

EXPERIENCE
CloudScale Systems | 2021 - Present
- Architected 10+ REST microservices in Python.
- Increased throughput by 45% and reduced database latency.

DataFlow Inc | 2019 - 2021
- Built data processing pipelines handling 500,000 daily events.

EDUCATION
Master of Science in Computer Science
Stanford University | 2019

CERTIFICATIONS
AWS Certified Solutions Architect
"""


def test_parse_valid_txt(parser, sample_text):
    data = parser.parse(sample_text.encode("utf-8"), file_name="resume.txt")
    assert isinstance(data, ResumeData)
    assert data.email == "jane.doe@example.com"
    assert data.phone == "(555) 987-6543"
    assert "linkedin" in data.links
    assert "github" in data.links
    assert data.name == "Jane Doe"
    assert "skills" in data.sections
    assert "experience" in data.sections
    assert data.highest_degree_level == 3  # Master's
    assert data.experience_years >= 4.0
    assert data.metrics_count >= 1


def test_empty_file_raises_error(parser):
    with pytest.raises(ValueError, match="empty"):
        parser.parse(b"", file_name="empty.txt")


def test_unsupported_extension(parser):
    with pytest.raises(ValueError, match="Unsupported file extension"):
        parser.parse(b"Some text", file_name="resume.exe")


def test_scanned_pdf_heuristic(parser):
    # A dummy PDF with pages but no text
    import pymupdf as fitz
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.write()
    doc.close()

    data = parser.parse(pdf_bytes, file_name="scanned.pdf")
    assert data.is_scanned_pdf is True
    assert any("scanned image" in w for w in data.extraction_warnings)


def test_docx_parsing(parser, sample_text):
    import docx
    doc = docx.Document()
    for line in sample_text.split("\n"):
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    docx_bytes = buf.getvalue()

    data = parser.parse(docx_bytes, file_name="resume.docx")
    assert data.name == "Jane Doe"
    assert data.email == "jane.doe@example.com"
    assert "Python" in data.clean_text


def test_email_extraction(parser):
    text = "Please reach out to dev.user-name@domain.co.uk or alternate@sub.domain.org."
    assert parser.extract_email(text) == "dev.user-name@domain.co.uk"
    assert parser.extract_email("No email present here") is None


def test_phone_extraction(parser):
    assert parser.extract_phone("Call me at +1 (555) 123-4567 anytime") is not None
    assert parser.extract_phone("Office: 415.555.2671") is not None
    assert parser.extract_phone("No phone numbers 123") is None


def test_metrics_counter(parser):
    text = "Improved load speed by 35%, scaled cluster to 50,000 users, and saved $120k annually."
    count = parser.count_metrics(text)
    assert count >= 3

