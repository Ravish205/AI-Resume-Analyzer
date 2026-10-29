"""
Resume Parser Module for AI Resume Analyzer.

Extracts structured information from PDF, DOCX, and TXT resumes including:
- Contact details (name, email, phone, web links)
- Structural sections (summary, experience, education, skills, projects, certifications)
- Education credentials & degrees
- Estimated total years of experience
- Quantifiable impact metrics
- Diagnostics for scanned documents requiring OCR
"""

import io
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Attempt spaCy import with graceful fallback
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
except ImportError:
    nlp = None

# Attempt PyMuPDF (pymupdf / fitz) import
try:
    import pymupdf as fitz
    HAS_PYMUPDF = True
except ImportError:
    try:
        import fitz
        HAS_PYMUPDF = True
    except ImportError:
        HAS_PYMUPDF = False

# Attempt python-docx import
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

from config import (
    DEGREE_HIERARCHY,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    SECTION_PATTERNS,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)


@dataclass
class ResumeData:
    """Structured representation of parsed resume data."""
    raw_text: str = ""
    clean_text: str = ""
    file_name: str = ""
    file_type: str = ""
    file_size_bytes: int = 0
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    links: Dict[str, str] = field(default_factory=dict)
    sections: Dict[str, str] = field(default_factory=dict)
    education: List[Dict[str, Any]] = field(default_factory=list)
    highest_degree_level: int = 0
    highest_degree_name: Optional[str] = None
    experience_years: float = 0.0
    metrics_count: int = 0
    word_count: int = 0
    is_scanned_pdf: bool = False
    extraction_warnings: List[str] = field(default_factory=list)


class ResumeParser:
    """Robust parser for extracting content and metadata from resume files."""

    def __init__(self) -> None:
        self.section_patterns = SECTION_PATTERNS

    def parse(
        self,
        file_input: Union[str, Path, bytes, io.BytesIO],
        file_name: str = ""
    ) -> ResumeData:
        """
        Parse resume from a file path, raw bytes, or BytesIO buffer.
        
        Args:
            file_input: File path string/Path, bytes, or BytesIO stream.
            file_name: Name of the file (required for format detection if bytes).
            
        Returns:
            ResumeData populated with extracted text and structured attributes.
            
        Raises:
            ValueError: If file format is unsupported, file is empty or exceeds size limit.
        """
        warnings: List[str] = []
        file_bytes: bytes
        
        # Handle file input types
        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            file_name = file_name or path.name
            file_bytes = path.read_bytes()
        elif isinstance(file_input, io.BytesIO):
            file_bytes = file_input.getvalue()
        elif isinstance(file_input, bytes):
            file_bytes = file_input
        else:
            raise ValueError(f"Unsupported file input type: {type(file_input)}")

        file_size = len(file_bytes)
        if file_size == 0:
            raise ValueError("The provided file is empty (0 bytes).")
        if file_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File size exceeds maximum limit of {MAX_FILE_SIZE_MB}MB ({file_size} bytes)."
            )

        # Detect extension
        ext = Path(file_name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension '{ext}'. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        raw_text = ""
        is_scanned = False

        if ext == ".pdf":
            raw_text, is_scanned = self._extract_pdf_text(file_bytes)
            if is_scanned:
                warnings.append(
                    "This PDF appears to be a scanned image or contains non-extractable text. "
                    "OCR (Optical Character Recognition) is recommended for best results."
                )
        elif ext == ".docx":
            raw_text = self._extract_docx_text(file_bytes)
        elif ext == ".txt":
            raw_text = self._extract_txt_text(file_bytes)

        clean_text = self._clean_text(raw_text)
        word_count = len(clean_text.split())

        if word_count < 10 and not is_scanned:
            warnings.append("Extracted text is very short. Please verify document formatting.")

        # Extract structured entities
        email = self.extract_email(clean_text)
        phone = self.extract_phone(clean_text)
        links = self.extract_links(clean_text)
        name = self.extract_name(clean_text, email)
        sections = self.segment_sections(clean_text)
        education_data, highest_level, highest_degree = self.extract_education(clean_text)
        exp_years = self.calculate_experience_years(clean_text)
        metrics_count = self.count_metrics(clean_text)

        return ResumeData(
            raw_text=raw_text,
            clean_text=clean_text,
            file_name=file_name,
            file_type=ext,
            file_size_bytes=file_size,
            name=name,
            email=email,
            phone=phone,
            links=links,
            sections=sections,
            education=education_data,
            highest_degree_level=highest_level,
            highest_degree_name=highest_degree,
            experience_years=exp_years,
            metrics_count=metrics_count,
            word_count=word_count,
            is_scanned_pdf=is_scanned,
            extraction_warnings=warnings,
        )

    def _extract_pdf_text(self, pdf_bytes: bytes) -> Tuple[str, bool]:
        """Extract text from PDF bytes using PyMuPDF (fitz) with scanned PDF detection."""
        if not HAS_PYMUPDF:
            # Fallback naive decoder if fitz is somehow missing
            text = pdf_bytes.decode("latin-1", errors="ignore")
            return text, False

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            extracted_pages: List[str] = []
            total_pages = len(doc)
            
            for page in doc:
                page_text = page.get_text("text")
                extracted_pages.append(page_text)
                
            combined = "\n".join(extracted_pages).strip()
            
            # Scanned PDF heuristic: document has pages but nearly no text extracted
            is_scanned = (total_pages > 0 and len(combined.replace(" ", "")) < 60)
            return combined, is_scanned
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise ValueError(f"Failed to read PDF document: {e}")

    def _extract_docx_text(self, docx_bytes: bytes) -> str:
        """Extract text from DOCX bytes including paragraphs and tables."""
        if not HAS_DOCX:
            # Fallback naive extraction from XML
            import zipfile
            import xml.etree.ElementTree as ET
            with zipfile.ZipFile(io.BytesIO(docx_bytes)) as z:
                xml_content = z.read("word/document.xml")
                root = ET.fromstring(xml_content)
                text_elems = root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')
                return " ".join(t.text for t in text_elems if t.text)

        try:
            doc = docx.Document(io.BytesIO(docx_bytes))
            lines: List[str] = []
            for p in doc.paragraphs:
                if p.text.strip():
                    lines.append(p.text.strip())
            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_text:
                        lines.append(" | ".join(row_text))
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            raise ValueError(f"Failed to read DOCX document: {e}")

    def _extract_txt_text(self, txt_bytes: bytes) -> str:
        """Extract text from plain text bytes attempting multiple encodings."""
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return txt_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        return txt_bytes.decode("utf-8", errors="replace")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize whitespace, bullet points, and formatting."""
        if not text:
            return ""
        # Normalize various bullet characters
        text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219\u00B7\u25AA\u25CF]", "\n* ", text)
        # Normalize non-breaking spaces
        text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple horizontal whitespaces
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract_email(self, text: str) -> Optional[str]:
        """Extract the most plausible candidate email address."""
        pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
        matches = re.findall(pattern, text)
        if matches:
            # Exclude generic emails if possible
            for email in matches:
                lower = email.lower()
                if not any(k in lower for k in ["example.com", "yoursite.com", "domain.com"]):
                    return email
            return matches[0]
        return None

    def extract_phone(self, text: str) -> Optional[str]:
        """Extract phone numbers supporting standard domestic and international formats."""
        pattern = r"(?:(?:\+?\d{1,3}[\s.-]*)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4})"
        match = re.search(pattern, text)
        if match:
            raw = match.group(0).strip()
            digits = re.sub(r"\D", "", raw)
            if 10 <= len(digits) <= 15:
                return raw
        return None

    def extract_links(self, text: str) -> Dict[str, str]:
        """Extract LinkedIn, GitHub, and portfolio URLs."""
        links: Dict[str, str] = {}
        linkedin_match = re.search(r"https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+/?", text, re.IGNORECASE)
        if linkedin_match:
            links["linkedin"] = linkedin_match.group(0)
        
        github_match = re.search(r"https?://(?:www\.)?github\.com/[A-Za-z0-9_-]+/?", text, re.IGNORECASE)
        if github_match:
            links["github"] = github_match.group(0)

        portfolio_match = re.search(r"https?://[A-Za-z0-9.-]+\.[a-z]{2,}/[A-Za-z0-9._~:/?#[\]@!$&'()*+,;=-]*", text, re.IGNORECASE)
        if portfolio_match and "linkedin" not in portfolio_match.group(0) and "github" not in portfolio_match.group(0):
            links["portfolio"] = portfolio_match.group(0)
            
        return links

    def extract_name(self, text: str, email: Optional[str] = None) -> Optional[str]:
        """
        Extract candidate name using NLP entity recognition and heuristic patterns.
        
        Avoids false positives like 'Curriculum Vitae', 'Resume', or job titles.
        """
        disallowed = {
            "resume", "curriculum", "vitae", "cv", "profile", "contact",
            "summary", "developer", "engineer", "software", "experience",
            "education", "skills", "phone", "email", "address", "page"
        }

        # Strategy 1: spaCy NER Person check on the header (first 5 lines)
        header_lines = [line.strip() for line in text.split("\n")[:8] if line.strip()]
        if nlp is not None:
            header_text = "\n".join(header_lines)
            try:
                doc = nlp(header_text)
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        cand = ent.text.strip()
                        words = cand.lower().split()
                        if 1 < len(words) <= 4 and not any(w in disallowed for w in words):
                            # Validate characters are alphabetical
                            if re.match(r"^[A-Za-z\s.'-]+$", cand):
                                return cand
            except Exception:
                pass

        # Strategy 2: Structural heuristic - first non-empty line that looks like a name
        for line in header_lines:
            # Strip titles/credentials like Ph.D., PMP
            clean_line = re.sub(r",?\s*(?:phd|msc|bs|pmp|cpa|mba)\b", "", line, flags=re.I).strip()
            words = clean_line.split()
            if 2 <= len(words) <= 4:
                lower_words = [w.lower() for w in words]
                if not any(w in disallowed for w in lower_words):
                    if all(w[0].isupper() for w in words if w):
                        if re.match(r"^[A-Za-z\s.'-]+$", clean_line):
                            return clean_line

        # Strategy 3: Email username derivation fallback
        if email:
            prefix = email.split("@")[0]
            parts = re.split(r"[._-]", prefix)
            if len(parts) >= 2 and all(p.isalpha() and len(p) > 1 for p in parts[:2]):
                return f"{parts[0].capitalize()} {parts[1].capitalize()}"

        return None

    def segment_sections(self, text: str) -> Dict[str, str]:
        """Segment resume into sections based on recognized headings."""
        sections: Dict[str, str] = {}
        lines = text.split("\n")
        
        current_section = "summary"
        section_lines: Dict[str, List[str]] = {k: [] for k in self.section_patterns.keys()}
        section_lines["summary"] = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if this line is a section heading
            clean_heading = re.sub(r"[:\-_|*#]", "", line_str).strip().lower()
            found_section = None

            if len(clean_heading.split()) <= 4:
                for sec, patterns in self.section_patterns.items():
                    if clean_heading in patterns or any(clean_heading.startswith(p) for p in patterns):
                        found_section = sec
                        break

            if found_section:
                current_section = found_section
            else:
                section_lines.setdefault(current_section, []).append(line_str)

        for sec, lns in section_lines.items():
            if lns:
                sections[sec] = "\n".join(lns).strip()

        return sections

    def extract_education(self, text: str) -> Tuple[List[Dict[str, Any]], int, Optional[str]]:
        """
        Extract academic degrees, universities, and graduation years.
        
        Returns:
            Tuple of (education_list, highest_degree_rank, highest_degree_name)
        """
        education_records: List[Dict[str, Any]] = []
        highest_rank = 0
        highest_degree_name: Optional[str] = None

        degree_regexes = [
            (r"\b(ph\.?d|doctor of philosophy|doctorate)\b", "PhD / Doctorate", 4),
            (r"\b(m\.?s\.?|master of science|master of arts|m\.?tech|mba|masters?)\b", "Master's Degree", 3),
            (r"\b(b\.?s\.?|bachelor of science|bachelor of arts|b\.?tech|b\.?e\.?|bachelors?)\b", "Bachelor's Degree", 2),
            (r"\b(associate of science|associate degree|associate of arts|diploma)\b", "Associate / Diploma", 1),
        ]

        text_lower = text.lower()
        for pattern, display_name, rank in degree_regexes:
            if re.search(pattern, text_lower):
                if rank > highest_rank:
                    highest_rank = rank
                    highest_degree_name = display_name
                education_records.append({"degree": display_name, "level": rank})

        # Year search
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
        if years and education_records:
            education_records[0]["grad_year"] = max(years)

        return education_records, highest_rank, highest_degree_name

    def calculate_experience_years(self, text: str) -> float:
        """
        Estimate candidate's total years of experience using date ranges and explicit mentions.
        """
        # Look for explicit mentions like '5+ years of experience' or '3.5 years in software engineering'
        explicit_match = re.search(r"(\d+(?:\.\d+)?)\+?\s*years?(?:\s+of)?(?:\s+relevant)?\s+experience", text, re.IGNORECASE)
        if explicit_match:
            try:
                return float(explicit_match.group(1))
            except ValueError:
                pass

        # Look for date ranges: e.g. '2019 - 2023', 'Jan 2020 - Present', '03/2018 - 05/2021'
        year_range_pattern = r"\b(20[0-2][0-9]|19[8-9][0-9])\s*(?:-|–|—|to)\s*(present|current|20[0-2][0-9])\b"
        ranges = re.findall(year_range_pattern, text, re.IGNORECASE)

        total_years = 0.0
        current_year = 2026

        if ranges:
            seen_years: set = set()
            for start, end in ranges:
                try:
                    s_yr = int(start)
                    e_yr = current_year if end.lower() in ("present", "current") else int(end)
                    if 1980 <= s_yr <= current_year and s_yr <= e_yr <= current_year + 1:
                        for y in range(s_yr, e_yr):
                            seen_years.add(y)
                except ValueError:
                    continue
            total_years = float(len(seen_years))

        # Default fallback reasonable bounds
        return min(max(total_years, 0.0), 40.0)

    def count_metrics(self, text: str) -> int:
        """
        Count occurrences of quantifiable metrics (percentages, dollar amounts, metrics multipliers).
        """
        patterns = [
            r"\b\d+(?:\.\d+)?%",                                # Percentages (e.g. 35%, 99.9%)
            r"\$\s*\d+(?:,\d+)*(?:\.\d+)?\s*[kKmMbB]?",        # Dollar figures ($120k, $5M)
            r"\b\d+x\b",                                       # Multipliers (e.g. 5x faster, 10x)
            r"\b\d{1,3}(?:,\d{3})+\b",                         # Large counts (e.g. 50,000)
            r"\b\d+\+?\s*(?:users|clients|projects|queries|requests|endpoints|services|records)\b"
        ]
        count = 0
        for pat in patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            count += len(matches)
        return count
