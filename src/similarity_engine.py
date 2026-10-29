"""
Similarity Engine Module for AI Resume Analyzer.

Calculates semantic and lexical similarity between resume and job description text using:
- TF-IDF vectorization with unigram and bigram tokenization via scikit-learn
- Optional Sentence-Transformers semantic embedding similarity
- Section-level similarity breakdowns (e.g., Experience vs Job Responsibilities)
- Graceful pure-Python fallback for maximum resilience
"""

import math
import re
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Attempt scikit-learn imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Attempt sentence-transformers import
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class SimilarityEngine:
    """Calculates textual and semantic similarity scores."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._transformer_model = None

    def _get_transformer_model(self):
        """Lazy load sentence transformer if available."""
        if HAS_SENTENCE_TRANSFORMERS and self._transformer_model is None:
            try:
                self._transformer_model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer '{self.model_name}': {e}")
                self._transformer_model = None
        return self._transformer_model

    def compute_similarity(
        self,
        resume_text: str,
        job_text: str,
        use_transformer: bool = False
    ) -> float:
        """
        Compute normalized cosine similarity score between 0.0 and 1.0.
        
        Args:
            resume_text: Cleaned text from resume.
            job_text: Cleaned text from job description.
            use_transformer: If True and sentence-transformers is installed, use dense embeddings.
            
        Returns:
            Cosine similarity score strictly within [0.0, 1.0].
        """
        if not resume_text or not job_text:
            return 0.0

        clean_r = resume_text.strip()
        clean_j = job_text.strip()
        if not clean_r or not clean_j:
            return 0.0

        # Optional Dense Transformer Similarity
        if use_transformer and HAS_SENTENCE_TRANSFORMERS:
            model = self._get_transformer_model()
            if model is not None:
                try:
                    embeddings = model.encode([clean_r, clean_j], convert_to_tensor=True)
                    sim = float(util.cos_sim(embeddings[0], embeddings[1])[0][0])
                    # Bound to [0.0, 1.0]
                    return max(0.0, min(1.0, sim))
                except Exception as e:
                    logger.warning(f"Dense transformer similarity failed: {e}. Falling back to TF-IDF.")

        # Primary: Scikit-learn TF-IDF Vectorizer
        if HAS_SKLEARN:
            try:
                vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                    max_features=5000
                )
                tfidf_matrix = vectorizer.fit_transform([clean_r, clean_j])
                sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
                return max(0.0, min(1.0, sim))
            except Exception as e:
                logger.warning(f"TF-IDF calculation failed: {e}. Using pure-python fallback.")

        # Fallback: Pure-Python Token Cosine Similarity
        return self._pure_python_cosine_similarity(clean_r, clean_j)

    def compute_section_breakdown(
        self,
        sections: Dict[str, str],
        job_text: str
    ) -> Dict[str, float]:
        """
        Calculate individual similarity scores for each resume section against the job description.
        
        Returns:
            Dictionary mapping section name to similarity score [0.0 - 1.0].
        """
        breakdown: Dict[str, float] = {}
        for section_name, section_text in sections.items():
            if section_text and section_text.strip():
                score = self.compute_similarity(section_text, job_text)
                breakdown[section_name] = round(score, 3)
            else:
                breakdown[section_name] = 0.0
        return breakdown

    def _pure_python_cosine_similarity(self, text1: str, text2: str) -> float:
        """Lightweight pure-python cosine similarity implementation."""
        words1 = re.findall(r"\b[a-zA-Z0-9_+#.-]+\b", text1.lower())
        words2 = re.findall(r"\b[a-zA-Z0-9_+#.-]+\b", text2.lower())

        if not words1 or not words2:
            return 0.0

        vec1 = Counter(words1)
        vec2 = Counter(words2)

        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[x] * vec2[x] for x in intersection)

        sum1 = sum(v ** 2 for v in vec1.values())
        sum2 = sum(v ** 2 for v in vec2.values())
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        return max(0.0, min(1.0, float(numerator) / denominator))

