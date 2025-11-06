from __future__ import annotations

import re
from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


def simple_clean(text: str) -> str:
    """Minimal text normalizer: lowercase, strip punctuation/extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vectorizer(max_features: int = 20000, ngram_range: Tuple[int, int] = (1, 2)) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer using the simple_clean preprocessor."""
    return TfidfVectorizer(
        preprocessor=simple_clean,
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
    )

