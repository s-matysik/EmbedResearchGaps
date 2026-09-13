"""Keyword normalisation, lexical filters and corpus-level TF-IDF salience."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import GapConfig

__all__ = [
    "normalize_keyword",
    "token_count",
    "split_author_keywords",
    "is_methodological_term",
    "keyword_frequencies",
    "core_domain_terms",
    "TfidfSalience",
    "compute_tfidf_salience",
]

#: Separators accepted in author-keyword fields of bibliographic exports.
_KEYWORD_SEPARATORS = (";", "|")


def normalize_keyword(keyword: str) -> str:
    """Lower-case a keyword and collapse internal whitespace.

    Multi-word terms are kept as atomic units -- no stemming and no synonym
    merging, because synonymy is handled implicitly by embedding similarity.

    >>> normalize_keyword("  Purchase   Intention ")
    'purchase intention'
    """
    return " ".join(str(keyword).lower().strip().split())


def token_count(keyword: str) -> int:
    """Number of whitespace-separated tokens in a normalised keyword."""
    normalized = normalize_keyword(keyword)
    return len(normalized.split()) if normalized else 0


def split_author_keywords(value: object) -> list[str]:
    """Split a raw author-keyword field into a list of keywords.

    Accepts the semicolon-delimited strings used by Scopus and Web of Science
    exports as well as values that are already lists.  Empty and missing
    fields yield an empty list.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        items: Iterable[str] = [str(v) for v in value]
    else:
        text = str(value)
        for separator in _KEYWORD_SEPARATORS[1:]:
            text = text.replace(separator, _KEYWORD_SEPARATORS[0])
        items = text.split(_KEYWORD_SEPARATORS[0])
    return [kw.strip() for kw in items if str(kw).strip()]


def is_methodological_term(keyword: str, config: GapConfig | None = None) -> bool:
    """Whether a keyword names a method, tool or study design rather than a topic.

    A term matches when it equals one of the configured tool acronyms, or when
    it contains (or is contained in) one of the configured phrases.

    >>> is_methodological_term("structural equation model")
    True
    >>> is_methodological_term("customer engagement")
    False
    """
    config = config or GapConfig()
    normalized = normalize_keyword(keyword)
    if not normalized:
        return False
    if normalized in config.methodological_tools:
        return True
    for pattern in config.methodological_patterns:
        if pattern in normalized or normalized in pattern:
            return True
    return False


def keyword_frequencies(keyword_lists: Iterable[Sequence[str]]) -> Counter:
    """Document frequency of every normalised keyword in the corpus.

    A keyword repeated inside one record is counted once for that record, so
    the values are document frequencies rather than raw occurrence counts.
    """
    counter: Counter = Counter()
    for keywords in keyword_lists:
        # dict.fromkeys keeps the insertion order of the Counter stable, so
        # that ``most_common`` breaks ties identically in every process.
        for normalized in dict.fromkeys(normalize_keyword(kw) for kw in keywords):
            if normalized:
                counter[normalized] += 1
    return counter


def core_domain_terms(
    frequencies: Mapping[str, int],
    n_documents: int,
    share: float = 0.50,
) -> set[str]:
    """Keywords present in more than ``share`` of the documents.

    These are the labels of the field itself (for a gamification-in-marketing
    corpus: "gamification", "marketing") and are never research gaps.
    """
    if n_documents <= 0:
        return set()
    threshold = n_documents * share
    return {kw for kw, freq in frequencies.items() if freq > threshold}


class TfidfSalience:
    """Corpus-level TF-IDF salience of author keywords.

    Each record is represented as the concatenation of its normalised author
    keywords; an n-gram TF-IDF matrix is fitted over those pseudo-documents
    and the mean column weight is used as the corpus-level salience of a term
    (equation 6 of the AMCIS paper).  Keywords absent from the vocabulary are
    resolved by a four-stage back-off: exact match, substring containment,
    token overlap of at least one half, and finally a small floor value.
    """

    #: Value assigned when no vocabulary entry can be matched at all.
    FALLBACK = 0.001

    def __init__(self, term_scores: Mapping[str, float], match_types: Mapping[str, str]):
        self.term_scores: dict[str, float] = dict(term_scores)
        self.match_types: dict[str, str] = dict(match_types)

    def __getitem__(self, keyword: str) -> float:
        return self.get(keyword)

    def __len__(self) -> int:
        return len(self.term_scores)

    def get(self, keyword: str, default: float | None = None) -> float:
        """Salience of ``keyword``; ``FALLBACK`` when it was never resolved."""
        normalized = normalize_keyword(keyword)
        if normalized in self.term_scores:
            return self.term_scores[normalized]
        return self.FALLBACK if default is None else default

    @property
    def max_score(self) -> float:
        return max(self.term_scores.values()) if self.term_scores else 1.0

    def match_statistics(self) -> dict[str, int]:
        """How many keywords were resolved by each back-off stage."""
        return dict(Counter(self.match_types.values()))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "keyword": list(self.term_scores),
                "tfidf": [self.term_scores[k] for k in self.term_scores],
                "match_type": [self.match_types.get(k, "unknown") for k in self.term_scores],
            }
        ).sort_values("tfidf", ascending=False, ignore_index=True)


def _best_vocabulary_match(keyword: str, vocabulary: Mapping[str, float]) -> tuple[float, str]:
    """Resolve ``keyword`` against a TF-IDF vocabulary with graded back-off."""
    if keyword in vocabulary:
        return float(vocabulary[keyword]), "exact"

    best_score, match_type = 0.0, "none"
    for term, score in vocabulary.items():
        if (keyword in term or term in keyword) and score > best_score:
            best_score, match_type = float(score), "contains"
    if best_score > 0.0:
        return best_score, match_type

    keyword_tokens = set(keyword.split())
    if len(keyword_tokens) > 1:
        for term, score in vocabulary.items():
            term_tokens = set(term.split())
            if not term_tokens:
                continue
            overlap = len(keyword_tokens & term_tokens) / len(keyword_tokens)
            if overlap >= 0.5 and score > best_score:
                best_score, match_type = float(score), "overlap"
    if best_score > 0.0:
        return best_score, match_type
    return TfidfSalience.FALLBACK, "fallback"


def compute_tfidf_salience(
    keyword_lists: Sequence[Sequence[str]],
    ngram_range: tuple[int, int] = (1, 5),
    max_features: int = 10_000,
) -> TfidfSalience:
    """Mean TF-IDF weight of every author keyword in the corpus.

    Parameters
    ----------
    keyword_lists:
        One sequence of author keywords per record.
    ngram_range:
        Token n-gram range of the vectoriser; the default ``(1, 5)`` lets
        multi-word author keywords match a single vocabulary entry.
    max_features:
        Vocabulary cap passed to :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
    """
    documents: list[str] = []
    unique_keywords: set[str] = set()
    for keywords in keyword_lists:
        normalized = [normalize_keyword(kw) for kw in keywords]
        normalized = [kw for kw in normalized if kw]
        documents.append(" ".join(normalized))
        unique_keywords.update(normalized)

    non_empty = [doc for doc in documents if doc]
    if not non_empty or not unique_keywords:
        return TfidfSalience({}, {})

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[\w\-]+\b",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(non_empty)
    vocabulary = dict(
        zip(vectorizer.get_feature_names_out(), np.asarray(matrix.mean(axis=0)).ravel())
    )

    scores: dict[str, float] = {}
    match_types: dict[str, str] = {}
    for keyword in unique_keywords:
        score, match_type = _best_vocabulary_match(keyword, vocabulary)
        scores[keyword] = score
        match_types[keyword] = match_type
    return TfidfSalience(scores, match_types)
