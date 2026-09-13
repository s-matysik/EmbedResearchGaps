"""Keyword normalisation, lexical filters and TF-IDF salience."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from embedresearchgaps.config import GapConfig
from embedresearchgaps.text import (
    TfidfSalience,
    compute_tfidf_salience,
    core_domain_terms,
    is_methodological_term,
    keyword_frequencies,
    normalize_keyword,
    split_author_keywords,
    token_count,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Purchase   Intention ", "purchase intention"),
        ("GAMIFICATION", "gamification"),
        ("brand\tlove", "brand love"),
        ("", ""),
        ("  ", ""),
    ],
)
def test_normalize_keyword(raw: str, expected: str) -> None:
    assert normalize_keyword(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("supply chain", 2), ("gamification", 1), ("", 0), ("a b c d", 4)],
)
def test_token_count(raw: str, expected: int) -> None:
    assert token_count(raw) == expected


def test_split_author_keywords_handles_scopus_and_wos_formats() -> None:
    assert split_author_keywords("gamification; marketing ; loyalty") == [
        "gamification",
        "marketing",
        "loyalty",
    ]
    assert split_author_keywords("a | b") == ["a", "b"]
    assert split_author_keywords(["x", " y "]) == ["x", "y"]


@pytest.mark.parametrize("missing", [None, float("nan"), "", "  ", ";;"])
def test_split_author_keywords_missing_values(missing: object) -> None:
    assert split_author_keywords(missing) == []


def test_is_methodological_term_matches_tools_and_phrases() -> None:
    assert is_methodological_term("PLS")
    assert is_methodological_term("structural equation model")
    assert is_methodological_term("Systematic Literature Review")
    assert not is_methodological_term("customer engagement")
    assert not is_methodological_term("")


def test_is_methodological_term_respects_custom_config() -> None:
    config = GapConfig(methodological_patterns=("event study",), methodological_tools=frozenset())
    assert is_methodological_term("event study", config)
    assert not is_methodological_term("regression", config)


def test_keyword_frequencies_counts_documents_not_occurrences() -> None:
    frequencies = keyword_frequencies(
        [["gamification", "Gamification", "loyalty"], ["gamification"], []]
    )
    assert frequencies["gamification"] == 2
    assert frequencies["loyalty"] == 1


def test_core_domain_terms_uses_strict_majority() -> None:
    frequencies = {"gamification": 8, "loyalty": 5, "funware": 1}
    assert core_domain_terms(frequencies, 10, share=0.5) == {"gamification"}
    assert core_domain_terms(frequencies, 10, share=0.9) == set()
    assert core_domain_terms(frequencies, 0) == set()


class TestTfidfSalience:
    def test_exact_match_is_preferred(self) -> None:
        salience = compute_tfidf_salience(
            [["supply chain", "resilience"], ["supply chain"], ["resilience", "logistics"]]
        )
        assert salience.match_types["supply chain"] == "exact"
        assert salience.get("supply chain") > 0

    def test_unseen_keyword_falls_back_to_floor(self) -> None:
        salience = compute_tfidf_salience([["alpha beta"], ["alpha beta"]])
        assert salience.get("zzz qqq") == pytest.approx(TfidfSalience.FALLBACK)

    def test_frequent_term_is_less_salient_than_rare_term(self) -> None:
        documents = [["gamification", "loyalty"]] * 9 + [["gamification", "funware"]]
        salience = compute_tfidf_salience(documents)
        assert salience.get("funware") < salience.get("gamification")

    def test_empty_corpus_yields_empty_salience(self) -> None:
        salience = compute_tfidf_salience([[], []])
        assert len(salience) == 0
        assert salience.max_score == 1.0

    def test_to_frame_is_sorted_descending(self) -> None:
        salience = compute_tfidf_salience(
            [["alpha one", "beta two"], ["alpha one"], ["gamma three"]]
        )
        frame = salience.to_frame()
        assert list(frame.columns) == ["keyword", "tfidf", "match_type"]
        assert frame["tfidf"].is_monotonic_decreasing

    def test_match_statistics_cover_every_keyword(self) -> None:
        documents = [["supply chain risk"], ["supply chain"], ["resilience"]]
        salience = compute_tfidf_salience(documents)
        assert sum(salience.match_statistics().values()) == len(salience)
