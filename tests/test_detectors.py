"""The four detectors, exercised on keyword spaces with designed embeddings.

Placing keywords at known angles makes every cosine similarity and every
centroid distance exact, so the detectors can be checked against the
published criteria rather than against whatever an embedding model happens to
produce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import unit_vector

from embedresearchgaps.clustering import fit_kmeans
from embedresearchgaps.config import ClusteringConfig, GapConfig
from embedresearchgaps.gaps import (
    COMBINATION_SEPARATOR,
    Gap,
    deduplicate_gaps,
    detect_conceptual_combinations,
    detect_cross_cluster_concepts,
    detect_emerging_concepts,
    detect_peripheral_keywords,
    gaps_to_frame,
)
from embedresearchgaps.keywords import KeywordSpace
from embedresearchgaps.text import compute_tfidf_salience


def make_space(entries: dict[str, tuple[int, list[float]]]) -> KeywordSpace:
    """Build a keyword space from ``{keyword: (frequency, embedding)}``."""
    keywords = list(entries)
    embeddings = np.asarray([entries[k][1] for k in keywords], dtype=float)
    frequency = np.asarray([entries[k][0] for k in keywords], dtype=int)
    centroid = embeddings.mean(axis=0)
    centrality = np.linalg.norm(embeddings - centroid, axis=1)
    return KeywordSpace(keywords, embeddings, frequency, centrality)


def make_articles(keyword_lists: list[list[str]], cluster: int = 0, citations: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": [f"Article {i}" for i in range(len(keyword_lists))],
            "doi": [f"10.0/{i}" for i in range(len(keyword_lists))],
            "keywords_normalized": keyword_lists,
            "cluster": cluster,
            "citations": citations,
        }
    )


class TestEmergingConcepts:
    ENTRIES = {
        "core theme": (8, [0.0, 0.0, 1.0]),
        "second theme": (5, [0.05, 0.0, 1.0]),
        "rare peripheral": (1, [3.0, 3.0, 0.0]),
        "rare central": (1, [0.05, 0.05, 1.0]),
        "one": (1, [9.0, 0.0, 0.0]),
        "structural equation model": (1, [0.0, 9.0, 0.0]),
    }

    @pytest.fixture()
    def space(self) -> KeywordSpace:
        space = make_space(self.ENTRIES)
        space.assignment = fit_kmeans(
            space.embeddings, ClusteringConfig(distance_metric="euclidean"), n_clusters=2
        )
        return space

    @pytest.fixture()
    def articles(self) -> pd.DataFrame:
        return make_articles(
            [["core theme", "rare peripheral"], ["core theme", "rare central"], ["second theme"]]
        )

    def test_only_eligible_rare_two_token_keywords_are_returned(
        self, space: KeywordSpace, articles: pd.DataFrame
    ) -> None:
        tfidf = compute_tfidf_salience(list(articles["keywords_normalized"]))
        gaps = detect_emerging_concepts(
            space, articles, 0, tfidf, {"rare peripheral": 1, "rare central": 1},
            core_terms=["core theme"], config=GapConfig(max_gaps_per_cluster=5),
        )
        returned = {gap.keyword for gap in gaps}
        assert returned == {"rare peripheral", "rare central"}
        assert all(gap.gap_type == "Emerging Concept" for gap in gaps)
        assert all(gap.frequency == 1 for gap in gaps)

    def test_peripheral_candidate_outranks_central_one(
        self, space: KeywordSpace, articles: pd.DataFrame
    ) -> None:
        tfidf = compute_tfidf_salience(list(articles["keywords_normalized"]))
        gaps = detect_emerging_concepts(
            space, articles, 0, tfidf, {"rare peripheral": 1, "rare central": 1},
            ["core theme"], GapConfig(max_gaps_per_cluster=5),
        )
        scores = {gap.keyword: gap.score for gap in gaps}
        assert scores["rare peripheral"] > scores["rare central"]

    def test_respects_max_gaps_per_cluster(
        self, space: KeywordSpace, articles: pd.DataFrame
    ) -> None:
        tfidf = compute_tfidf_salience(list(articles["keywords_normalized"]))
        gaps = detect_emerging_concepts(
            space, articles, 0, tfidf, {}, ["core theme"], GapConfig(max_gaps_per_cluster=1)
        )
        assert len(gaps) == 1

    def test_article_evidence_is_attached(
        self, space: KeywordSpace, articles: pd.DataFrame
    ) -> None:
        tfidf = compute_tfidf_salience(list(articles["keywords_normalized"]))
        gaps = detect_emerging_concepts(
            space, articles, 0, tfidf, {}, ["core theme"], GapConfig(max_gaps_per_cluster=5)
        )
        peripheral = next(g for g in gaps if g.keyword == "rare peripheral")
        assert peripheral.n_articles == 1
        assert peripheral.article_titles == ("Article 0",)
        assert peripheral.citations == 10.0

    def test_no_candidates_returns_empty_list(self) -> None:
        space = make_space({"core theme": (9, [1.0, 0.0, 0.0]), "other core": (9, [0.0, 1.0, 0.0])})
        gaps = detect_emerging_concepts(
            space, make_articles([["core theme"]]), 0,
            compute_tfidf_salience([["core theme"]]), {}, [], GapConfig(),
        )
        assert gaps == []


class TestConceptualCombinations:
    def build(self, angle: float) -> tuple[KeywordSpace, pd.DataFrame]:
        space = make_space(
            {
                "alpha topic": (4, unit_vector(0.0)),
                "beta topic": (4, unit_vector(angle)),
                "gamma topic": (3, unit_vector(1.0)),
            }
        )
        articles = make_articles(
            [["alpha topic", "gamma topic"], ["alpha topic"], ["beta topic"], ["beta topic"]]
        )
        return space, articles

    def test_pair_inside_the_similarity_window_is_reported(self) -> None:
        space, articles = self.build(angle=63.0)  # cos ~ 0.454
        tfidf = compute_tfidf_salience(list(articles["keywords_normalized"]))
        gaps = detect_conceptual_combinations(
            space, articles, 0, tfidf, GapConfig(min_tfidf=0.0, max_gaps_per_cluster=5)
        )
        pairs = {tuple(sorted(gap.members)) for gap in gaps}
        assert ("alpha topic", "beta topic") in pairs
        combination = next(g for g in gaps if set(g.members) == {"alpha topic", "beta topic"})
        assert combination.metrics["similarity"] == pytest.approx(np.cos(np.deg2rad(63.0)), abs=1e-6)
        assert COMBINATION_SEPARATOR in combination.keyword

    def test_near_synonyms_are_rejected(self) -> None:
        space, articles = self.build(angle=10.0)  # cos ~ 0.985 > 0.65
        gaps = detect_conceptual_combinations(
            space, articles, 0,
            compute_tfidf_salience(list(articles["keywords_normalized"])),
            GapConfig(min_tfidf=0.0),
        )
        assert all(set(g.members) != {"alpha topic", "beta topic"} for g in gaps)

    def test_unrelated_pairs_are_rejected(self) -> None:
        space, articles = self.build(angle=88.0)  # cos ~ 0.035 < 0.25
        gaps = detect_conceptual_combinations(
            space, articles, 0,
            compute_tfidf_salience(list(articles["keywords_normalized"])),
            GapConfig(min_tfidf=0.0),
        )
        assert all(set(g.members) != {"alpha topic", "beta topic"} for g in gaps)

    def test_co_occurring_pairs_are_excluded(self) -> None:
        space, articles = self.build(angle=63.0)
        # alpha and gamma co-occur in article 0 and sit at cos(1 deg) anyway
        gaps = detect_conceptual_combinations(
            space, articles, 0,
            compute_tfidf_salience(list(articles["keywords_normalized"])),
            GapConfig(min_tfidf=0.0, max_gaps_per_cluster=9),
        )
        assert all(set(g.members) != {"alpha topic", "gamma topic"} for g in gaps)

    def test_low_tfidf_pairs_are_filtered(self) -> None:
        space, articles = self.build(angle=63.0)
        gaps = detect_conceptual_combinations(
            space, articles, 0,
            compute_tfidf_salience(list(articles["keywords_normalized"])),
            GapConfig(min_tfidf=10.0),
        )
        assert gaps == []

    def test_infrequent_keywords_are_not_eligible(self) -> None:
        space = make_space(
            {"alpha topic": (1, unit_vector(0.0)), "beta topic": (1, unit_vector(63.0))}
        )
        articles = make_articles([["alpha topic"], ["beta topic"]])
        gaps = detect_conceptual_combinations(
            space, articles, 0,
            compute_tfidf_salience(list(articles["keywords_normalized"])),
            GapConfig(min_tfidf=0.0),
        )
        assert gaps == []


class TestCrossClusterConcepts:
    @pytest.fixture()
    def articles(self) -> pd.DataFrame:
        rows = [
            {"keywords_normalized": ["local theme", "shared theme"], "cluster": 0},
            {"keywords_normalized": ["local theme"], "cluster": 0},
            {"keywords_normalized": ["remote theme", "shared theme"], "cluster": 1},
            {"keywords_normalized": ["remote theme"], "cluster": 1},
            {"keywords_normalized": ["ubiquitous theme"], "cluster": 1},
        ]
        frame = pd.DataFrame(rows)
        frame["title"] = [f"Article {i}" for i in range(len(frame))]
        frame["doi"] = ""
        frame["citations"] = 5
        return frame

    @pytest.fixture()
    def space(self) -> KeywordSpace:
        return make_space(
            {"local theme": (2, [1.0, 0.0, 0.0]), "shared theme": (1, [0.0, 1.0, 0.0])}
        )

    def test_keyword_absent_from_this_cluster_is_reported(
        self, space: KeywordSpace, articles: pd.DataFrame
    ) -> None:
        tfidf = compute_tfidf_salience(list(articles["keywords_normalized"]))
        gaps = detect_cross_cluster_concepts(
            space, articles, 0, tfidf,
            {"remote theme": 2, "ubiquitous theme": 1}, [],
            GapConfig(min_tfidf=0.0, max_presence_ratio=0.5, max_gaps_per_cluster=5),
        )
        assert {gap.keyword for gap in gaps} == {"remote theme", "ubiquitous theme"}
        assert all(gap.gap_type == "Cross-Cluster Concept" for gap in gaps)

    def test_keyword_present_in_this_cluster_is_skipped(
        self, space: KeywordSpace, articles: pd.DataFrame
    ) -> None:
        tfidf = compute_tfidf_salience(list(articles["keywords_normalized"]))
        gaps = detect_cross_cluster_concepts(
            space, articles, 0, tfidf, {}, [], GapConfig(min_tfidf=0.0, max_presence_ratio=1.0)
        )
        assert "shared theme" not in {gap.keyword for gap in gaps}

    def test_presence_ratio_cap_is_enforced(
        self, space: KeywordSpace, articles: pd.DataFrame
    ) -> None:
        tfidf = compute_tfidf_salience(list(articles["keywords_normalized"]))
        gaps = detect_cross_cluster_concepts(
            space, articles, 0, tfidf, {"remote theme": 2}, [],
            GapConfig(min_tfidf=0.0, max_presence_ratio=0.1),
        )
        assert "remote theme" not in {gap.keyword for gap in gaps}

    def test_frequency_cap_in_other_clusters_is_enforced(
        self, space: KeywordSpace, articles: pd.DataFrame
    ) -> None:
        tfidf = compute_tfidf_salience(list(articles["keywords_normalized"]))
        gaps = detect_cross_cluster_concepts(
            space, articles, 0, tfidf, {"remote theme": 2}, [],
            GapConfig(min_tfidf=0.0, max_presence_ratio=1.0, max_frequency_other_clusters=1),
        )
        assert "remote theme" not in {gap.keyword for gap in gaps}

    def test_missing_cluster_column_raises(self, space: KeywordSpace) -> None:
        with pytest.raises(ValueError, match="'cluster' column"):
            detect_cross_cluster_concepts(
                space, pd.DataFrame({"keywords_normalized": [["a"]]}), 0,
                compute_tfidf_salience([["a"]]), {}, [], GapConfig(),
            )


class TestPeripheralKeywords:
    def test_flags_only_keywords_above_the_cluster_threshold(self) -> None:
        entries: dict[str, tuple[int, list[float]]] = {
            f"filler{i:02d} term": (1, [1.0 + 0.01 * i, 0.0, 0.0]) for i in range(20)
        }
        entries["outlier term"] = (1, [1.0, 4.0, 0.0])
        space = make_space(entries)
        space.assignment = fit_kmeans(
            space.embeddings, ClusteringConfig(distance_metric="euclidean"), n_clusters=2
        )
        articles = make_articles([[k] for k in entries])
        gaps = detect_peripheral_keywords(
            space, articles, compute_tfidf_salience(list(articles["keywords_normalized"])),
            [], GapConfig(periphery_percentile=95.0),
        )
        assert "outlier term" in {gap.keyword for gap in gaps}
        assert all(gap.gap_type == "Peripheral Keyword" for gap in gaps)
        assert gaps == sorted(gaps, key=lambda g: g.score, reverse=True)

    def test_share_flagged_tracks_the_percentile(self) -> None:
        rng = np.random.default_rng(4)
        entries = {
            f"kw{i:03d} term": (1, list(rng.normal(size=3)))
            for i in range(200)
        }
        space = make_space(entries)
        space.assignment = fit_kmeans(
            space.embeddings, ClusteringConfig(distance_metric="euclidean"), n_clusters=4
        )
        articles = make_articles([[k] for k in entries])
        tfidf = compute_tfidf_salience(list(articles["keywords_normalized"]))
        flagged = len(
            detect_peripheral_keywords(space, articles, tfidf, [], GapConfig(periphery_percentile=90.0))
        )
        assert 0.05 <= flagged / len(entries) <= 0.16

    def test_frequent_keywords_are_not_flagged(self) -> None:
        space = make_space(
            {"central term": (1, [1.0, 0.0, 0.0]), "frequent outlier": (9, [0.0, 5.0, 0.0]),
             "second term": (1, [1.01, 0.0, 0.0]), "third term": (1, [0.99, 0.0, 0.0])}
        )
        space.assignment = fit_kmeans(
            space.embeddings, ClusteringConfig(distance_metric="euclidean"), n_clusters=2
        )
        articles = make_articles([[k] for k in space.keywords])
        gaps = detect_peripheral_keywords(
            space, articles, compute_tfidf_salience(list(articles["keywords_normalized"])),
            [], GapConfig(max_keyword_frequency=1),
        )
        assert "frequent outlier" not in {gap.keyword for gap in gaps}


class TestGapBookkeeping:
    def test_deduplicate_keeps_the_highest_score(self) -> None:
        gaps = [
            Gap("Emerging Concept", "alpha term", 0, 0.4, 1, "low"),
            Gap("Emerging Concept", "alpha term", 1, 0.9, 1, "high"),
            Gap("Cross-Cluster Concept", "alpha term", 2, 0.5, 1, "other type"),
        ]
        deduplicated = deduplicate_gaps(gaps)
        assert len(deduplicated) == 2
        emerging = next(g for g in deduplicated if g.gap_type == "Emerging Concept")
        assert emerging.score == 0.9 and emerging.cluster == 1

    def test_gaps_to_frame_expands_metrics(self) -> None:
        frame = gaps_to_frame(
            [Gap("Emerging Concept", "alpha term", 0, 0.5, 1, "why", metrics={"tfidf": 0.01})]
        )
        assert frame.loc[0, "metric_tfidf"] == 0.01
        assert frame.loc[0, "keyword"] == "alpha term"

    def test_gaps_to_frame_of_empty_list_has_schema(self) -> None:
        frame = gaps_to_frame([])
        assert frame.empty
        assert {"gap_type", "keyword", "score", "cluster"} <= set(frame.columns)
