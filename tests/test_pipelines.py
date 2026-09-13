"""End-to-end behaviour of both operating modes."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import GROUP_EXCLUSIVE, GROUPS, build_synthetic_frame

from embedresearchgaps import (
    ClusteringConfig,
    EncoderConfig,
    GapConfig,
    RunConfig,
    articles_first,
    from_dataframe,
    keywords_first,
    run,
)

PLANTED_GAPS = {kw for spec in GROUPS.values() for kw in spec["planted_gaps"]}  # type: ignore[union-attr]


class TestArticlesFirst:
    def test_recovers_the_planted_thematic_structure(self, synthetic_corpus, offline_config) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        assert result.n_clusters == 3
        assert sorted(result.diagnostics["cluster_sizes"].values()) == [20, 20, 20]
        assert result.diagnostics["cluster_quality"]["silhouette"] > 0.0

    def test_finds_every_planted_emerging_concept(self, synthetic_corpus, offline_config) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        emerging = set(result.gaps[result.gaps["gap_type"] == "Emerging Concept"]["keyword"])
        assert PLANTED_GAPS <= emerging

    def test_group_exclusive_keywords_become_cross_cluster_gaps(
        self, synthetic_corpus, offline_config
    ) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        cross = set(result.gaps[result.gaps["gap_type"] == "Cross-Cluster Concept"]["keyword"])
        assert cross & set(GROUP_EXCLUSIVE.values())

    def test_core_and_methodological_terms_are_never_reported(
        self, synthetic_corpus, offline_config
    ) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        reported = set(result.gaps["keyword"])
        assert "structural equation model" not in reported
        for spec in GROUPS.values():
            assert not reported & set(spec["core"])  # type: ignore[arg-type]

    def test_single_token_keywords_are_excluded_by_default(
        self, synthetic_corpus, offline_config
    ) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        plain = result.gaps[result.gaps["gap_type"] != "Conceptual Combination"]
        assert all(len(str(kw).split()) == 2 for kw in plain["keyword"])

    def test_deduplication_reduces_the_candidate_list(
        self, synthetic_corpus, offline_config
    ) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        diagnostics = result.diagnostics
        assert diagnostics["gaps_after_deduplication"] <= diagnostics["gaps_before_deduplication"]

    def test_embeddings_and_index_are_retained_for_figures(
        self, synthetic_corpus, offline_config
    ) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        assert result.embeddings["articles"].shape[0] == len(result.articles)
        assert result.embeddings["keywords"].shape[0] == len(result.keyword_index)

    def test_centroid_similarity_is_square_and_unit_diagonal(
        self, synthetic_corpus, offline_config
    ) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        matrix = result.centroid_similarity
        assert matrix.shape == (3, 3)
        np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-9)


class TestKeywordsFirst:
    def test_runs_and_clusters_the_keyword_space(self, synthetic_corpus, offline_config) -> None:
        result = keywords_first(synthetic_corpus, offline_config)
        assert result.mode == "keywords_first"
        assert result.n_clusters == 3
        assert len(result.keywords) == len(synthetic_corpus.unique_keywords())

    def test_every_candidate_exceeds_its_cluster_threshold(
        self, synthetic_corpus, offline_config
    ) -> None:
        config = dataclasses.replace(
            offline_config, gaps=GapConfig(periphery_percentile=80.0, max_keyword_frequency=2)
        )
        result = keywords_first(synthetic_corpus, config)
        assert result.n_gaps > 0
        for _, row in result.gaps.iterrows():
            assert row["metric_distance_from_centroid"] >= row["metric_cluster_threshold"]

    def test_peripheral_share_respects_the_percentile(
        self, synthetic_corpus, offline_config
    ) -> None:
        config = dataclasses.replace(
            offline_config, gaps=GapConfig(periphery_percentile=90.0, max_keyword_frequency=None)
        )
        result = keywords_first(synthetic_corpus, config)
        assert result.diagnostics["peripheral_share"] <= 0.15

    def test_proportional_frequency_rule_is_applied(
        self, synthetic_corpus, offline_config
    ) -> None:
        config = dataclasses.replace(
            offline_config, gaps=GapConfig(max_keyword_frequency=None), top_n_articles=60
        )
        result = keywords_first(synthetic_corpus, config)
        assert result.diagnostics["frequency_threshold"] == 1

    def test_citations_are_attached_when_available(self, synthetic_corpus, offline_config) -> None:
        config = dataclasses.replace(
            offline_config, gaps=GapConfig(periphery_percentile=70.0, max_keyword_frequency=3)
        )
        result = keywords_first(synthetic_corpus, config)
        assert result.gaps["citations"].notna().any()

    def test_too_few_keywords_raises(self, offline_config) -> None:
        corpus = from_dataframe(
            pd.DataFrame(
                {
                    "Title": ["A", "B"],
                    "Abstract": ["x y", "y z"],
                    "Author Keywords": ["alpha beta", "alpha beta"],
                }
            )
        )
        with pytest.raises(ValueError, match="at least 4 unique keywords"):
            keywords_first(corpus, offline_config)


class TestSemanticRanking:
    def test_ranking_truncates_and_orders_the_corpus(self, synthetic_corpus, offline_config) -> None:
        config = dataclasses.replace(
            offline_config,
            top_n_articles=20,
            problem_description="employee wellbeing engagement burnout remote work autonomy",
        )
        result = articles_first(synthetic_corpus, config)
        assert len(result.articles) == 20
        assert result.diagnostics["ranking"]["ranked"] is True
        assert result.articles["query_distance"].is_monotonic_increasing

    def test_ranking_selects_the_matching_theme(self, synthetic_corpus, offline_config) -> None:
        config = dataclasses.replace(
            offline_config,
            top_n_articles=15,
            problem_description="credit risk lending default bank capital liquidity borrower",
            clustering=ClusteringConfig(n_clusters=2, k_selection="fixed"),
        )
        result = articles_first(synthetic_corpus, config)
        share = result.articles["title"].str.startswith("Finance").mean()
        assert share > 0.6

    def test_unranked_run_keeps_input_order(self, synthetic_corpus, offline_config) -> None:
        result = articles_first(synthetic_corpus, dataclasses.replace(offline_config, top_n_articles=10))
        assert result.diagnostics["ranking"]["ranked"] is False
        assert list(result.articles["title"]) == list(synthetic_corpus.frame["title"].head(10))


class TestDeterminism:
    @pytest.mark.parametrize("mode", ["articles_first", "keywords_first"])
    def test_same_seed_gives_identical_output(self, synthetic_corpus, offline_config, mode) -> None:
        first = run(synthetic_corpus, mode, offline_config)
        second = run(synthetic_corpus, mode, offline_config)
        pd.testing.assert_frame_equal(first.gaps, second.gaps)
        assert first.diagnostics["cluster_sizes"] == second.diagnostics["cluster_sizes"]

    def test_unknown_mode_raises(self, synthetic_corpus, offline_config) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            run(synthetic_corpus, "topic_model", offline_config)


class TestRobustness:
    def test_missing_abstracts_are_tolerated(self, synthetic_frame, offline_config) -> None:
        frame = synthetic_frame.copy()
        frame.loc[::2, "Abstract"] = ""
        result = articles_first(from_dataframe(frame), offline_config)
        assert result.n_clusters == 3

    def test_missing_citations_are_tolerated(self, synthetic_frame, offline_config) -> None:
        frame = synthetic_frame.drop(columns=["Cited by"])
        result = keywords_first(
            from_dataframe(frame),
            dataclasses.replace(offline_config, gaps=GapConfig(periphery_percentile=70.0, max_keyword_frequency=3)),
        )
        assert result.gaps["citations"].isna().all()

    def test_missing_years_skip_the_year_diagnostics(self, synthetic_frame, offline_config) -> None:
        frame = synthetic_frame.drop(columns=["Year"])
        result = articles_first(from_dataframe(frame), offline_config)
        assert result.diagnostics["corpus"]["year_range"] is None

    def test_auto_k_is_used_when_no_cluster_count_is_given(self, synthetic_corpus) -> None:
        encoder = EncoderConfig(name="tfidf-svd", dimensions=64, extra={"random_state": 42})
        config = RunConfig(
            top_n_articles=60,
            article_encoder=encoder,
            keyword_encoder=encoder,
            clustering=ClusteringConfig(k_min=2, k_max=6),
        )
        result = articles_first(synthetic_corpus, config)
        assert result.diagnostics["k_selection"]["k"] == result.n_clusters
        assert result.diagnostics["k_selection"]["rule"] != "fixed"

    def test_cluster_with_too_few_keywords_is_skipped_not_fatal(
        self, synthetic_frame, offline_config
    ) -> None:
        frame = synthetic_frame.copy()
        outlier = frame.iloc[[0]].copy()
        outlier["Title"] = "Isolated paper about quantum gravity and spacetime foam"
        outlier["Abstract"] = "quantum gravity spacetime foam loop holography entropy horizon"
        outlier["Author Keywords"] = "quantum gravity"
        frame = pd.concat([frame, outlier], ignore_index=True)
        config = dataclasses.replace(
            offline_config, clustering=ClusteringConfig(n_clusters=4, k_selection="fixed")
        )
        result = articles_first(from_dataframe(frame), config)
        assert result.n_clusters == 4
        assert isinstance(result.diagnostics["skipped_clusters"], dict)

    def test_custom_cluster_labeler_is_used(self, synthetic_corpus, offline_config) -> None:
        result = articles_first(
            synthetic_corpus, offline_config,
            cluster_labeler=lambda cluster, articles, members: f"Theme {cluster} (n={len(articles)})",
        )
        assert result.cluster_labels[0].startswith("Theme 0")


class TestPersistence:
    def test_save_writes_tables_and_metadata(self, synthetic_corpus, offline_config, tmp_path: Path) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        written = result.save(tmp_path, prefix="mode_b")
        for key in ("gaps", "keywords", "articles", "config", "summary", "centroid_similarity"):
            assert Path(written[key]).exists()
        reloaded = pd.read_csv(written["gaps"])
        assert len(reloaded) == result.n_gaps

    def test_summary_is_json_serialisable(self, synthetic_corpus, offline_config) -> None:
        import json

        result = keywords_first(synthetic_corpus, offline_config)
        payload = json.loads(result.to_json())
        assert payload["mode"] == "keywords_first"
        assert payload["n_clusters"] == result.n_clusters

    def test_display_column_restores_source_spelling(self, offline_config) -> None:
        frame = pd.DataFrame(
            {
                "Title": [f"Paper about ESG and SME finance {i}" for i in range(12)],
                "Abstract": ["esg reporting sme finance disclosure rating" for _ in range(12)],
                "Author Keywords": (
                    ["ESG disclosure; SME finance; capital cost"] * 6
                    + ["ESG disclosure; carbon accounting; capital cost"] * 6
                ),
                "Year": list(range(2013, 2025)),
                "Cited by": [10] * 12,
            }
        )
        corpus = from_dataframe(frame)
        assert corpus.spelling["esg disclosure"] == "ESG disclosure"
        assert corpus.spelling["sme finance"] == "SME finance"
        result = keywords_first(
            corpus,
            dataclasses.replace(
                offline_config,
                clustering=ClusteringConfig(n_clusters=2, k_selection="fixed"),
                gaps=GapConfig(periphery_percentile=50.0, max_keyword_frequency=6),
            ),
        )
        assert "keyword_display" in result.gaps.columns
        assert any("ESG" in label or "SME" in label for label in result.cluster_labels.values())
        for _, row in result.gaps.iterrows():
            assert row["keyword_display"].lower() == row["keyword"]

    def test_cluster_labels_keep_acronyms(self, synthetic_corpus, offline_config) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        assert all(label and label[0].isupper() for label in result.cluster_labels.values())

    def test_top_gaps_is_sorted(self, synthetic_corpus, offline_config) -> None:
        result = articles_first(synthetic_corpus, offline_config)
        top = result.top_gaps(5)
        assert top["score"].is_monotonic_decreasing
