"""Stability-aware consensus across k-means seeds."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from embedresearchgaps import ClusteringConfig, GapConfig, consensus_gaps, jaccard
from embedresearchgaps.consensus import DEFAULT_SEEDS


def test_jaccard_edge_cases() -> None:
    assert jaccard({"a", "b"}, {"b"}) == pytest.approx(0.5)
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert np.isnan(jaccard(set(), set()))


class TestConsensusGaps:
    @pytest.fixture(scope="class")
    def consensus(self, request):
        corpus = request.getfixturevalue("synthetic_corpus")
        from embedresearchgaps import EncoderConfig, RunConfig

        encoder = EncoderConfig(name="tfidf-svd", dimensions=64, extra={"random_state": 42})
        config = RunConfig(
            top_n_articles=60,
            article_encoder=encoder,
            keyword_encoder=encoder,
            clustering=ClusteringConfig(n_clusters=3, k_selection="fixed"),
            gaps=GapConfig(max_gaps_per_cluster=3),
        )
        return consensus_gaps(corpus, "articles_first", config, seeds=(1, 2, 3, 4))

    def test_support_is_a_fraction_of_seeds(self, consensus) -> None:
        assert not consensus.gaps.empty
        assert consensus.gaps["support"].between(0, 1).all()
        assert set(consensus.gaps["n_seeds_found"].unique()) <= {1, 2, 3, 4}
        np.testing.assert_allclose(
            consensus.gaps["support"], consensus.gaps["n_seeds_found"] / 4
        )

    def test_gaps_are_ranked_by_support_then_score(self, consensus) -> None:
        frame = consensus.gaps
        assert frame["support"].is_monotonic_decreasing
        for _, group in frame.groupby("support"):
            assert group["mean_score"].is_monotonic_decreasing

    def test_score_statistics_bracket_the_mean(self, consensus) -> None:
        frame = consensus.gaps
        assert (frame["min_score"] <= frame["mean_score"] + 1e-12).all()
        assert (frame["mean_score"] <= frame["max_score"] + 1e-12).all()

    def test_reference_run_is_the_first_seed(self, consensus) -> None:
        assert consensus.reference.mode == "articles_first"
        assert consensus.per_seed[0]["seed"] == 1
        assert len(consensus.per_seed) == 4

    def test_stability_is_reported(self, consensus) -> None:
        stability = consensus.stability
        assert stability["n_seeds"] == 4
        assert 0.0 <= stability["mean_pairwise_jaccard"] <= 1.0
        assert stability["min_pairwise_jaccard"] <= stability["max_pairwise_jaccard"]
        assert sum(stability["candidates_per_support"].values()) == len(consensus.gaps)

    def test_at_support_filters(self, consensus) -> None:
        majority = consensus.at_support(0.5)
        unanimous = consensus.at_support(1.0)
        assert len(unanimous) <= len(majority) <= len(consensus.gaps)
        assert (majority["support"] >= 0.5).all()

    def test_summary_counts_match_the_table(self, consensus) -> None:
        summary = consensus.summary()
        assert summary["n_candidates"] == len(consensus.gaps)
        assert summary["n_unanimous"] == int((consensus.gaps["support"] == 1.0).sum())
        assert summary["n_majority"] == int((consensus.gaps["support"] >= 0.5).sum())
        json.dumps(summary, default=str)  # must be serialisable

    def test_repeated_seeds_are_collapsed(self, synthetic_corpus, offline_config) -> None:
        with pytest.raises(ValueError, match="at least 2 distinct seeds"):
            consensus_gaps(
                synthetic_corpus, "articles_first", offline_config, seeds=(42, 42, 42)
            )
        result = consensus_gaps(
            synthetic_corpus, "articles_first", offline_config, seeds=(42, 7, 42, 7, 42)
        )
        assert result.seeds == (42, 7)
        assert result.stability["n_seeds"] == 2

    def test_some_candidates_are_seed_invariant(self, consensus) -> None:
        assert (consensus.gaps["support"] == 1.0).any()

    def test_min_support_drops_unstable_candidates(self, synthetic_corpus, offline_config) -> None:
        loose = consensus_gaps(
            synthetic_corpus, "keywords_first", offline_config, seeds=(1, 2, 3, 4, 5)
        )
        strict = consensus_gaps(
            synthetic_corpus, "keywords_first", offline_config,
            seeds=(1, 2, 3, 4, 5), min_support=1.0,
        )
        assert len(strict.gaps) <= len(loose.gaps)
        assert strict.min_support == 1.0

    def test_default_seeds_are_five(self) -> None:
        assert len(DEFAULT_SEEDS) == 5

    @pytest.mark.parametrize("seeds", [(), (42,)])
    def test_too_few_seeds_raises(self, synthetic_corpus, offline_config, seeds) -> None:
        with pytest.raises(ValueError, match="at least 2 distinct seeds"):
            consensus_gaps(synthetic_corpus, "articles_first", offline_config, seeds=seeds)

    @pytest.mark.parametrize("support", [-0.1, 1.5])
    def test_invalid_support_raises(self, synthetic_corpus, offline_config, support) -> None:
        with pytest.raises(ValueError, match="min_support"):
            consensus_gaps(
                synthetic_corpus, "articles_first", offline_config,
                seeds=(1, 2), min_support=support,
            )

    def test_save_writes_table_and_summary(self, consensus, tmp_path: Path) -> None:
        written = consensus.save(tmp_path, prefix="modeB")
        assert Path(written["gaps"]).exists()
        payload = json.loads(Path(written["summary"]).read_text())
        assert payload["seeds"] == [1, 2, 3, 4]

    def test_works_for_keywords_first_mode(self, synthetic_corpus, offline_config) -> None:
        result = consensus_gaps(
            synthetic_corpus, "keywords_first", offline_config, seeds=(1, 2, 3)
        )
        assert result.mode == "keywords_first"
        assert set(result.gaps["gap_type"].unique()) <= {"Peripheral Keyword"}

    def test_empty_candidate_sets_yield_an_empty_table(
        self, synthetic_corpus, offline_config
    ) -> None:
        config = dataclasses.replace(
            offline_config, gaps=GapConfig(min_tfidf=99.0, max_keyword_frequency=0)
        )
        result = consensus_gaps(
            synthetic_corpus, "articles_first", config, seeds=(1, 2)
        )
        assert result.gaps.empty
        assert {"support", "keyword", "gap_type"} <= set(result.gaps.columns)
        assert result.summary()["n_candidates"] == 0
