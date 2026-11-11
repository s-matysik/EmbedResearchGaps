"""The published description must match the code that produced the results.

Three parameters were printed wrongly in the first submission -- a combination
window of [0.45, 0.80] instead of (0.25, 0.65), a "recency" term the score does
not have, and a core-domain share of 0.30 instead of 0.50 -- and none of the
387 tests could notice, because no test read the prose.

These tests close that gap from the code side: they pin the values an
accompanying paper quotes, so changing a default without updating the text
fails the suite rather than the review.  The expected values below are the
ones stated in Section 2.2 and supplementary S10 of the article.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embedresearchgaps.config import (
    CombinationWeights,
    CrossClusterWeights,
    EmergingWeights,
    GapConfig,
    RunConfig,
)
from embedresearchgaps.gaps import score_combination, score_emerging

#: Exactly as printed in the article. Changing a default means changing both.
DOCUMENTED = {
    "periphery_percentile": 95.0,
    "max_keyword_frequency": 1,
    "gap_token_count": 2,
    "core_domain_share": 0.50,
    "min_tfidf": 0.005,
    "max_gaps_per_cluster": 3,
    "max_presence_ratio": 0.10,
    "max_frequency_other_clusters": 3,
    "combination_similarity_range": (0.25, 0.65),
    "combination_similarity_optimum": 0.45,
}


class TestDocumentedDefaults:
    """Defaults quoted in the article against the dataclass they describe."""

    @pytest.mark.parametrize("name,expected", sorted(DOCUMENTED.items()))
    def test_gap_config_default(self, name: str, expected: object) -> None:
        assert getattr(GapConfig(), name) == expected, (
            f"GapConfig.{name} changed; Section 2.2 and supplementary S10 quote "
            f"{expected!r} and must be updated with it"
        )

    def test_analysis_corpus_size(self) -> None:
        assert RunConfig().top_n_articles == 50

    def test_emerging_weights_are_the_published_four(self) -> None:
        weights = EmergingWeights()
        assert (weights.distance, weights.centrality, weights.tfidf, weights.rarity) == (
            0.25, 0.15, 0.30, 0.30
        )
        # There is no recency component: an earlier draft of the paper said so,
        # and this assertion is what makes that claim checkable.
        assert not hasattr(weights, "recency")

    def test_combination_weights(self) -> None:
        weights = CombinationWeights()
        assert (weights.similarity, weights.frequency, weights.tfidf) == (0.40, 0.30, 0.30)

    def test_cross_cluster_weights_sum_to_one(self) -> None:
        weights = CrossClusterWeights()
        assert abs(weights.frequency + weights.specificity + weights.tfidf - 1.0) < 1e-9


class TestDocumentedFormulae:
    """Behaviour the article states in words, asserted numerically."""

    def test_combination_similarity_peaks_at_the_optimum(self) -> None:
        peak = score_combination(0.45, 1, 1, 0.1)
        assert peak > score_combination(0.30, 1, 1, 0.1)
        assert peak > score_combination(0.60, 1, 1, 0.1)

    def test_combination_similarity_decays_linearly(self) -> None:
        # The article states the term is 1 - 2|s - 0.45|; two symmetric offsets
        # must therefore score identically.
        assert score_combination(0.35, 1, 1, 0.1) == pytest.approx(
            score_combination(0.55, 1, 1, 0.1)
        )

    def test_emerging_score_is_the_documented_weighted_sum(self) -> None:
        assert score_emerging(1.0, 0.0, 0.0, 0.0) == pytest.approx(0.25)
        assert score_emerging(0.0, 1.0, 0.0, 0.0) == pytest.approx(0.15)
        assert score_emerging(0.0, 0.0, 1.0, 0.0) == pytest.approx(0.30)
        assert score_emerging(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.30)
        assert score_emerging(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)


class TestDepositedRunConfiguration:
    """The deposited runs must have used the documented values.

    The article's parameter table is presented as the configuration of the
    reported runs, not as the library defaults, so the deposited
    ``*_run_config.json`` is the authority. The test is skipped where the
    results tree is not present, so the suite still passes from a source-only
    checkout.
    """

    @pytest.mark.parametrize("corpus", ["management", "finance", "gamification"])
    def test_run_config_matches_the_article(self, corpus: str) -> None:
        path = (Path(__file__).resolve().parents[1]
                / "paper" / "results" / corpus / "articles_first_run_config.json")
        if not path.exists():
            pytest.skip("deposited results are not part of this checkout")
        gaps = json.loads(path.read_text())["gaps"]
        for name, expected in DOCUMENTED.items():
            found = gaps[name]
            if isinstance(expected, tuple):
                found = tuple(found)
            assert found == expected, f"{corpus}: {name} was {found!r}, article says {expected!r}"
