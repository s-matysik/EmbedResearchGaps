"""The three composite scoring formulae, equations (7), (9) and (10).

Expected values are computed by hand from the published weights so that a
change to the formulae cannot pass unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest

from embedresearchgaps.config import (
    CombinationWeights,
    CrossClusterWeights,
    EmergingWeights,
    GapConfig,
)
from embedresearchgaps.gaps import (
    candidate_mask,
    safe_minmax,
    score_combination,
    score_cross_cluster,
    score_emerging,
    zscore,
)


class TestNormalisationHelpers:
    def test_safe_minmax_maps_to_unit_interval(self) -> None:
        np.testing.assert_allclose(safe_minmax([2.0, 4.0, 6.0]), [0.0, 0.5, 1.0])

    def test_safe_minmax_of_constant_is_neutral(self) -> None:
        np.testing.assert_allclose(safe_minmax([3.0, 3.0, 3.0]), [0.5, 0.5, 0.5])

    def test_safe_minmax_of_empty_is_empty(self) -> None:
        assert safe_minmax([]).size == 0

    def test_zscore_standardises(self) -> None:
        values = zscore([1.0, 2.0, 3.0])
        assert values.mean() == pytest.approx(0.0)
        assert np.std(values) == pytest.approx(1.0)

    def test_zscore_of_constant_is_zero(self) -> None:
        np.testing.assert_allclose(zscore([5.0, 5.0]), [0.0, 0.0])


class TestScoreEmerging:
    def test_matches_hand_computed_value(self) -> None:
        # 0.25*1.0 + 0.15*0.5 + 0.30*0.8 + 0.30*1.0 = 0.865
        assert score_emerging(1.0, 0.5, 0.8, 1.0) == pytest.approx(0.865)

    def test_bounds(self) -> None:
        assert score_emerging(0, 0, 0, 0) == pytest.approx(0.0)
        assert score_emerging(1, 1, 1, 1) == pytest.approx(1.0)

    def test_tfidf_and_rarity_dominate_distance(self) -> None:
        salient = score_emerging(0.0, 0.0, 1.0, 1.0)
        peripheral = score_emerging(1.0, 1.0, 0.0, 0.0)
        assert salient > peripheral

    def test_custom_weights_are_used(self) -> None:
        weights = EmergingWeights(distance=1.0, centrality=0.0, tfidf=0.0, rarity=0.0)
        assert score_emerging(0.4, 1.0, 1.0, 1.0, weights) == pytest.approx(0.4)

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            EmergingWeights(distance=0.5, centrality=0.5, tfidf=0.5, rarity=0.5)


class TestScoreCombination:
    def test_matches_hand_computed_value(self) -> None:
        # similarity term 1-2|0.45-0.45| = 1; frequency min(6/10,1) = 0.6;
        # tfidf min(10*0.02,1) = 0.2  ->  0.4*1 + 0.3*0.6 + 0.3*0.2 = 0.64
        assert score_combination(0.45, 3, 3, 0.02) == pytest.approx(0.64)

    def test_similarity_term_peaks_at_the_optimum(self) -> None:
        at_optimum = score_combination(0.45, 2, 2, 0.01)
        below = score_combination(0.30, 2, 2, 0.01)
        above = score_combination(0.60, 2, 2, 0.01)
        assert at_optimum > below and at_optimum > above

    def test_frequency_term_saturates(self) -> None:
        assert score_combination(0.45, 50, 50, 0.01) == pytest.approx(
            score_combination(0.45, 5, 5, 0.01)
        )

    def test_custom_optimum_shifts_the_peak(self) -> None:
        assert score_combination(0.60, 2, 2, 0.01, optimum=0.60) > score_combination(
            0.60, 2, 2, 0.01, optimum=0.45
        )

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            CombinationWeights(similarity=0.5, frequency=0.5, tfidf=0.5)


class TestScoreCrossCluster:
    def test_matches_hand_computed_value(self) -> None:
        # 0.25*min(2/5,1) + 0.35*(1-0.04) + 0.40*(0.01/0.05)
        # = 0.25*0.4 + 0.35*0.96 + 0.40*0.2 = 0.516
        assert score_cross_cluster(2, 0.04, 0.01, 0.05) == pytest.approx(0.516)

    def test_specificity_rewards_rare_keywords(self) -> None:
        rare = score_cross_cluster(2, 0.02, 0.01, 0.05)
        common = score_cross_cluster(2, 0.09, 0.01, 0.05)
        assert rare > common

    def test_zero_tfidf_max_does_not_divide_by_zero(self) -> None:
        assert np.isfinite(score_cross_cluster(1, 0.05, 0.0, 0.0))

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            CrossClusterWeights(frequency=0.5, specificity=0.5, tfidf=0.5)


class TestCandidateMask:
    KEYWORDS = [
        "algorithmic management",  # eligible
        "gamification",            # one token
        "structural equation model",  # methodological
        "supply chain",            # core domain in this test
        "digital presenteeism",    # eligible
        "",                        # empty
    ]

    def test_default_two_token_filter(self) -> None:
        mask = candidate_mask(self.KEYWORDS, GapConfig(), core_terms=["supply chain"])
        assert list(mask) == [True, False, False, False, True, False]

    def test_token_filter_can_be_disabled(self) -> None:
        mask = candidate_mask(self.KEYWORDS, GapConfig(gap_token_count=None))
        assert mask[1]  # single-token keyword now eligible

    def test_methodological_filter_can_be_disabled(self) -> None:
        config = GapConfig(gap_token_count=3, filter_methodological=False)
        mask = candidate_mask(self.KEYWORDS, config)
        assert mask[2]

    def test_extra_stop_keywords_are_removed(self) -> None:
        config = GapConfig(extra_stop_keywords=("Digital Presenteeism",))
        mask = candidate_mask(self.KEYWORDS, config)
        assert not mask[4]

    def test_invalid_token_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="gap_token_count"):
            GapConfig(gap_token_count=0)
