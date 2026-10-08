"""Optional citation bounds on gap candidates.

The filter is inactive unless a bound is configured, so the first test in this
module is the regression guard: with the default :class:`GapConfig` the
detectors must return exactly what they returned before the filter existed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from test_detectors import make_space

from embedresearchgaps.clustering import fit_kmeans
from embedresearchgaps.config import ClusteringConfig, GapConfig
from embedresearchgaps.gaps import (
    citation_filter_active,
    citation_mask,
    detect_cross_cluster_concepts,
    detect_emerging_concepts,
    detect_peripheral_keywords,
    keyword_citation_velocity,
    keyword_citations,
    passes_citation_filter,
    reference_year,
)
from embedresearchgaps.keywords import KeywordSpace
from embedresearchgaps.text import compute_tfidf_salience

ENTRIES = {
    "core theme": (8, [0.0, 0.0, 1.0]),
    "second theme": (5, [0.05, 0.0, 1.0]),
    "rare peripheral": (1, [3.0, 3.0, 0.0]),
    "rare central": (1, [0.05, 0.05, 1.0]),
}

#: "rare peripheral" is carried by a highly cited record, "rare central" by a
#: barely cited one, so a lower bound and an upper bound select opposite halves.
CITATIONS = {"rare peripheral": 120, "rare central": 2}


def make_frame(citations: list[int | float | None] | None = None) -> pd.DataFrame:
    keyword_lists = [
        ["core theme", "rare peripheral"],
        ["core theme", "rare central"],
        ["second theme"],
    ]
    frame = pd.DataFrame(
        {
            "title": [f"Article {i}" for i in range(len(keyword_lists))],
            "doi": [f"10.0/{i}" for i in range(len(keyword_lists))],
            "keywords_normalized": keyword_lists,
            "cluster": 0,
        }
    )
    if citations is not None:
        frame["citations"] = citations
    return frame


@pytest.fixture()
def space() -> KeywordSpace:
    space = make_space(ENTRIES)
    space.assignment = fit_kmeans(
        space.embeddings, ClusteringConfig(distance_metric="euclidean"), n_clusters=2
    )
    return space


@pytest.fixture()
def articles() -> pd.DataFrame:
    return make_frame([CITATIONS["rare peripheral"], CITATIONS["rare central"], 7])


def emerging(space: KeywordSpace, articles: pd.DataFrame, config: GapConfig) -> set[str]:
    tfidf = compute_tfidf_salience(list(articles["keywords_normalized"]))
    gaps = detect_emerging_concepts(
        space, articles, 0, tfidf, {"rare peripheral": 1, "rare central": 1},
        core_terms=["core theme"], config=config,
    )
    return {gap.keyword for gap in gaps}


class TestDefaultIsInactive:
    def test_default_config_returns_both_candidates(
        self, space: KeywordSpace, articles: pd.DataFrame
    ) -> None:
        assert emerging(space, articles, GapConfig(max_gaps_per_cluster=5)) == {
            "rare peripheral",
            "rare central",
        }

    def test_mask_short_circuits_without_bounds(self) -> None:
        """No bound set: every keyword passes without the frame being read."""
        mask = citation_mask(["a", "b", "c"], pd.DataFrame(), GapConfig())
        assert mask.tolist() == [True, True, True]
        assert mask.dtype == np.bool_


class TestPredicate:
    @pytest.mark.parametrize(
        ("citations", "config", "expected"),
        [
            (50.0, GapConfig(min_citations=10), True),
            (5.0, GapConfig(min_citations=10), False),
            (10.0, GapConfig(min_citations=10), True),
            (5.0, GapConfig(max_citations=10), True),
            (50.0, GapConfig(max_citations=10), False),
            (10.0, GapConfig(max_citations=10), True),
            (7.0, GapConfig(min_citations=5, max_citations=10), True),
            (11.0, GapConfig(min_citations=5, max_citations=10), False),
            (None, GapConfig(min_citations=5), True),
            (None, GapConfig(min_citations=5, keep_uncited_candidates=False), False),
            (None, GapConfig(), True),
        ],
    )
    def test_bounds_are_inclusive(
        self, citations: float | None, config: GapConfig, expected: bool
    ) -> None:
        assert passes_citation_filter(citations, config) is expected


class TestSupportLookup:
    def test_support_is_the_best_cited_record(self, articles: pd.DataFrame) -> None:
        assert keyword_citations("rare peripheral", articles) == 120.0
        assert keyword_citations("core theme", articles) == 120.0

    def test_missing_column_means_unknown(self) -> None:
        assert keyword_citations("rare peripheral", make_frame()) is None

    def test_missing_values_mean_unknown(self) -> None:
        frame = make_frame([None, None, None])
        assert keyword_citations("rare peripheral", frame) is None

    def test_absent_keyword_means_unknown(self, articles: pd.DataFrame) -> None:
        assert keyword_citations("never used", articles) is None


class TestDetectors:
    def test_lower_bound_keeps_the_well_cited_candidate(
        self, space: KeywordSpace, articles: pd.DataFrame
    ) -> None:
        config = GapConfig(max_gaps_per_cluster=5, min_citations=50)
        assert emerging(space, articles, config) == {"rare peripheral"}

    def test_upper_bound_keeps_the_barely_cited_candidate(
        self, space: KeywordSpace, articles: pd.DataFrame
    ) -> None:
        config = GapConfig(max_gaps_per_cluster=5, max_citations=50)
        assert emerging(space, articles, config) == {"rare central"}

    def test_window_can_exclude_everything(
        self, space: KeywordSpace, articles: pd.DataFrame
    ) -> None:
        config = GapConfig(max_gaps_per_cluster=5, min_citations=200)
        assert emerging(space, articles, config) == set()

    def test_unknown_support_survives_by_default(self, space: KeywordSpace) -> None:
        """A corpus without citation data must not be emptied by a bound."""
        config = GapConfig(max_gaps_per_cluster=5, min_citations=50)
        assert emerging(space, make_frame(), config) == {"rare peripheral", "rare central"}

    def test_unknown_support_can_be_dropped(self, space: KeywordSpace) -> None:
        config = GapConfig(
            max_gaps_per_cluster=5, min_citations=50, keep_uncited_candidates=False
        )
        assert emerging(space, make_frame(), config) == set()

    def test_periphery_detector_honours_the_bound(
        self, space: KeywordSpace, articles: pd.DataFrame
    ) -> None:
        tfidf = compute_tfidf_salience(list(articles["keywords_normalized"]))
        unfiltered = detect_peripheral_keywords(
            space, articles, tfidf, ["core theme"],
            GapConfig(periphery_percentile=10.0, max_keyword_frequency=1),
        )
        filtered = detect_peripheral_keywords(
            space, articles, tfidf, ["core theme"],
            GapConfig(periphery_percentile=10.0, max_keyword_frequency=1, max_citations=50),
        )
        assert {gap.keyword for gap in unfiltered} >= {gap.keyword for gap in filtered}
        assert "rare peripheral" not in {gap.keyword for gap in filtered}

    def test_cross_cluster_detector_honours_the_bound(self, space: KeywordSpace) -> None:
        """The bound must be selective, not merely subtractive.

        Cluster 1 contributes two cross-cluster candidates with opposite
        citation support; an upper bound has to drop one and keep the other.
        The corpus is padded so that both stay below
        ``max_presence_ratio``, which would otherwise reject them first.
        """
        keyword_lists = [["core theme"]] * 10 + [["shadow banking"], ["tariff shock"]]
        frame = pd.DataFrame(
            {
                "title": [f"A{i}" for i in range(len(keyword_lists))],
                "doi": [f"10.0/{i}" for i in range(len(keyword_lists))],
                "keywords_normalized": keyword_lists,
                "cluster": [0] * 10 + [1, 1],
                "citations": [5] * 10 + [300, 4],
            }
        )
        tfidf = compute_tfidf_salience(list(frame["keywords_normalized"]))
        kwargs = dict(
            space=space, all_articles=frame, cluster_id=0, tfidf=tfidf,
            global_frequency={"shadow banking": 1, "tariff shock": 1},
            core_terms=["core theme"],
        )
        unfiltered = detect_cross_cluster_concepts(
            **kwargs, config=GapConfig(min_tfidf=0.0, max_gaps_per_cluster=5)
        )
        filtered = detect_cross_cluster_concepts(
            **kwargs,
            config=GapConfig(min_tfidf=0.0, max_gaps_per_cluster=5, max_citations=100),
        )
        assert {"shadow banking", "tariff shock"} <= {gap.keyword for gap in unfiltered}
        assert {gap.keyword for gap in filtered} == {"tariff shock"}


class TestVelocity:
    """Citations per year, the rate form of the filter.

    ``old`` and ``fresh`` carry the same raw citation count, so any difference
    between them is attributable to the rate and not to the level.
    """

    #: 60 citations over 20 years = 3/year; 60 over 2 years = 30/year.
    FRAME = pd.DataFrame(
        {
            "title": ["Old", "Fresh", "Filler"],
            "doi": ["10.0/o", "10.0/f", "10.0/x"],
            "keywords_normalized": [
                ["core theme", "rare peripheral"],
                ["core theme", "rare central"],
                ["second theme"],
            ],
            "cluster": 0,
            "citations": [60, 60, 1],
            "year": [2005, 2023, 2024],
        }
    )

    def test_reference_year_defaults_to_the_newest_record(self) -> None:
        assert reference_year(self.FRAME, GapConfig()) == 2024

    def test_explicit_reference_year_wins(self) -> None:
        assert reference_year(self.FRAME, GapConfig(citation_year_reference=2030)) == 2030

    def test_age_counts_the_publication_year_itself(self) -> None:
        """A paper published in the reference year is one year old, not zero."""
        frame = pd.DataFrame(
            {
                "title": ["Same year"],
                "doi": ["10.0/s"],
                "keywords_normalized": [["rare central"]],
                "cluster": 0,
                "citations": [7],
                "year": [2024],
            }
        )
        assert keyword_citation_velocity("rare central", frame, GapConfig()) == 7.0

    def test_velocity_separates_equally_cited_records(self) -> None:
        config = GapConfig()
        assert keyword_citation_velocity("rare peripheral", self.FRAME, config) == 3.0
        assert keyword_citation_velocity("rare central", self.FRAME, config) == 30.0

    def test_unknown_without_year_column(self) -> None:
        frame = make_frame([10, 10, 10])
        assert keyword_citation_velocity("rare peripheral", frame, GapConfig()) is None

    @pytest.mark.parametrize(
        ("threshold", "expected"),
        [
            (1.0, {"rare peripheral", "rare central"}),
            (2.0, {"rare peripheral", "rare central"}),
            (5.0, {"rare central"}),
            (50.0, set()),
        ],
    )
    def test_threshold_selects_by_rate_not_by_level(
        self, space: KeywordSpace, threshold: float, expected: set[str]
    ) -> None:
        config = GapConfig(max_gaps_per_cluster=5, min_citations_per_year=threshold)
        assert emerging(space, self.FRAME, config) == expected

    def test_raw_count_bound_cannot_separate_them(self, space: KeywordSpace) -> None:
        """The level is identical, which is the point of using the rate."""
        config = GapConfig(max_gaps_per_cluster=5, min_citations=50)
        assert emerging(space, self.FRAME, config) == {"rare peripheral", "rare central"}

    def test_fractional_threshold_is_allowed(self, space: KeywordSpace) -> None:
        frame = self.FRAME.copy()
        frame["citations"] = [10, 1, 1]  # old: 10/20 = 0.5/year, fresh: 1/2 = 0.5/year
        config = GapConfig(max_gaps_per_cluster=5, min_citations_per_year=0.5)
        assert emerging(space, frame, config) == {"rare peripheral", "rare central"}
        config = GapConfig(max_gaps_per_cluster=5, min_citations_per_year=0.6)
        assert emerging(space, frame, config) == set()

    def test_rate_and_window_combine(self, space: KeywordSpace) -> None:
        config = GapConfig(
            max_gaps_per_cluster=5, min_citations_per_year=5.0, max_citations=100
        )
        assert emerging(space, self.FRAME, config) == {"rare central"}
        config = GapConfig(
            max_gaps_per_cluster=5, min_citations_per_year=5.0, max_citations=10
        )
        assert emerging(space, self.FRAME, config) == set()

    def test_unknown_velocity_survives_by_default(self, space: KeywordSpace) -> None:
        config = GapConfig(max_gaps_per_cluster=5, min_citations_per_year=5.0)
        assert emerging(space, make_frame([60, 60, 1]), config) == {
            "rare peripheral",
            "rare central",
        }

    def test_unknown_velocity_can_be_dropped(self, space: KeywordSpace) -> None:
        config = GapConfig(
            max_gaps_per_cluster=5,
            min_citations_per_year=5.0,
            keep_uncited_candidates=False,
        )
        assert emerging(space, make_frame([60, 60, 1]), config) == set()

    def test_reference_year_changes_the_outcome(self, space: KeywordSpace) -> None:
        """Harvesting the counts later means the same totals accrued more slowly."""
        config = GapConfig(
            max_gaps_per_cluster=5, min_citations_per_year=20.0,
            citation_year_reference=2024,
        )
        assert emerging(space, self.FRAME, config) == {"rare central"}
        config = GapConfig(
            max_gaps_per_cluster=5, min_citations_per_year=20.0,
            citation_year_reference=2030,
        )
        assert emerging(space, self.FRAME, config) == set()


class TestConfigValidation:
    def test_negative_bound_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_citations"):
            GapConfig(min_citations=-1)

    def test_inverted_window_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            GapConfig(min_citations=100, max_citations=10)

    def test_negative_rate_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_citations_per_year"):
            GapConfig(min_citations_per_year=-0.5)

    def test_implausible_reference_year_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="citation_year_reference"):
            GapConfig(citation_year_reference=0)

    def test_filter_is_inactive_without_any_bound(self) -> None:
        assert citation_filter_active(GapConfig()) is False
        assert citation_filter_active(GapConfig(min_citations_per_year=1.0)) is True
