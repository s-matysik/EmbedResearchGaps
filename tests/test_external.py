"""Loading a ranking produced outside this package.

The point of the loader is not that it parses a file: it is that what comes out
is accepted unchanged by the parts of the package that assess a candidate list.
The tests therefore check the parsing contract *and* that the downstream
entry points -- the control-set split, the candidate table, the overlap
measure -- work on the result.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from embedresearchgaps import (
    EXTERNAL_GAP_TYPE,
    Corpus,
    jaccard,
    load_external_ranking,
)
from embedresearchgaps.validation import selected_and_control_keywords


@pytest.fixture()
def ranking_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Term": ["Circular sourcing", "Nearshoring strategy", "supplier selection"],
            "Total link strength": [9.5, 4.25, 1.0],
        }
    )


class TestInputShapes:
    def test_sequence_of_keywords(self) -> None:
        result = load_external_ranking(["alpha", "beta", "gamma"])
        assert result.mode == "external"
        assert list(result.gaps["keyword"]) == ["alpha", "beta", "gamma"]

    def test_dataframe_with_aliased_columns(self, ranking_frame: pd.DataFrame) -> None:
        """Headers that are not valid identifiers must still resolve."""
        result = load_external_ranking(ranking_frame)
        assert result.diagnostics["external"]["keyword_column"] == "Term"
        assert result.diagnostics["external"]["score_column"] == "Total link strength"
        assert list(result.gaps["score"]) == [9.5, 4.25, 1.0]

    def test_csv_roundtrip(self, ranking_frame: pd.DataFrame, tmp_path: Path) -> None:
        path = tmp_path / "ranking.csv"
        ranking_frame.to_csv(path, index=False)
        assert load_external_ranking(path).n_gaps == 3

    def test_excel_roundtrip(self, ranking_frame: pd.DataFrame, tmp_path: Path) -> None:
        path = tmp_path / "ranking.xlsx"
        ranking_frame.to_excel(path, index=False)
        assert load_external_ranking(path).n_gaps == 3

    def test_missing_file_is_reported_as_such(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_external_ranking(tmp_path / "absent.csv")

    def test_unsupported_type_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="DataFrame"):
            load_external_ranking(42)


class TestParsing:
    def test_keywords_are_normalized_so_they_match_a_corpus(self) -> None:
        result = load_external_ranking(["  Circular   Sourcing  "])
        assert list(result.gaps["keyword"]) == ["circular sourcing"]

    def test_normalization_can_be_disabled(self) -> None:
        result = load_external_ranking(["Circular Sourcing"], normalize=False)
        assert list(result.gaps["keyword"]) == ["Circular Sourcing"]

    def test_duplicates_are_dropped_and_counted(self) -> None:
        result = load_external_ranking(["alpha", "ALPHA", "beta"])
        assert result.n_gaps == 2
        assert result.diagnostics["external"]["duplicates_dropped"] == 1

    def test_rank_derived_score_is_contiguous_after_a_duplicate(self) -> None:
        """Dropping a duplicate must not leave a hole in the derived scale."""
        result = load_external_ranking(["alpha", "ALPHA", "beta", "gamma"])
        assert list(result.gaps["score"].round(4)) == [1.0, 0.6667, 0.3333]

    def test_file_position_is_preserved_in_metrics(self) -> None:
        result = load_external_ranking(["alpha", "ALPHA", "beta"])
        assert [g.metrics["external_rank"] for g in result.gap_objects] == [1.0, 3.0]

    def test_explicit_rank_column_orders_the_output(self) -> None:
        frame = pd.DataFrame({"keyword": ["c", "a", "b"], "rank": [3, 1, 2]})
        assert list(load_external_ranking(frame).gaps["keyword"]) == ["a", "b", "c"]

    def test_score_column_orders_descending(self) -> None:
        frame = pd.DataFrame({"keyword": ["low", "high"], "score": [0.1, 0.9]})
        assert list(load_external_ranking(frame).gaps["keyword"]) == ["high", "low"]

    def test_top_n_truncates_after_ordering(self) -> None:
        frame = pd.DataFrame({"keyword": ["low", "high"], "score": [0.1, 0.9]})
        result = load_external_ranking(frame, top_n=1)
        assert list(result.gaps["keyword"]) == ["high"]

    def test_type_column_is_honoured(self) -> None:
        frame = pd.DataFrame({"keyword": ["a", "b"], "type": ["Emerging", None]})
        types = list(load_external_ranking(frame).gaps["gap_type"])
        assert types == ["Emerging", EXTERNAL_GAP_TYPE]

    def test_default_type_is_not_one_of_our_detector_types(self) -> None:
        from embedresearchgaps import GAP_TYPES

        assert EXTERNAL_GAP_TYPE not in GAP_TYPES

    def test_explicit_column_that_does_not_exist_is_rejected(self) -> None:
        with pytest.raises(KeyError, match="nope"):
            load_external_ranking(pd.DataFrame({"keyword": ["a"]}), keyword_column="nope")

    def test_unresolvable_keyword_column_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no keyword column"):
            load_external_ranking(pd.DataFrame({"foo": ["a"], "bar": [1]}))

    def test_empty_ranking_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            load_external_ranking(pd.DataFrame({"keyword": []}))

    def test_blank_keywords_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            load_external_ranking(pd.DataFrame({"keyword": ["", "   "]}))


class TestCorpusEvidence:
    def test_without_a_corpus_no_evidence_is_invented(self) -> None:
        result = load_external_ranking(["circular sourcing"])
        gap = result.gap_objects[0]
        assert (gap.frequency, gap.n_articles, gap.citations) == (0, 0, None)
        assert result.diagnostics["external"]["corpus_attached"] is False

    def test_corpus_attaches_frequency_and_records(self, synthetic_corpus: Corpus) -> None:
        present = synthetic_corpus.frame["keywords_normalized"].iloc[0][0]
        result = load_external_ranking([present], synthetic_corpus)
        gap = result.gap_objects[0]
        assert gap.frequency > 0
        assert gap.n_articles > 0
        assert len(gap.article_titles) > 0

    def test_candidates_absent_from_the_corpus_are_reported(
        self, synthetic_corpus: Corpus
    ) -> None:
        result = load_external_ranking(["not in any record"], synthetic_corpus)
        assert result.diagnostics["external"]["unmatched_in_corpus"] == [
            "not in any record"
        ]
        assert result.gap_objects[0].frequency == 0


class TestDownstreamCompatibility:
    """The loader exists so that these calls work on a foreign ranking."""

    def test_control_set_is_disjoint_from_the_external_candidates(
        self, synthetic_corpus: Corpus
    ) -> None:
        present = list(synthetic_corpus.frame["keywords_normalized"].iloc[0])[:2]
        result = load_external_ranking(present, synthetic_corpus)
        selected, control = selected_and_control_keywords(result)
        assert set(selected) == set(present)
        assert control
        assert not set(selected) & set(control)

    def test_overlap_with_a_built_in_run_is_measurable(
        self, synthetic_corpus: Corpus, offline_config
    ) -> None:
        import warnings

        from embedresearchgaps import articles_first

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ours = articles_first(synthetic_corpus, offline_config)
        theirs = load_external_ranking(
            list(ours.gaps["keyword"])[:2] + ["a keyword they alone propose"],
            synthetic_corpus,
        )
        overlap = jaccard(set(ours.gaps["keyword"]), set(theirs.gaps["keyword"]))
        assert 0.0 < overlap < 1.0

    def test_result_saves_like_any_other_run(
        self, synthetic_corpus: Corpus, tmp_path: Path
    ) -> None:
        result = load_external_ranking(["alpha", "beta"], synthetic_corpus)
        written = result.save(tmp_path, prefix="external")
        assert written
        assert all(Path(p).exists() for p in written.values())

    def test_no_partition_is_claimed(self, synthetic_corpus: Corpus) -> None:
        """An external ranking carries no clustering of ours."""
        result = load_external_ranking(["alpha", "beta"], synthetic_corpus)
        assert set(result.gaps["cluster"]) == {-1}
        assert result.cluster_labels == {}
        assert result.centroid_similarity is None
