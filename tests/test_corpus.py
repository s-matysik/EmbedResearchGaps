"""Corpus loading, column resolution and record filtering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from embedresearchgaps import Corpus, from_dataframe, load_csv


def minimal_frame(**overrides: object) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Title": ["First paper", "Second paper", "Third paper"],
            "Abstract": ["abstract one", "abstract two", "abstract three"],
            "Author Keywords": ["alpha; beta", "beta;gamma", "alpha"],
            "Year": [2020, 2021, 2022],
            "Cited by": [10, 0, 5],
            "DOI": ["10.1/a", "10.1/b", "10.1/c"],
            "Source title": ["J A", "J B", "J C"],
            "Authors": ["X", "Y", "Z"],
        }
    )
    for column, value in overrides.items():
        frame[column] = value
    return frame


class TestColumnResolution:
    def test_scopus_style_headers(self) -> None:
        corpus = from_dataframe(minimal_frame())
        assert corpus.column_map["author_keywords"] == "Author Keywords"
        assert corpus.column_map["citations"] == "Cited by"
        assert corpus.n_articles == 3

    def test_web_of_science_style_headers(self) -> None:
        frame = pd.DataFrame(
            {"TI": ["A paper"], "AB": ["text"], "DE": ["alpha; beta"], "PY": [2019], "TC": [3]}
        )
        corpus = from_dataframe(frame)
        assert corpus.frame.loc[0, "title"] == "A paper"
        assert corpus.frame.loc[0, "keywords_normalized"] == ["alpha", "beta"]
        assert corpus.frame.loc[0, "citations"] == 3

    def test_explicit_column_map_wins(self) -> None:
        frame = pd.DataFrame({"t": ["A"], "kw": ["alpha"], "Title": ["ignored"]})
        corpus = from_dataframe(frame, column_map={"title": "t", "author_keywords": "kw"})
        assert corpus.frame.loc[0, "title"] == "A"

    def test_missing_required_column_raises(self) -> None:
        with pytest.raises(ValueError, match="required column"):
            from_dataframe(pd.DataFrame({"Title": ["A"], "Abstract": ["b"]}))

    def test_optional_columns_are_filled(self) -> None:
        corpus = from_dataframe(
            pd.DataFrame({"Title": ["A paper"], "Author Keywords": ["alpha"]})
        )
        assert corpus.frame.loc[0, "abstract"] == ""
        assert corpus.frame.loc[0, "doi"] == ""
        assert np.isnan(corpus.frame.loc[0, "citations"])
        assert not corpus.has_citations


class TestRecordFiltering:
    def test_records_without_text_are_dropped(self) -> None:
        frame = minimal_frame()
        frame.loc[1, ["Title", "Abstract"]] = ["", ""]
        corpus = from_dataframe(frame)
        assert corpus.n_articles == 2
        assert corpus.dropped_empty_text == 1

    def test_records_without_keywords_are_dropped_by_default(self) -> None:
        frame = minimal_frame()
        frame.loc[2, "Author Keywords"] = ""
        corpus = from_dataframe(frame)
        assert corpus.n_articles == 2
        assert corpus.dropped_empty_keywords == 1

    def test_keyword_requirement_can_be_relaxed(self) -> None:
        frame = minimal_frame()
        frame.loc[2, "Author Keywords"] = np.nan
        corpus = from_dataframe(frame, require_keywords=False)
        assert corpus.n_articles == 3
        assert corpus.frame.loc[2, "keywords_normalized"] == []

    def test_duplicate_keywords_inside_a_record_are_collapsed(self) -> None:
        frame = minimal_frame()
        frame.loc[0, "Author Keywords"] = "Alpha; alpha ; ALPHA"
        corpus = from_dataframe(frame)
        assert corpus.frame.loc[0, "keywords_normalized"] == ["alpha"]

    def test_year_is_extracted_from_a_date_string(self) -> None:
        frame = minimal_frame(Year=["2020-05-01", "2021-01-01", "not a date"])
        corpus = from_dataframe(frame)
        assert corpus.frame.loc[0, "year"] == 2020
        assert np.isnan(corpus.frame.loc[2, "year"])


class TestCorpusInterface:
    def test_summary_reports_corpus_statistics(self) -> None:
        summary = from_dataframe(minimal_frame()).summary()
        assert summary["n_articles"] == 3
        assert summary["n_unique_keywords"] == 3
        assert summary["year_range"] == (2020, 2022)
        assert summary["mean_keywords_per_article"] == pytest.approx(5 / 3)

    def test_unique_keywords_preserve_first_appearance_order(self) -> None:
        corpus = from_dataframe(minimal_frame())
        assert corpus.unique_keywords() == ["alpha", "beta", "gamma"]

    def test_head_and_subset_return_new_corpora(self) -> None:
        corpus = from_dataframe(minimal_frame())
        assert corpus.head(2).n_articles == 2
        assert corpus.subset([True, False, True]).n_articles == 2
        assert corpus.n_articles == 3  # original untouched

    def test_combined_text_concatenates_title_and_abstract(self) -> None:
        corpus = from_dataframe(minimal_frame())
        assert corpus.frame.loc[0, "combined_text"] == "First paper abstract one"


def test_load_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "export.csv"
    minimal_frame().to_csv(path, index=False)
    corpus = load_csv(path)
    assert isinstance(corpus, Corpus)
    assert corpus.n_articles == 3
    assert corpus.source_path == str(path)
