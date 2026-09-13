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


def wos_tab_delimited(base: pd.DataFrame) -> pd.DataFrame:
    """A Web of Science tab-delimited export: two-letter field tags."""
    return pd.DataFrame({
        "PT": ["J"] * len(base),
        "AU": base["Authors"], "AF": base["Authors"],
        "TI": base["Title"], "SO": base["Source title"],
        "AB": base["Abstract"],
        "DE": base["Author Keywords"],
        "ID": ["ALGORITHM; PERFORMANCE"] * len(base),   # Keywords Plus
        "PY": base["Year"], "TC": base["Cited by"], "DI": base["DOI"],
        "UT": [f"WOS:00000{i}" for i in range(len(base))],
    })


def wos_full_record(base: pd.DataFrame) -> pd.DataFrame:
    """A Web of Science full-record export: spelled-out headers."""
    return pd.DataFrame({
        "Publication Type": ["J"] * len(base),
        "Author Full Names": base["Authors"],
        "Article Title": base["Title"],
        "Source Title": base["Source title"],
        "Abstract": base["Abstract"],
        "Author Keywords": base["Author Keywords"],
        "Keywords Plus": ["ALGORITHM; PERFORMANCE"] * len(base),
        "Publication Year": base["Year"],
        "Times Cited, All Databases": base["Cited by"],
        "DOI": base["DOI"],
        "UT (Unique WOS ID)": [f"WOS:00000{i}" for i in range(len(base))],
    })


class TestWebOfScienceExports:
    """Both Web of Science export flavours load without a column map."""

    def test_tab_delimited_tags_resolve(self, synthetic_frame: pd.DataFrame) -> None:
        corpus = from_dataframe(wos_tab_delimited(synthetic_frame))
        assert corpus.n_articles == len(synthetic_frame)
        assert corpus.column_map["title"] == "TI"
        assert corpus.column_map["author_keywords"] == "DE"
        assert corpus.column_map["citations"] == "TC"
        assert corpus.has_citations and corpus.has_abstracts

    def test_full_record_headers_resolve(self, synthetic_frame: pd.DataFrame) -> None:
        corpus = from_dataframe(wos_full_record(synthetic_frame))
        assert corpus.column_map["title"] == "Article Title"
        assert corpus.column_map["citations"] == "Times Cited, All Databases"
        assert corpus.column_map["eid"] == "UT (Unique WOS ID)"
        assert corpus.has_citations

    @pytest.mark.parametrize("builder", [wos_tab_delimited, wos_full_record])
    def test_keywords_plus_is_not_taken_for_author_keywords(
        self, synthetic_frame: pd.DataFrame, builder
    ) -> None:
        """Keywords Plus is database-assigned, so it must not be picked up."""
        corpus = from_dataframe(builder(synthetic_frame))
        flat = {kw for keywords in corpus.keyword_lists for kw in keywords}
        assert "algorithm" not in flat and "performance" not in flat

    @pytest.mark.parametrize("builder", [wos_tab_delimited, wos_full_record])
    def test_keywords_plus_can_be_requested_explicitly(
        self, synthetic_frame: pd.DataFrame, builder
    ) -> None:
        frame = builder(synthetic_frame)
        column = "ID" if "ID" in frame.columns else "Keywords Plus"
        corpus = from_dataframe(frame, column_map={"author_keywords": column})
        flat = {kw for keywords in corpus.keyword_lists for kw in keywords}
        assert flat == {"algorithm", "performance"}

    def test_both_flavours_give_the_same_corpus_as_scopus(
        self, synthetic_frame: pd.DataFrame
    ) -> None:
        scopus = from_dataframe(synthetic_frame)
        for builder in (wos_tab_delimited, wos_full_record):
            other = from_dataframe(builder(synthetic_frame))
            assert other.keyword_lists == scopus.keyword_lists
            assert list(other.frame["combined_text"]) == list(scopus.frame["combined_text"])
            assert list(other.frame["citations"]) == list(scopus.frame["citations"])

    def test_tab_separated_file_round_trip(
        self, synthetic_frame: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = tmp_path / "savedrecs.txt"
        wos_tab_delimited(synthetic_frame).to_csv(path, sep="\t", index=False)
        corpus = load_csv(path, sep="\t")
        assert corpus.n_articles == len(synthetic_frame)
        assert corpus.column_map["author_keywords"] == "DE"

    def test_both_modes_run_on_a_wos_corpus(
        self, synthetic_frame: pd.DataFrame, offline_config
    ) -> None:
        from embedresearchgaps import articles_first, keywords_first

        corpus = from_dataframe(wos_full_record(synthetic_frame))
        assert articles_first(corpus, offline_config).n_gaps > 0
        assert keywords_first(corpus, offline_config).n_gaps > 0
