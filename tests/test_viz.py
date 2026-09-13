"""Figure generation for both modes and for a validation report."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest
from conftest import build_synthetic_frame
from test_validation import make_stub

from embedresearchgaps import ClusteringConfig, articles_first, from_dataframe, keywords_first
from embedresearchgaps.validation import validate_gaps
from embedresearchgaps.viz import (
    embed_2d,
    generate_figures,
    plot_article_map,
    plot_centroid_heatmap,
    plot_cooccurrence_network,
    plot_gap_summary,
    plot_k_selection,
    plot_keyword_map,
    plot_validation_comparison,
    plot_year_distribution,
)


@pytest.fixture(scope="module")
def mode_b(synthetic_corpus_module, offline_config_module):
    return articles_first(synthetic_corpus_module, offline_config_module)


@pytest.fixture(scope="module")
def synthetic_corpus_module():
    return from_dataframe(build_synthetic_frame())


@pytest.fixture(scope="module")
def offline_config_module():
    from embedresearchgaps import EncoderConfig, GapConfig, RunConfig

    encoder = EncoderConfig(name="tfidf-svd", dimensions=64, extra={"random_state": 42})
    return RunConfig(
        top_n_articles=60,
        article_encoder=encoder,
        keyword_encoder=encoder,
        clustering=ClusteringConfig(n_clusters=3, k_selection="fixed"),
        gaps=GapConfig(max_gaps_per_cluster=3),
    )


def test_embed_2d_falls_back_to_pca_for_tiny_inputs() -> None:
    import numpy as np

    coords = embed_2d(np.eye(3))
    assert coords.shape == (3, 2)


def test_embed_2d_returns_two_dimensions() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    assert embed_2d(rng.normal(size=(40, 10))).shape == (40, 2)


class TestIndividualFigures:
    def test_keyword_map(self, mode_b, tmp_path: Path) -> None:
        path = Path(plot_keyword_map(mode_b, tmp_path / "keywords.png"))
        assert path.exists() and path.stat().st_size > 10_000

    def test_article_map(self, mode_b, tmp_path: Path) -> None:
        assert Path(plot_article_map(mode_b, tmp_path / "articles.png")).exists()

    def test_cooccurrence_network(self, mode_b, tmp_path: Path) -> None:
        assert Path(plot_cooccurrence_network(mode_b, tmp_path / "network.png")).exists()

    def test_centroid_heatmap(self, mode_b, tmp_path: Path) -> None:
        assert Path(plot_centroid_heatmap(mode_b, tmp_path / "heatmap.png")).exists()

    def test_gap_summary(self, mode_b, tmp_path: Path) -> None:
        assert Path(plot_gap_summary(mode_b, tmp_path / "gaps.png")).exists()

    def test_year_distribution(self, mode_b, tmp_path: Path) -> None:
        assert Path(plot_year_distribution(mode_b, tmp_path / "years.png")).exists()

    def test_article_map_unavailable_in_keywords_first(
        self, synthetic_corpus_module, offline_config_module, tmp_path: Path
    ) -> None:
        result = keywords_first(synthetic_corpus_module, offline_config_module)
        with pytest.raises(ValueError, match="no article embeddings"):
            plot_article_map(result, tmp_path / "x.png")

    def test_k_selection_requires_a_trace(self, mode_b, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no k-selection trace"):
            plot_k_selection(mode_b, tmp_path / "k.png")

    def test_k_selection_is_drawn_when_k_was_selected(
        self, synthetic_corpus_module, offline_config_module, tmp_path: Path
    ) -> None:
        config = dataclasses.replace(
            offline_config_module, clustering=ClusteringConfig(k_min=2, k_max=5)
        )
        result = articles_first(synthetic_corpus_module, config)
        assert Path(plot_k_selection(result, tmp_path / "k.png")).exists()

    def test_year_distribution_requires_years(
        self, offline_config_module, tmp_path: Path
    ) -> None:
        frame = build_synthetic_frame().drop(columns=["Year"])
        result = articles_first(from_dataframe(frame), offline_config_module)
        with pytest.raises(ValueError, match="no publication years"):
            plot_year_distribution(result, tmp_path / "years.png")


class TestFigureBundle:
    def test_generate_figures_covers_mode_b(self, mode_b, tmp_path: Path) -> None:
        written = generate_figures(mode_b, tmp_path, prefix="fig")
        for name in ("keyword_map", "article_map", "cooccurrence", "centroid_heatmap",
                     "gap_summary", "year_distribution"):
            assert Path(written[name]).exists()

    def test_inapplicable_figures_are_skipped_not_raised(
        self, synthetic_corpus_module, offline_config_module, tmp_path: Path
    ) -> None:
        result = keywords_first(synthetic_corpus_module, offline_config_module)
        written = generate_figures(result, tmp_path, prefix="modeA")
        assert "article_map" not in written
        assert "article_map" in written["skipped"]

    def test_include_restricts_the_bundle(self, mode_b, tmp_path: Path) -> None:
        written = generate_figures(mode_b, tmp_path, include=["centroid_heatmap"])
        assert set(written) == {"centroid_heatmap"}


def test_validation_figure(mode_b, tmp_path: Path) -> None:
    report = validate_gaps(
        mode_b, [make_stub("m1", set(mode_b.gaps["keyword"])), make_stub("m2", set())],
        "problem", "domain", 10,
    )
    assert Path(plot_validation_comparison(report, tmp_path / "validation.png")).exists()


def test_validation_figure_requires_a_comparison(tmp_path: Path) -> None:
    class Empty:
        comparison = None

    with pytest.raises(ValueError, match="no SELECTED/ALL comparison"):
        plot_validation_comparison(Empty(), tmp_path / "x.png")
