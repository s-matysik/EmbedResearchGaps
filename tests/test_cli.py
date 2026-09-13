"""The command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import build_synthetic_frame
from typer.testing import CliRunner

from embedresearchgaps import __version__
from embedresearchgaps.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def corpus_csv(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("cli") / "corpus.csv"
    build_synthetic_frame().to_csv(path, index=False)
    return path


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_encoders_command_lists_backends() -> None:
    result = runner.invoke(app, ["encoders"])
    assert result.exit_code == 0
    assert "tfidf-svd" in result.stdout


def test_run_articles_first(corpus_csv: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run", str(corpus_csv), "--outdir", str(tmp_path), "--mode", "articles_first",
            "--encoder", "tfidf-svd", "--clusters", "3", "--no-figures",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "candidate gaps" in result.stdout
    gaps = tmp_path / "articles_first_gaps.csv"
    assert gaps.exists()
    config = json.loads((tmp_path / "articles_first_run_config.json").read_text())
    assert config["clustering"]["n_clusters"] == 3


def test_run_keywords_first_with_figures(corpus_csv: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run", str(corpus_csv), "-o", str(tmp_path), "-m", "keywords_first",
            "-k", "3", "--percentile", "80", "--max-frequency", "2",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "keywords_first_gaps.csv").exists()
    assert list(tmp_path.glob("keywords_first_fig_*.png"))


def test_run_with_problem_file(corpus_csv: Path, tmp_path: Path) -> None:
    problem = tmp_path / "problem.txt"
    problem.write_text("credit risk lending default bank capital", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "run", str(corpus_csv), "-o", str(tmp_path), "--problem-file", str(problem),
            "--top-n", "20", "-k", "2", "--no-figures",
        ],
    )
    assert result.exit_code == 0, result.stdout
    summary = json.loads((tmp_path / "articles_first_summary.json").read_text())
    assert summary["n_articles"] == 20
    assert summary["diagnostics"]["ranking"]["ranked"] is True


def test_unknown_mode_is_rejected(corpus_csv: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["run", str(corpus_csv), "-o", str(tmp_path), "-m", "topics", "--no-figures"]
    )
    assert result.exit_code != 0


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", str(tmp_path / "absent.csv")])
    assert result.exit_code != 0


def test_token_filter_can_be_disabled(corpus_csv: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run", str(corpus_csv), "-o", str(tmp_path), "-m", "articles_first",
            "-k", "3", "--tokens", "0", "--no-figures",
        ],
    )
    assert result.exit_code == 0, result.stdout
    config = json.loads((tmp_path / "articles_first_run_config.json").read_text())
    assert config["gaps"]["gap_token_count"] is None
