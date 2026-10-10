"""Command-line interface: ``embedresearchgaps``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import ClusteringConfig, EncoderConfig, GapConfig, RunConfig
from .corpus import load_csv
from .external import load_external_ranking
from .pipelines import articles_first, keywords_first
from .viz import generate_figures

app = typer.Typer(
    add_completion=False,
    help="Embedding-driven identification of research gaps in author-keyword space.",
)


def _build_config(
    mode: str,
    top_n: int,
    problem: Optional[str],
    encoder: str,
    model: Optional[str],
    api_key_env: Optional[str],
    n_clusters: Optional[int],
    percentile: float,
    max_gaps: int,
    min_tfidf: float,
    max_frequency: Optional[int],
    token_count: Optional[int],
    min_citations: Optional[int],
    max_citations: Optional[int],
    min_citations_per_year: Optional[float],
    citation_year: Optional[int],
    drop_uncited: bool,
    seed: int,
) -> RunConfig:
    encoder_config = EncoderConfig(
        name=encoder, model=model, api_key_env=api_key_env
    )
    return RunConfig(
        top_n_articles=top_n,
        problem_description=problem,
        article_encoder=encoder_config,
        keyword_encoder=encoder_config,
        clustering=ClusteringConfig(
            n_clusters=n_clusters,
            k_selection="fixed" if n_clusters else "auto",
            random_state=seed,
        ),
        gaps=GapConfig(
            periphery_percentile=percentile,
            max_gaps_per_cluster=max_gaps,
            min_tfidf=min_tfidf,
            max_keyword_frequency=max_frequency,
            gap_token_count=token_count,
            min_citations=min_citations,
            max_citations=max_citations,
            min_citations_per_year=min_citations_per_year,
            citation_year_reference=citation_year,
            keep_uncited_candidates=not drop_uncited,
        ),
        random_state=seed,
    )


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def encoders() -> None:
    """List the registered embedding backends."""
    from .encoders import available_encoders

    for name in available_encoders():
        typer.echo(name)


@app.command(name="run")
def run_command(
    csv: Path = typer.Argument(..., exists=True, readable=True, help="Bibliographic CSV export."),
    outdir: Path = typer.Option(Path("results"), "--outdir", "-o", help="Output directory."),
    mode: str = typer.Option(
        "articles_first", "--mode", "-m",
        help="'keywords_first' (mode A) or 'articles_first' (mode B).",
    ),
    top_n: int = typer.Option(50, "--top-n", help="Size of the analysis corpus."),
    problem: Optional[str] = typer.Option(
        None, "--problem", "-p",
        help="Research problem in natural language; enables semantic ranking.",
    ),
    problem_file: Optional[Path] = typer.Option(
        None, "--problem-file", help="Read the research problem from a text file."
    ),
    encoder: str = typer.Option("tfidf-svd", "--encoder", "-e", help="Embedding backend."),
    model: Optional[str] = typer.Option(None, "--model", help="Model name for the backend."),
    api_key_env: Optional[str] = typer.Option(
        None, "--api-key-env", help="Environment variable holding the API key."
    ),
    n_clusters: Optional[int] = typer.Option(
        None, "--clusters", "-k", help="Fix k instead of selecting it automatically."
    ),
    percentile: float = typer.Option(95.0, "--percentile", help="Periphery percentile (mode A)."),
    max_gaps: int = typer.Option(3, "--max-gaps", help="Gaps per type per cluster (mode B)."),
    min_tfidf: float = typer.Option(0.005, "--min-tfidf", help="Minimum TF-IDF salience."),
    max_frequency: Optional[int] = typer.Option(
        1, "--max-frequency",
        help="Maximum keyword frequency; omit for the proportional rule.",
    ),
    token_count: Optional[int] = typer.Option(
        2, "--tokens", help="Restrict gaps to keywords of this many tokens; 0 disables."
    ),
    min_citations: Optional[int] = typer.Option(
        None, "--min-citations",
        help="Keep only candidates whose best-cited source record has at least this many citations.",
    ),
    max_citations: Optional[int] = typer.Option(
        None, "--max-citations",
        help="Keep only candidates whose best-cited source record has at most this many citations.",
    ),
    min_citations_per_year: Optional[float] = typer.Option(
        None, "--min-citations-per-year",
        help="Keep only candidates whose best source record gains at least this many citations per year.",
    ),
    citation_year: Optional[int] = typer.Option(
        None, "--citation-year",
        help="Year the citation counts are current for; defaults to the most recent year in the corpus.",
    ),
    drop_uncited: bool = typer.Option(
        False, "--drop-uncited/--keep-uncited",
        help="With a citation bound set, also drop candidates whose citation support is unknown.",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    figures: bool = typer.Option(True, "--figures/--no-figures", help="Render figures."),
) -> None:
    """Run one pipeline mode on a CSV export and write tables and figures."""
    if problem_file is not None:
        problem = problem_file.read_text(encoding="utf-8").strip()

    config = _build_config(
        mode, top_n, problem, encoder, model, api_key_env, n_clusters,
        percentile, max_gaps, min_tfidf, max_frequency,
        None if token_count in (0, None) else token_count,
        min_citations, max_citations, min_citations_per_year, citation_year,
        drop_uncited, seed,
    )

    corpus = load_csv(csv)
    typer.echo(f"corpus: {json.dumps(corpus.summary(), default=str)}")

    if mode == "keywords_first":
        result = keywords_first(corpus, config)
    elif mode == "articles_first":
        result = articles_first(corpus, config)
    else:
        raise typer.BadParameter("mode must be 'keywords_first' or 'articles_first'")

    written = result.save(outdir, prefix=mode)
    if figures:
        written.update(generate_figures(result, outdir, prefix=f"{mode}_fig"))

    typer.echo(
        f"clusters: {result.n_clusters}  candidate gaps: {result.n_gaps}  "
        f"types: {json.dumps(result.gap_type_counts())}"
    )
    for name, path in written.items():
        typer.echo(f"  {name}: {path}")


@app.command(name="external")
def external_command(
    ranking: Path = typer.Argument(
        ..., exists=True, readable=True,
        help="Ranking produced elsewhere: CSV, TSV or Excel export.",
    ),
    corpus_csv: Optional[Path] = typer.Option(
        None, "--corpus", "-c", exists=True, readable=True,
        help="Corpus the ranking refers to; attaches frequency, source records "
             "and citations, and enables the control set.",
    ),
    outdir: Path = typer.Option(Path("results"), "--outdir", "-o", help="Output directory."),
    keyword_column: Optional[str] = typer.Option(
        None, "--keyword-column", help="Keyword column; resolved from aliases when omitted."
    ),
    score_column: Optional[str] = typer.Option(
        None, "--score-column", help="Score column; rank order is used when absent."
    ),
    top_n: Optional[int] = typer.Option(
        None, "--top-n", help="Keep only the first N candidates after ordering."
    ),
    name: str = typer.Option("external", "--name", help="Label for this ranking."),
) -> None:
    """Load a ranking produced outside this package for assessment.

    The result carries the same shape a pipeline run produces, so the
    validation subpackage, the control-set split and the overlap measures apply
    to it unchanged. No clustering of ours is attached: cluster is -1 and no
    partition diagnostics are produced.
    """
    corpus = load_csv(corpus_csv) if corpus_csv is not None else None
    result = load_external_ranking(
        ranking, corpus,
        keyword_column=keyword_column, score_column=score_column,
        top_n=top_n, name=name,
    )
    info = result.diagnostics["external"]
    typer.echo(
        f"external ranking {name!r}: {info['candidates']} candidates from "
        f"{info['rows_read']} rows ({info['duplicates_dropped']} duplicates dropped)"
    )
    if corpus is not None and info["unmatched_in_corpus"]:
        typer.echo(
            f"  warning: {len(info['unmatched_in_corpus'])} candidate(s) absent from the "
            f"corpus, so they carry no corpus evidence: "
            f"{', '.join(info['unmatched_in_corpus'][:5])}"
        )
    for key, path in result.save(outdir, prefix=f"external_{name}").items():
        typer.echo(f"  {key}: {path}")


def main() -> None:  # pragma: no cover - console entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
