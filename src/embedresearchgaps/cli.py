"""Command-line interface: ``embedresearchgaps``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import ClusteringConfig, EncoderConfig, GapConfig, RunConfig
from .corpus import load_csv
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
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    figures: bool = typer.Option(True, "--figures/--no-figures", help="Render figures."),
) -> None:
    """Run one pipeline mode on a CSV export and write tables and figures."""
    if problem_file is not None:
        problem = problem_file.read_text(encoding="utf-8").strip()

    config = _build_config(
        mode, top_n, problem, encoder, model, api_key_env, n_clusters,
        percentile, max_gaps, min_tfidf, max_frequency,
        None if token_count in (0, None) else token_count, seed,
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


def main() -> None:  # pragma: no cover - console entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
