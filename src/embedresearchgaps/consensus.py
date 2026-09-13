"""Stability-aware consensus over several k-means initialisations.

Both published procedures partition a semantic space with k-means and read
candidates off the resulting clusters.  On real bibliographic corpora the
silhouette of such a partition is typically small (0.017-0.103 across the
three corpora, two modes and four corpus sizes of the sensitivity study
shipped with this package), so the partition -- and with it the candidate set
-- depends on the initialisation.  Running one seed therefore
gives a candidate list whose reproducibility is unknown.

:func:`consensus_gaps` repeats a mode across seeds and reports, for every
candidate, the fraction of seeds that produced it.  Filtering on that support
turns an unstable single-seed list into a ranking whose reliability is
measured rather than assumed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import ClusteringConfig, RunConfig
from .corpus import Corpus
from .pipelines import run
from .results import PipelineResult

__all__ = ["ConsensusResult", "consensus_gaps", "DEFAULT_SEEDS", "jaccard"]

#: Seeds used when the caller does not supply their own.
DEFAULT_SEEDS: tuple[int, ...] = (42, 7, 123, 2026, 31337)


def jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard overlap of two candidate sets; ``nan`` when both are empty."""
    union = left | right
    return len(left & right) / len(union) if union else float("nan")


@dataclass
class ConsensusResult:
    """Candidates pooled over several seeds, with their support.

    Attributes
    ----------
    gaps:
        One row per ``(gap_type, keyword)`` with ``support`` (fraction of
        seeds that produced it), ``n_seeds_found``, score statistics and the
        cluster labels it was attributed to.
    reference:
        The run of the first seed, kept for figures and diagnostics.
    per_seed:
        Summary of every individual run, one entry per ``(seed, top_n)`` pair.
    stability:
        Mean, minimum and maximum pairwise Jaccard overlap of the candidate
        sets across runs, the overlap attributable to the seed alone versus
        to the corpus size, and the number of candidates at each support
        level.
    """

    gaps: pd.DataFrame
    reference: PipelineResult
    per_seed: list[dict[str, Any]] = field(default_factory=list)
    stability: dict[str, Any] = field(default_factory=dict)
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    min_support: float = 0.0
    corpus_sizes: tuple[int, ...] = ()

    @property
    def mode(self) -> str:
        return self.reference.mode

    def at_support(self, threshold: float) -> pd.DataFrame:
        """Candidates found in at least ``threshold`` of the runs."""
        return self.gaps[self.gaps["support"] >= threshold].reset_index(drop=True)

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "seeds": list(self.seeds),
            "corpus_sizes": list(self.corpus_sizes),
            "min_support": self.min_support,
            "n_candidates": int(len(self.gaps)),
            "n_unanimous": int((self.gaps["support"] == 1.0).sum()),
            "n_majority": int((self.gaps["support"] >= 0.5).sum()),
            "stability": self.stability,
            "per_seed": self.per_seed,
        }

    def save(self, directory: str | Path, prefix: str | None = None) -> dict[str, str]:
        import json

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{prefix}_" if prefix else ""
        written = {}
        path = out / f"{stem}consensus_gaps.csv"
        self.gaps.to_csv(path, index=False)
        written["gaps"] = str(path)
        path = out / f"{stem}consensus_summary.json"
        path.write_text(
            json.dumps(self.summary(), indent=2, default=str), encoding="utf-8"
        )
        written["summary"] = str(path)
        return written


def consensus_gaps(
    corpus: Corpus,
    mode: str = "articles_first",
    config: RunConfig | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    min_support: float = 0.0,
    top_n: Sequence[int] | None = None,
) -> ConsensusResult:
    """Run one mode across seeds -- and optionally corpus sizes -- and pool
    the candidates by support.

    Parameters
    ----------
    corpus, mode, config:
        As for :func:`~embedresearchgaps.pipelines.run`.  Only the k-means
        initialisation changes between runs; the corpus, the embeddings and
        every threshold stay fixed, so differences in the candidate set are
        attributable to the partition alone.
    seeds:
        At least two seeds; the first one provides the reference run.
    min_support:
        Drop candidates found in a smaller fraction of the runs.  ``0.5``
        keeps the majority-supported candidates, ``1.0`` only the unanimous
        ones.
    top_n:
        Analysis-corpus sizes to pool over, e.g. ``(30, 50, 100, 150)``.  The
        candidate set of both modes is as sensitive to this arbitrary cut-off
        as it is to the k-means seed, so pooling over seeds alone measures
        only half the instability.  When given, one run is executed per
        ``(seed, size)`` pair and support is the fraction of those runs that
        produced the candidate; ``stability`` then also reports the overlap
        attributable to the seed alone (pairs sharing a size) separately from
        the overlap across sizes.  ``None`` keeps ``config.top_n_articles``.

    Returns
    -------
    ConsensusResult

    Examples
    --------
    >>> result = consensus_gaps(corpus, "articles_first", config,       # doctest: +SKIP
    ...                         min_support=0.5)
    >>> result.gaps[["keyword", "gap_type", "support", "mean_score"]]   # doctest: +SKIP
    """
    # Repeating a seed reproduces an identical partition and would inflate
    # support without adding evidence, so duplicates are collapsed.
    seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if len(seeds) < 2:
        raise ValueError("consensus needs at least 2 distinct seeds")
    if not 0.0 <= min_support <= 1.0:
        raise ValueError("min_support must lie in [0, 1]")
    config = config or RunConfig()

    sizes = tuple(dict.fromkeys(int(n) for n in top_n)) if top_n else (
        int(config.top_n_articles),
    )
    if any(n < 2 for n in sizes):
        raise ValueError("every top_n value must be >= 2")
    settings = [(seed, size) for size in sizes for seed in seeds]

    frames: list[pd.DataFrame] = []
    candidate_sets: dict[tuple[int, int], set[str]] = {}
    per_seed: list[dict[str, Any]] = []
    reference: PipelineResult | None = None

    for seed, size in settings:
        seeded = dataclasses.replace(
            config,
            clustering=dataclasses.replace(config.clustering, random_state=seed),
            random_state=seed,
            top_n_articles=size,
        )
        result = run(corpus, mode, seeded)
        if reference is None:
            reference = result
        frame = result.gaps.copy()
        frame["seed"] = seed
        frame["top_n"] = size
        frame["run"] = f"{seed}@{size}"
        frames.append(frame)
        candidate_sets[(seed, size)] = (
            set(frame["keyword"].astype(str)) if not frame.empty else set()
        )
        per_seed.append(
            {
                "seed": seed,
                "top_n": size,
                "k": result.n_clusters,
                "n_gaps": result.n_gaps,
                "gap_types": result.gap_type_counts(),
                "silhouette": result.diagnostics.get("cluster_quality", {}).get("silhouette"),
            }
        )

    assert reference is not None
    pooled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if pooled.empty:
        gaps = pd.DataFrame(
            columns=["gap_type", "keyword", "support", "n_runs_found",
                     "n_seeds_found", "n_sizes_found", "mean_score",
                     "min_score", "max_score", "cluster_labels"]
        )
    else:
        grouped = pooled.groupby(["gap_type", "keyword"], dropna=False)
        gaps = grouped.agg(
            n_runs_found=("run", "nunique"),
            n_seeds_found=("seed", "nunique"),
            n_sizes_found=("top_n", "nunique"),
            mean_score=("score", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
            mean_frequency=("frequency", "mean"),
        ).reset_index()
        labels = grouped["cluster_label"].agg(
            lambda values: " | ".join(sorted({str(v) for v in values if pd.notna(v)}))
        )
        gaps = gaps.merge(labels.rename("cluster_labels"), on=["gap_type", "keyword"])
        # An empty per-run frame contributes object-dtype columns to the
        # concatenation, which would leave the aggregates unsortable.
        for column in ("mean_score", "min_score", "max_score", "mean_frequency"):
            gaps[column] = pd.to_numeric(gaps[column], errors="coerce")
        gaps["support"] = gaps["n_runs_found"] / len(settings)
        gaps = gaps[gaps["support"] >= min_support]
        gaps = gaps.sort_values(
            ["support", "mean_score"], ascending=False, ignore_index=True
        )

    keys = list(candidate_sets)
    overlaps = [jaccard(candidate_sets[a], candidate_sets[b]) for a, b in combinations(keys, 2)]
    finite = [value for value in overlaps if not np.isnan(value)]
    within_size = [
        jaccard(candidate_sets[a], candidate_sets[b])
        for a, b in combinations(keys, 2)
        if a[1] == b[1]
    ]
    across_size = [
        jaccard(candidate_sets[a], candidate_sets[b])
        for a, b in combinations(keys, 2)
        if a[1] != b[1]
    ]
    within_size = [v for v in within_size if not np.isnan(v)]
    across_size = [v for v in across_size if not np.isnan(v)]
    support_counts = (
        gaps["support"].round(3).value_counts().sort_index(ascending=False).to_dict()
        if not gaps.empty
        else {}
    )
    stability = {
        "n_seeds": len(seeds),
        "n_corpus_sizes": len(sizes),
        "corpus_sizes": list(sizes),
        "n_runs": len(settings),
        "mean_pairwise_jaccard": float(np.mean(finite)) if finite else float("nan"),
        "min_pairwise_jaccard": float(np.min(finite)) if finite else float("nan"),
        "max_pairwise_jaccard": float(np.max(finite)) if finite else float("nan"),
        # seed effect: pairs that share a corpus size; size effect: pairs that
        # do not.  With a single size the second is empty by construction.
        "mean_jaccard_same_size": (
            float(np.mean(within_size)) if within_size else float("nan")
        ),
        "mean_jaccard_across_sizes": (
            float(np.mean(across_size)) if across_size else float("nan")
        ),
        "candidates_per_support": {str(k): int(v) for k, v in support_counts.items()},
        "mean_candidates_per_run": float(
            np.mean([len(candidate_sets[key]) for key in keys])
        ),
    }

    return ConsensusResult(
        gaps=gaps,
        reference=reference,
        per_seed=per_seed,
        stability=stability,
        seeds=seeds,
        min_support=min_support,
        corpus_sizes=sizes,
    )
