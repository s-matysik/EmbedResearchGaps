"""Result container shared by both pipeline modes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RunConfig
from .gaps import Gap, gaps_to_frame

__all__ = ["PipelineResult"]


@dataclass
class PipelineResult:
    """Everything one pipeline run produced.

    Attributes
    ----------
    mode:
        ``'keywords_first'`` or ``'articles_first'``.
    articles:
        The analysis corpus, with a ``cluster`` column in articles-first mode
        and a ``rank``/``query_distance`` column when a problem description
        was supplied.
    keywords:
        One row per unique author keyword with frequency, cluster, centroid
        distance, centrality and TF-IDF salience.
    gaps:
        Candidate gaps, flattened by :func:`~embedresearchgaps.gaps.gaps_to_frame`.
    cluster_labels:
        Human-readable label per cluster.
    centroid_similarity:
        Pairwise cosine similarity of cluster centroids, equation (11).
    diagnostics:
        Cluster-quality indices, k-selection trace, filter counts and corpus
        statistics -- everything needed to report the run in a paper.
    """

    mode: str
    config: RunConfig
    articles: pd.DataFrame
    keywords: pd.DataFrame
    gaps: pd.DataFrame
    gap_objects: list[Gap] = field(default_factory=list)
    cluster_labels: dict[int, str] = field(default_factory=dict)
    centroid_similarity: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    #: Embedding matrices kept for visualisation: ``'articles'`` (row-aligned
    #: with :attr:`articles`) and ``'keywords'`` (aligned with
    #: :attr:`keyword_index`).  Never written to CSV.
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    keyword_index: list[str] = field(default_factory=list)

    @property
    def n_gaps(self) -> int:
        return len(self.gaps)

    @property
    def n_clusters(self) -> int:
        return int(self.diagnostics.get("n_clusters", 0))

    def top_gaps(self, n: int = 10) -> pd.DataFrame:
        """The ``n`` highest-scoring gaps across all types."""
        if self.gaps.empty:
            return self.gaps
        return self.gaps.nlargest(n, "score").reset_index(drop=True)

    def gap_type_counts(self) -> dict[str, int]:
        if self.gaps.empty:
            return {}
        return self.gaps["gap_type"].value_counts().to_dict()

    def summary(self) -> dict[str, Any]:
        """Compact, JSON-serialisable description of the run."""
        return {
            "mode": self.mode,
            "n_articles": int(len(self.articles)),
            "n_keywords": int(len(self.keywords)),
            "n_clusters": self.n_clusters,
            "n_gaps": self.n_gaps,
            "gap_types": self.gap_type_counts(),
            "cluster_labels": {int(k): v for k, v in self.cluster_labels.items()},
            "diagnostics": self.diagnostics,
        }

    def to_json(self, indent: int = 2) -> str:
        def _default(obj: Any) -> Any:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (set, frozenset, tuple)):
                return list(obj)
            return str(obj)

        return json.dumps(self.summary(), indent=indent, default=_default, ensure_ascii=False)

    def save(self, directory: str | Path, prefix: str | None = None) -> dict[str, str]:
        """Write tables, run configuration and summary to ``directory``.

        Returns a mapping of logical name to written path.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{prefix}_" if prefix else ""
        written: dict[str, str] = {}

        exports = {
            "gaps": self.gaps,
            "keywords": self.keywords,
            "articles": self.articles.drop(
                columns=[c for c in ("article_embedding",) if c in self.articles.columns]
            ),
        }
        for name, frame in exports.items():
            path = out / f"{stem}{name}.csv"
            frame.to_csv(path, index=False)
            written[name] = str(path)

        if self.centroid_similarity is not None:
            path = out / f"{stem}centroid_similarity.csv"
            pd.DataFrame(
                self.centroid_similarity,
                index=[f"C{i}" for i in range(len(self.centroid_similarity))],
                columns=[f"C{i}" for i in range(len(self.centroid_similarity))],
            ).to_csv(path)
            written["centroid_similarity"] = str(path)

        path = out / f"{stem}run_config.json"
        path.write_text(self.config.to_json(), encoding="utf-8")
        written["config"] = str(path)

        path = out / f"{stem}summary.json"
        path.write_text(self.to_json(), encoding="utf-8")
        written["summary"] = str(path)
        return written
