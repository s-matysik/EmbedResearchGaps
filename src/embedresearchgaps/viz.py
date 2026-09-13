"""Figures for a pipeline run and for a validation study.

Every function builds its own ``Figure``, writes it to ``path`` and returns
the path, so figures can be produced head-lessly in a script or a notebook.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .results import PipelineResult

__all__ = [
    "PALETTE",
    "embed_2d",
    "plot_keyword_map",
    "plot_article_map",
    "plot_cooccurrence_network",
    "plot_centroid_heatmap",
    "plot_gap_summary",
    "plot_year_distribution",
    "plot_k_selection",
    "plot_validation_comparison",
    "generate_figures",
]

#: Colour-blind-safe qualitative palette (Okabe-Ito, extended).
PALETTE: tuple[str, ...] = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#F0E442", "#666666", "#8C564B", "#17BECF",
)


def _colour(index: int) -> str:
    return PALETTE[int(index) % len(PALETTE)]


def _save(fig: plt.Figure, path: str | Path, dpi: int = 300) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(destination)


def embed_2d(matrix: np.ndarray, random_state: int = 42, perplexity: float | None = None) -> np.ndarray:
    """Project embeddings to two dimensions with t-SNE.

    Falls back to PCA for fewer than five points, where t-SNE perplexity
    cannot be set meaningfully.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[0] < 5:
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=random_state).fit_transform(matrix)
    from sklearn.manifold import TSNE

    effective = perplexity or min(15.0, max(2.0, (matrix.shape[0] - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=effective,
        random_state=random_state,
        init="pca",
    ).fit_transform(matrix)


def plot_keyword_map(
    result: PipelineResult,
    path: str | Path,
    annotate_gaps: bool = True,
    max_labels: int = 15,
) -> str:
    """t-SNE map of the keyword space, coloured by cluster.

    Candidate gaps are outlined and labelled, which makes the core/periphery
    structure the method exploits visible in a single panel.
    """
    if "keywords" not in result.embeddings or not result.keyword_index:
        raise ValueError("result carries no keyword embeddings")
    coords = embed_2d(result.embeddings["keywords"], result.config.random_state)
    keywords = result.keyword_index

    cluster_column = "cluster" if result.mode == "keywords_first" else "article_cluster"
    lookup = (
        result.keywords.groupby("keyword")[cluster_column].agg(
            lambda s: Counter(s).most_common(1)[0][0]
        )
        if cluster_column in result.keywords.columns
        else pd.Series(dtype=int)
    )
    frequency = (
        result.keywords.groupby("keyword")["frequency"].max()
        if "frequency" in result.keywords.columns
        else pd.Series(dtype=int)
    )
    gap_keywords = set(result.gaps["keyword"]) if not result.gaps.empty else set()

    fig, ax = plt.subplots(figsize=(11, 8))
    clusters = sorted({int(lookup.get(kw, 0)) for kw in keywords})
    for cluster in clusters:
        idx = [i for i, kw in enumerate(keywords) if int(lookup.get(kw, 0)) == cluster]
        if not idx:
            continue
        sizes = [30 + 25 * float(frequency.get(keywords[i], 1)) for i in idx]
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            s=sizes, color=_colour(cluster), alpha=0.75, linewidths=0.4,
            edgecolors="white",
            label=f"C{cluster}: {result.cluster_labels.get(cluster, '')}"[:48],
        )

    if annotate_gaps and gap_keywords:
        labelled = 0
        for i, keyword in enumerate(keywords):
            if keyword not in gap_keywords:
                continue
            ax.scatter(coords[i, 0], coords[i, 1], s=170, facecolors="none",
                       edgecolors="black", linewidths=1.4, zorder=3)
            if labelled < max_labels:
                ax.annotate(
                    keyword, (coords[i, 0], coords[i, 1]), fontsize=8,
                    xytext=(5, 4), textcoords="offset points", zorder=4,
                )
                labelled += 1

    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title(
        f"Author-keyword semantic space ({result.mode}); "
        f"{len(gap_keywords)} candidate gaps outlined"
    )
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def plot_article_map(result: PipelineResult, path: str | Path, max_labels: int = 0) -> str:
    """t-SNE map of the article space, coloured by article cluster."""
    if "articles" not in result.embeddings:
        raise ValueError("result carries no article embeddings (keywords-first mode)")
    coords = embed_2d(result.embeddings["articles"], result.config.random_state)
    labels = result.articles["cluster"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 8))
    for cluster in sorted(set(labels.tolist())):
        mask = labels == cluster
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=90, color=_colour(cluster),
            alpha=0.8, edgecolors="white", linewidths=0.5,
            label=f"C{cluster}: {result.cluster_labels.get(int(cluster), '')} (n={int(mask.sum())})"[:52],
        )
    for i in range(min(max_labels, len(coords))):
        ax.annotate(
            str(result.articles["title"].iloc[i])[:40],
            (coords[i, 0], coords[i, 1]), fontsize=6, alpha=0.7,
            xytext=(4, 3), textcoords="offset points",
        )
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title("Article semantic space by cluster")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def plot_cooccurrence_network(
    result: PipelineResult,
    path: str | Path,
    min_edge_weight: int = 1,
    min_label_degree: int = 3,
) -> str:
    """Author-keyword co-occurrence network, coloured by dominant cluster."""
    import networkx as nx

    cluster_column = "cluster" if result.mode == "keywords_first" else "article_cluster"
    lookup = (
        result.keywords.groupby("keyword")[cluster_column].agg(
            lambda s: Counter(s).most_common(1)[0][0]
        )
        if cluster_column in result.keywords.columns
        else pd.Series(dtype=int)
    )

    weights: Counter = Counter()
    for keywords in result.articles["keywords_normalized"]:
        unique = sorted(set(keywords))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                weights[(unique[i], unique[j])] += 1

    graph = nx.Graph()
    for (left, right), weight in weights.items():
        if weight >= min_edge_weight:
            graph.add_edge(left, right, weight=weight)
    if graph.number_of_nodes() == 0:
        raise ValueError("no keyword co-occurrences to plot")

    fig, ax = plt.subplots(figsize=(12, 9))
    positions = nx.spring_layout(graph, k=0.55, iterations=120, seed=result.config.random_state)
    degrees = dict(graph.degree())
    edge_weights = [graph[u][v]["weight"] for u, v in graph.edges]
    max_weight = max(edge_weights) if edge_weights else 1
    nx.draw_networkx_edges(
        graph, positions, ax=ax, alpha=0.25, edge_color="#999999",
        width=[0.4 + 2.2 * w / max_weight for w in edge_weights],
    )
    nx.draw_networkx_nodes(
        graph, positions, ax=ax,
        node_size=[70 + 55 * degrees[n] for n in graph.nodes],
        node_color=[_colour(int(lookup.get(n, 0))) for n in graph.nodes],
        alpha=0.85, linewidths=0.4, edgecolors="white",
    )
    nx.draw_networkx_labels(
        graph, positions, ax=ax, font_size=8,
        labels={n: n for n in graph.nodes if degrees[n] >= min_label_degree},
    )
    clusters = sorted({int(lookup.get(n, 0)) for n in graph.nodes})
    ax.legend(
        handles=[
            Patch(facecolor=_colour(c), label=f"C{c}: {result.cluster_labels.get(c, '')}"[:46])
            for c in clusters
        ],
        loc="best", fontsize=8, framealpha=0.9,
    )
    ax.set_title(
        f"Author-keyword co-occurrence network "
        f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
    )
    ax.axis("off")
    return _save(fig, path)


def plot_centroid_heatmap(result: PipelineResult, path: str | Path) -> str:
    """Cosine similarity between cluster centroids, equation (11)."""
    if result.centroid_similarity is None:
        raise ValueError("result carries no centroid similarity matrix")
    matrix = np.asarray(result.centroid_similarity)
    n = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(1.1 * n + 3.2, 1.0 * n + 2.4))
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=float(np.min(matrix)), vmax=1.0)
    ax.set_xticks(range(n), [f"C{i}" for i in range(n)])
    ax.set_yticks(range(n), [f"C{i}" for i in range(n)])
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if matrix[i, j] > 0.5 * (1 + float(np.min(matrix))) else "black",
            )
    fig.colorbar(image, ax=ax, label="cosine similarity", fraction=0.046)
    ax.set_title("Inter-cluster thematic overlap")
    return _save(fig, path)


def plot_gap_summary(result: PipelineResult, path: str | Path, top_n: int = 12) -> str:
    """Gap counts per type and the highest-scoring candidates."""
    if result.gaps.empty:
        raise ValueError("no gaps to plot")
    gaps = result.gaps
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), width_ratios=[1, 1.5])

    counts = gaps["gap_type"].value_counts()
    axes[0].bar(
        range(len(counts)), counts.to_numpy(),
        color=[_colour(i) for i in range(len(counts))], edgecolor="black", linewidth=0.5,
    )
    axes[0].set_xticks(range(len(counts)), [t.replace(" ", "\n") for t in counts.index], fontsize=9)
    axes[0].set_ylabel("number of candidate gaps")
    axes[0].set_title("Candidates by gap type")
    for i, value in enumerate(counts.to_numpy()):
        axes[0].text(i, value, str(int(value)), ha="center", va="bottom", fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    top = gaps.nlargest(min(top_n, len(gaps)), "score")
    type_order = list(counts.index)
    colours = [_colour(type_order.index(t)) for t in top["gap_type"]]
    axes[1].barh(range(len(top)), top["score"].to_numpy(), color=colours,
                 edgecolor="black", linewidth=0.4)
    axes[1].set_yticks(range(len(top)), [str(k)[:44] for k in top["keyword"]], fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("composite score" if result.mode == "articles_first" else "centroid distance")
    axes[1].set_title(f"Top {len(top)} candidates")
    axes[1].legend(
        handles=[Patch(facecolor=_colour(i), label=t) for i, t in enumerate(type_order)],
        loc="lower right", fontsize=8,
    )
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def plot_year_distribution(result: PipelineResult, path: str | Path) -> str:
    """Publication year per cluster, used to read thematic evolution."""
    if "year" not in result.articles.columns or result.articles["year"].isna().all():
        raise ValueError("corpus carries no publication years")
    cluster_column = "cluster" if "cluster" in result.articles.columns else None
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if cluster_column is None:
        years = result.articles["year"].dropna().astype(int).value_counts().sort_index()
        ax.plot(years.index, years.to_numpy(), "o-", color=_colour(0), linewidth=2)
    else:
        for cluster in sorted(result.articles[cluster_column].dropna().unique()):
            subset = result.articles[result.articles[cluster_column] == cluster]["year"].dropna()
            if subset.empty:
                continue
            years = subset.astype(int).value_counts().sort_index()
            ax.plot(
                years.index, years.to_numpy(), "o-", color=_colour(int(cluster)),
                linewidth=1.8, markersize=5,
                label=f"C{int(cluster)}: {result.cluster_labels.get(int(cluster), '')}"[:44],
            )
        ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("publication year")
    ax.set_ylabel("number of articles")
    ax.set_title("Publication years by cluster")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def plot_k_selection(result: PipelineResult, path: str | Path) -> str:
    """WCSS elbow and silhouette curve behind the chosen number of clusters."""
    selection = result.diagnostics.get("k_selection")
    if not selection or len(selection.get("wcss") or []) < 2:
        raise ValueError("run carries no k-selection trace (k was fixed)")
    k_values = selection["k_range"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, selection["wcss"], "o-", color=_colour(0), label="WCSS")
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("within-cluster sum of squares", color=_colour(0))
    ax.tick_params(axis="y", labelcolor=_colour(0))
    twin = ax.twinx()
    twin.plot(k_values, selection["silhouette"], "s--", color=_colour(1), label="silhouette")
    twin.set_ylabel("silhouette coefficient", color=_colour(1))
    twin.tick_params(axis="y", labelcolor=_colour(1))
    ax.axvline(selection["k"], color="black", linestyle=":", linewidth=1.4)
    ax.set_title(f"Selected k = {selection['k']} ({selection['rule']})")
    ax.spines[["top"]].set_visible(False)
    return _save(fig, path)


def plot_validation_comparison(report: Any, path: str | Path) -> str:
    """GAP classification rate and mean Novelty Score, SELECTED vs ALL."""
    comparison = getattr(report, "comparison", None)
    if comparison is None:
        raise ValueError("validation report carries no SELECTED/ALL comparison")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["GAP", "EXPLORED", "METHOD", "LOW IMPORTANCE", "NOISE"]
    selected = [100 * comparison.selected_rates.get(k, 0.0) for k in labels]
    everything = [100 * comparison.all_rates.get(k, 0.0) for k in labels]
    x = np.arange(len(labels))
    axes[0].bar(x - 0.2, selected, 0.4, label="SELECTED", color=_colour(0), edgecolor="black", linewidth=0.4)
    axes[0].bar(x + 0.2, everything, 0.4, label="ALL", color=_colour(1), edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(x, [l.replace(" ", "\n") for l in labels], fontsize=8)
    axes[0].set_ylabel("share of judgements (%)")
    axes[0].set_title("Label distribution")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    means = [comparison.selected_novelty.get("mean", np.nan), comparison.all_novelty.get("mean", np.nan)]
    errors = [comparison.selected_novelty.get("std", np.nan), comparison.all_novelty.get("std", np.nan)]
    axes[1].bar(
        ["SELECTED", "ALL"], means, yerr=errors, capsize=6,
        color=[_colour(0), _colour(1)], edgecolor="black", linewidth=0.4, width=0.55,
    )
    for i, value in enumerate(means):
        if not np.isnan(value):
            axes[1].text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("mean Novelty Score (1-10)")
    axes[1].set_title(
        f"Novelty: {comparison.novelty_increase_pct:+.1f}% "
        f"({comparison.n_annotators} annotators)"
    )
    axes[1].set_ylim(0, 10)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def generate_figures(
    result: PipelineResult,
    directory: str | Path,
    prefix: str = "fig",
    include: Sequence[str] | None = None,
) -> dict[str, str]:
    """Render every applicable figure for ``result``.

    Figures that do not apply to the run (no article embeddings in
    keywords-first mode, no publication years, no gaps) are skipped and
    reported under the ``'skipped'`` key rather than raising.
    """
    out = Path(directory)
    builders = {
        "keyword_map": lambda p: plot_keyword_map(result, p),
        "article_map": lambda p: plot_article_map(result, p),
        "cooccurrence": lambda p: plot_cooccurrence_network(result, p),
        "centroid_heatmap": lambda p: plot_centroid_heatmap(result, p),
        "gap_summary": lambda p: plot_gap_summary(result, p),
        "year_distribution": lambda p: plot_year_distribution(result, p),
        "k_selection": lambda p: plot_k_selection(result, p),
    }
    wanted = list(include) if include else list(builders)
    written: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for name in wanted:
        try:
            written[name] = builders[name](out / f"{prefix}_{name}.png")
        except (ValueError, KeyError) as exc:
            skipped[name] = str(exc)
    if skipped:
        written["skipped"] = "; ".join(f"{k}: {v}" for k, v in skipped.items())
    return written
