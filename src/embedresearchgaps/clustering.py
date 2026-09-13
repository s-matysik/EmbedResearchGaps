"""Semantic ranking, k-selection, k-means clustering and cluster diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from .config import ClusteringConfig, DistanceMetric

__all__ = [
    "KSelectionResult",
    "ClusterAssignment",
    "rank_by_similarity",
    "select_k",
    "fit_kmeans",
    "distances_to_centroid",
    "cluster_quality",
    "centroid_similarity_matrix",
    "periphery_threshold",
]


def rank_by_similarity(query: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank rows of ``matrix`` by cosine distance to ``query``.

    Implements equation (1) of both source papers: ``d(u, v) = 1 - cos(u, v)``,
    bounded in ``[0, 2]``.

    Returns
    -------
    order:
        Row indices sorted by ascending distance (most similar first).
    distances:
        Cosine distance of every row to the query, in original row order.
    """
    query = np.asarray(query, dtype=np.float64).reshape(1, -1)
    matrix = np.asarray(matrix, dtype=np.float64)
    if query.shape[1] != matrix.shape[1]:
        raise ValueError(
            f"dimension mismatch: query has {query.shape[1]}, corpus has {matrix.shape[1]}"
        )
    distances = cosine_distances(query, matrix).ravel()
    return np.argsort(distances, kind="stable"), distances


@dataclass
class KSelectionResult:
    """Outcome of automatic selection of the number of clusters."""

    k: int
    k_range: list[int]
    wcss: list[float]
    silhouette: list[float]
    k_elbow: int | None
    k_silhouette: int | None
    rule: str

    def to_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "k_range": self.k_range,
            "wcss": self.wcss,
            "silhouette": self.silhouette,
            "k_elbow": self.k_elbow,
            "k_silhouette": self.k_silhouette,
            "rule": self.rule,
        }


def _elbow(k_values: Sequence[int], wcss: Sequence[float]) -> int | None:
    """Knee of the WCSS curve, or ``None`` when no knee is detectable."""
    if len(k_values) < 3:
        return None
    try:
        from kneed import KneeLocator

        locator = KneeLocator(
            list(k_values), list(wcss), curve="convex", direction="decreasing", S=1.0
        )
        return int(locator.elbow) if locator.elbow is not None else None
    except ImportError:  # pragma: no cover - kneed is a declared dependency
        # Fallback: maximum distance to the chord joining the curve endpoints.
        x = np.asarray(k_values, dtype=float)
        y = np.asarray(wcss, dtype=float)
        x_norm = (x - x.min()) / max(float(np.ptp(x)), 1e-12)
        y_norm = (y - y.min()) / max(float(np.ptp(y)), 1e-12)
        chord = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
        chord /= max(np.linalg.norm(chord), 1e-12)
        points = np.column_stack([x_norm - x_norm[0], y_norm - y_norm[0]])
        projection = np.outer(points @ chord, chord)
        return int(x[np.argmax(np.linalg.norm(points - projection, axis=1))])


def select_k(matrix: np.ndarray, config: ClusteringConfig) -> KSelectionResult:
    """Choose the number of clusters for ``matrix``.

    With ``k_selection='auto'`` the silhouette maximum is used when it
    exceeds ``silhouette_floor`` and the WCSS elbow otherwise -- the rule
    stated in the AMCIS paper.  ``'fixed'`` requires ``n_clusters``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    n_samples = matrix.shape[0]
    if n_samples < 2:
        raise ValueError("need at least 2 samples to cluster")

    if config.k_selection == "fixed" or config.n_clusters is not None:
        if config.n_clusters is None:
            raise ValueError("k_selection='fixed' requires n_clusters")
        k = min(config.n_clusters, n_samples)
        return KSelectionResult(k, [k], [], [], None, None, "fixed")

    k_max = min(config.k_max, n_samples - 1)
    k_values = [k for k in range(config.k_min, k_max + 1)]
    if not k_values:
        return KSelectionResult(min(2, n_samples), [], [], [], None, None, "degenerate")

    wcss: list[float] = []
    silhouettes: list[float] = []
    for k in k_values:
        model = KMeans(
            n_clusters=k, init="k-means++", n_init=config.n_init, random_state=config.random_state
        )
        labels = model.fit_predict(matrix)
        wcss.append(float(model.inertia_))
        silhouettes.append(
            float(silhouette_score(matrix, labels)) if len(set(labels)) > 1 else -1.0
        )

    k_elbow = _elbow(k_values, wcss)
    k_silhouette = int(k_values[int(np.argmax(silhouettes))])
    best_silhouette = max(silhouettes)

    if config.k_selection == "elbow":
        k, rule = (k_elbow or k_values[len(k_values) // 2]), "elbow"
    elif config.k_selection == "silhouette":
        k, rule = k_silhouette, "silhouette"
    elif best_silhouette < config.silhouette_floor:
        k = k_elbow or k_values[len(k_values) // 2]
        rule = f"elbow (max silhouette {best_silhouette:.3f} < {config.silhouette_floor})"
    else:
        k, rule = k_silhouette, f"silhouette ({best_silhouette:.3f})"

    return KSelectionResult(
        k=int(k),
        k_range=k_values,
        wcss=wcss,
        silhouette=silhouettes,
        k_elbow=k_elbow,
        k_silhouette=k_silhouette,
        rule=rule,
    )


@dataclass
class ClusterAssignment:
    """Cluster labels, centroids and per-point centroid distances."""

    labels: np.ndarray
    centroids: np.ndarray
    distances: np.ndarray
    n_clusters: int
    inertia: float
    metric: DistanceMetric
    selection: KSelectionResult | None = None
    quality: dict[str, float] = field(default_factory=dict)

    def sizes(self) -> dict[int, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}


def distances_to_centroid(
    matrix: np.ndarray,
    centroid: np.ndarray,
    metric: DistanceMetric = "cosine",
) -> np.ndarray:
    """Distance of every row of ``matrix`` to ``centroid``.

    ``'cosine'`` returns ``1 - cos(x, c)``, the scale used in the published
    tables of peripheral keywords.  ``'euclidean'`` returns the L2 distance.
    For L2-normalised embeddings the two orderings coincide, because
    ``||x - c||^2 = 2 (1 - cos(x, c))`` when both vectors are unit length.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    centroid = np.asarray(centroid, dtype=np.float64).reshape(1, -1)
    if metric == "cosine":
        return cosine_distances(matrix, centroid).ravel()
    if metric == "euclidean":
        return np.linalg.norm(matrix - centroid, axis=1)
    raise ValueError(f"unknown distance metric {metric!r}")


def fit_kmeans(
    matrix: np.ndarray,
    config: ClusteringConfig,
    n_clusters: int | None = None,
) -> ClusterAssignment:
    """Cluster ``matrix`` with k-means++ and describe the partition.

    ``n_clusters`` overrides both ``config.n_clusters`` and automatic
    selection; it is used to cluster keyword sub-spaces with a size-derived k.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    n_samples = matrix.shape[0]
    if n_samples < 2:
        raise ValueError("need at least 2 samples to cluster")

    selection: KSelectionResult | None = None
    if n_clusters is None:
        selection = select_k(matrix, config)
        n_clusters = selection.k
    n_clusters = int(max(2, min(n_clusters, n_samples)))

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=config.n_init,
        random_state=config.random_state,
    )
    labels = model.fit_predict(matrix)

    distances = np.zeros(n_samples, dtype=np.float64)
    for cluster in range(n_clusters):
        mask = labels == cluster
        if not mask.any():
            continue
        distances[mask] = distances_to_centroid(
            matrix[mask], model.cluster_centers_[cluster], config.distance_metric
        )

    return ClusterAssignment(
        labels=labels,
        centroids=model.cluster_centers_,
        distances=distances,
        n_clusters=n_clusters,
        inertia=float(model.inertia_),
        metric=config.distance_metric,
        selection=selection,
        quality=cluster_quality(matrix, labels),
    )


def cluster_quality(matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Silhouette, Davies-Bouldin and Calinski-Harabasz of a partition.

    Returns an empty dict for degenerate partitions -- a single cluster, or as
    many clusters as points -- where these indices are undefined.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(labels)
    n_labels = len(set(labels.tolist()))
    if not 1 < n_labels < len(labels):
        return {}
    return {
        "silhouette": float(silhouette_score(matrix, labels)),
        "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
    }


def centroid_similarity_matrix(centroids: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between cluster centroids, equation (11)."""
    return cosine_similarity(np.asarray(centroids, dtype=np.float64))


def periphery_threshold(distances: np.ndarray, percentile: float) -> float:
    """Distance above which a point is peripheral inside its own cluster.

    Computed per cluster so that clusters of different semantic density are
    treated on their own scale rather than against a global cut-off.
    """
    distances = np.asarray(distances, dtype=np.float64)
    if distances.size == 0:
        return float("inf")
    return float(np.percentile(distances, percentile))
