"""Semantic ranking, k-selection, k-means and cluster diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from embedresearchgaps.clustering import (
    centroid_similarity_matrix,
    cluster_quality,
    distances_to_centroid,
    fit_kmeans,
    periphery_threshold,
    rank_by_similarity,
    select_k,
)
from embedresearchgaps.config import ClusteringConfig
from embedresearchgaps.encoders import l2_normalize


def three_blob_matrix(per_blob: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = np.array([[5.0, 0.0], [0.0, 5.0], [-5.0, -5.0]])
    return np.vstack([c + rng.normal(0, 0.35, (per_blob, 2)) for c in centres])


class TestRankBySimilarity:
    def test_identical_vector_has_zero_distance(self) -> None:
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        order, distances = rank_by_similarity(np.array([1.0, 0.0]), matrix)
        assert order[0] == 0
        assert distances[0] == pytest.approx(0.0)
        assert distances[1] == pytest.approx(1.0)
        assert distances[2] == pytest.approx(2.0)

    def test_ordering_is_ascending_in_distance(self) -> None:
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(20, 8))
        order, distances = rank_by_similarity(rng.normal(size=8), matrix)
        assert np.all(np.diff(distances[order]) >= -1e-12)

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension mismatch"):
            rank_by_similarity(np.ones(3), np.ones((5, 4)))


class TestSelectK:
    def test_recovers_three_blobs(self) -> None:
        result = select_k(three_blob_matrix(), ClusteringConfig(k_min=2, k_max=6))
        assert result.k == 3
        assert result.k_silhouette == 3
        assert len(result.wcss) == len(result.k_range)

    def test_wcss_decreases_with_k(self) -> None:
        result = select_k(three_blob_matrix(), ClusteringConfig(k_min=2, k_max=6))
        assert np.all(np.diff(result.wcss) <= 1e-6)

    def test_fixed_selection_short_circuits(self) -> None:
        config = ClusteringConfig(n_clusters=4, k_selection="fixed")
        result = select_k(three_blob_matrix(), config)
        assert (result.k, result.rule) == (4, "fixed")

    def test_fixed_without_n_clusters_raises(self) -> None:
        with pytest.raises(ValueError, match="requires n_clusters"):
            select_k(three_blob_matrix(), ClusteringConfig(k_selection="fixed"))

    def test_single_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 samples"):
            select_k(np.zeros((1, 3)), ClusteringConfig())

    def test_elbow_rule_used_when_silhouette_is_poor(self) -> None:
        rng = np.random.default_rng(11)
        uniform = rng.uniform(size=(40, 5))  # no cluster structure
        result = select_k(
            uniform, ClusteringConfig(k_min=2, k_max=8, silhouette_floor=0.9)
        )
        assert result.rule.startswith("elbow")


class TestFitKmeans:
    def test_partition_and_distances(self) -> None:
        matrix = three_blob_matrix()
        assignment = fit_kmeans(matrix, ClusteringConfig(k_min=2, k_max=6, distance_metric="euclidean"))
        assert assignment.n_clusters == 3
        assert sorted(assignment.sizes().values()) == [12, 12, 12]
        assert assignment.distances.shape == (len(matrix),)
        assert assignment.quality["silhouette"] > 0.8

    def test_explicit_n_clusters_overrides_selection(self) -> None:
        assignment = fit_kmeans(three_blob_matrix(), ClusteringConfig(), n_clusters=2)
        assert assignment.n_clusters == 2
        assert assignment.selection is None

    def test_is_deterministic_for_a_fixed_seed(self) -> None:
        matrix = three_blob_matrix()
        first = fit_kmeans(matrix, ClusteringConfig(random_state=7), n_clusters=3)
        second = fit_kmeans(matrix, ClusteringConfig(random_state=7), n_clusters=3)
        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_allclose(first.distances, second.distances)

    def test_k_is_clipped_to_sample_count(self) -> None:
        assignment = fit_kmeans(np.eye(3), ClusteringConfig(), n_clusters=10)
        assert assignment.n_clusters == 3


class TestDistances:
    def test_cosine_and_euclidean_agree_on_ordering_for_unit_vectors(self) -> None:
        rng = np.random.default_rng(5)
        matrix = l2_normalize(rng.normal(size=(30, 6)))
        centroid = matrix.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        cosine = distances_to_centroid(matrix, centroid, "cosine")
        euclidean = distances_to_centroid(matrix, centroid, "euclidean")
        np.testing.assert_allclose(euclidean**2, 2.0 * cosine, atol=1e-10)
        np.testing.assert_array_equal(np.argsort(cosine), np.argsort(euclidean))

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown distance metric"):
            distances_to_centroid(np.eye(2), np.ones(2), "manhattan")  # type: ignore[arg-type]


def test_cluster_quality_undefined_for_single_cluster() -> None:
    assert cluster_quality(np.eye(4), np.zeros(4, dtype=int)) == {}


def test_centroid_similarity_matrix_is_symmetric_with_unit_diagonal() -> None:
    matrix = centroid_similarity_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    np.testing.assert_allclose(matrix, matrix.T)
    assert matrix[0, 1] == pytest.approx(0.0)


class TestPeripheryThreshold:
    def test_percentile_matches_numpy(self) -> None:
        distances = np.arange(100, dtype=float)
        assert periphery_threshold(distances, 95) == pytest.approx(np.percentile(distances, 95))

    def test_flags_about_five_percent_at_p95(self) -> None:
        rng = np.random.default_rng(1)
        distances = rng.gamma(2.0, 0.1, size=2000)
        threshold = periphery_threshold(distances, 95)
        assert (distances >= threshold).mean() == pytest.approx(0.05, abs=0.005)

    def test_empty_input_is_never_exceeded(self) -> None:
        assert periphery_threshold(np.array([]), 95) == float("inf")
