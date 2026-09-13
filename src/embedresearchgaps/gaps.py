"""Gap candidates: scoring formulae and the four detectors.

Three detectors implement the typology of the AMCIS paper and are used by the
articles-first pipeline:

===== ============================ ===============================================
Type  Name                         Criterion
===== ============================ ===============================================
1     Emerging Concept             rare, peripheral, lexically salient keyword
2     Conceptual Combination       two established keywords that never co-occur
3     Cross-Cluster Concept        keyword present elsewhere, absent from cluster
===== ============================ ===============================================

The fourth detector, :func:`detect_peripheral_keywords`, implements the
percentile rule of the PACIS paper and is used by the keywords-first pipeline.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, asdict
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .clustering import periphery_threshold
from .config import (
    CombinationWeights,
    CrossClusterWeights,
    EmergingWeights,
    GapConfig,
)
from .keywords import KeywordSpace
from .text import TfidfSalience, is_methodological_term, normalize_keyword

__all__ = [
    "Gap",
    "GAP_TYPES",
    "COMBINATION_SEPARATOR",
    "safe_minmax",
    "zscore",
    "score_emerging",
    "score_combination",
    "score_cross_cluster",
    "candidate_mask",
    "detect_emerging_concepts",
    "detect_conceptual_combinations",
    "detect_cross_cluster_concepts",
    "detect_peripheral_keywords",
    "deduplicate_gaps",
    "gaps_to_frame",
]

#: Symbol joining the two members of a Type 2 conceptual combination.
COMBINATION_SEPARATOR = " x "

GAP_TYPES = (
    "Emerging Concept",
    "Conceptual Combination",
    "Cross-Cluster Concept",
    "Peripheral Keyword",
)


@dataclass
class Gap:
    """One candidate research gap."""

    gap_type: str
    keyword: str
    cluster: int
    score: float
    frequency: int
    rationale: str
    metrics: dict[str, float] = field(default_factory=dict)
    members: tuple[str, ...] = ()
    n_articles: int = 0
    article_titles: tuple[str, ...] = ()
    article_dois: tuple[str, ...] = ()
    citations: float | None = None
    cluster_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["members"] = list(self.members)
        payload["article_titles"] = list(self.article_titles)
        payload["article_dois"] = list(self.article_dois)
        return payload


def safe_minmax(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Min-max normalise to ``[0, 1]``; a constant vector maps to all ``0.5``.

    The neutral 0.5 for constant input keeps single-candidate clusters from
    receiving either a spuriously perfect or a spuriously zero component.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    low, high = float(np.min(array)), float(np.max(array))
    if high <= low:
        return np.full_like(array, 0.5)
    return (array - low) / (high - low)


def zscore(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Standardise to zero mean and unit variance; constant input maps to 0.

    Equation (8) of the AMCIS paper, computed over the filtered candidate
    subset rather than over all keywords.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    std = float(np.std(array))
    if std == 0.0:
        return np.zeros_like(array)
    return (array - float(np.mean(array))) / std


def score_emerging(
    distance: float,
    centrality: float,
    tfidf: float,
    rarity: float,
    weights: EmergingWeights | None = None,
) -> float:
    """Equation (7): composite score of a Type 1 Emerging Concept.

    All four arguments are expected to be min-max normalised within the
    candidate set of one cluster.
    """
    w = weights or EmergingWeights()
    return float(
        w.distance * distance + w.centrality * centrality + w.tfidf * tfidf + w.rarity * rarity
    )


def score_combination(
    similarity: float,
    frequency_1: int,
    frequency_2: int,
    mean_tfidf: float,
    weights: CombinationWeights | None = None,
    optimum: float = 0.45,
) -> float:
    """Equation (9): composite score of a Type 2 Conceptual Combination.

    The similarity term peaks at ``optimum`` and decays linearly: pairs that
    are near-synonymous carry no novelty, pairs that are unrelated carry no
    plausibility.
    """
    w = weights or CombinationWeights()
    similarity_term = 1.0 - 2.0 * abs(similarity - optimum)
    frequency_term = min((frequency_1 + frequency_2) / 10.0, 1.0)
    tfidf_term = min(10.0 * mean_tfidf, 1.0)
    return float(
        w.similarity * similarity_term + w.frequency * frequency_term + w.tfidf * tfidf_term
    )


def score_cross_cluster(
    frequency_other: int,
    presence_ratio: float,
    tfidf: float,
    tfidf_max: float,
    weights: CrossClusterWeights | None = None,
) -> float:
    """Equation (10): composite score of a Type 3 Cross-Cluster Concept."""
    w = weights or CrossClusterWeights()
    frequency_term = min(frequency_other / 5.0, 1.0)
    specificity_term = 1.0 - presence_ratio
    tfidf_term = tfidf / tfidf_max if tfidf_max > 0 else 0.0
    return float(
        w.frequency * frequency_term
        + w.specificity * specificity_term
        + w.tfidf * min(tfidf_term, 1.0)
    )


def candidate_mask(
    keywords: Sequence[str],
    config: GapConfig,
    core_terms: Iterable[str] = (),
) -> np.ndarray:
    """Boolean mask of keywords eligible to become a gap candidate.

    Applies the token-count filter, the methodological blocklist, the
    core-domain filter and any user-supplied stop keywords.
    """
    core = {normalize_keyword(t) for t in core_terms}
    stop = {normalize_keyword(t) for t in config.extra_stop_keywords}
    mask = np.ones(len(keywords), dtype=bool)
    for i, keyword in enumerate(keywords):
        normalized = normalize_keyword(keyword)
        if not normalized:
            mask[i] = False
        elif config.gap_token_count is not None and len(normalized.split()) != config.gap_token_count:
            mask[i] = False
        elif config.filter_methodological and is_methodological_term(normalized, config):
            mask[i] = False
        elif normalized in core or normalized in stop:
            mask[i] = False
    return mask


def _article_evidence(
    keyword: str,
    articles: pd.DataFrame,
    limit: int = 3,
) -> tuple[int, tuple[str, ...], tuple[str, ...], float | None]:
    """Records whose author keywords contain ``keyword``."""
    if articles.empty:
        return 0, (), (), None
    hits = articles[
        articles["keywords_normalized"].apply(lambda kws: normalize_keyword(keyword) in kws)
    ]
    if hits.empty:
        return 0, (), (), None
    citations = (
        float(hits["citations"].max())
        if "citations" in hits.columns and hits["citations"].notna().any()
        else None
    )
    return (
        len(hits),
        tuple(str(t) for t in hits["title"].head(limit)),
        tuple(str(d) for d in hits.get("doi", pd.Series(dtype=str)).head(limit)),
        citations,
    )


def detect_emerging_concepts(
    space: KeywordSpace,
    cluster_articles: pd.DataFrame,
    cluster_id: int,
    tfidf: TfidfSalience,
    global_frequency: Mapping[str, int],
    core_terms: Iterable[str],
    config: GapConfig,
) -> list[Gap]:
    """Type 1: rare, peripheral and lexically salient keywords.

    Candidates occur at most ``config.max_keyword_frequency`` times inside the
    cluster, pass :func:`candidate_mask`, and are ranked by equation (7).
    """
    frame = space.to_frame()
    max_frequency = config.max_keyword_frequency
    mask = candidate_mask(space.keywords, config, core_terms)
    if max_frequency is not None:
        mask &= frame["frequency"].to_numpy() <= max_frequency
    if not mask.any():
        return []

    candidates = frame.loc[mask].copy()
    distances = candidates["distance_from_subcentroid"].to_numpy()
    candidates["distance_zscore"] = zscore(distances)
    candidates["tfidf"] = [tfidf.get(kw) for kw in candidates["keyword"]]
    candidates["global_frequency"] = [
        int(global_frequency.get(kw, 0)) for kw in candidates["keyword"]
    ]

    components = {
        "distance": safe_minmax(candidates["distance_zscore"].to_numpy()),
        "centrality": safe_minmax(candidates["centrality"].to_numpy()),
        "tfidf": safe_minmax(candidates["tfidf"].to_numpy()),
        "rarity": 1.0 - safe_minmax(candidates["global_frequency"].to_numpy()),
    }
    weights = config.emerging_weights
    candidates["score"] = [
        score_emerging(
            components["distance"][i],
            components["centrality"][i],
            components["tfidf"][i],
            components["rarity"][i],
            weights,
        )
        for i in range(len(candidates))
    ]

    gaps: list[Gap] = []
    for _, row in candidates.nlargest(config.max_gaps_per_cluster, "score").iterrows():
        keyword = str(row["keyword"])
        n_articles, titles, dois, citations = _article_evidence(keyword, cluster_articles)
        gaps.append(
            Gap(
                gap_type="Emerging Concept",
                keyword=keyword,
                cluster=int(cluster_id),
                score=float(row["score"]),
                frequency=int(row["frequency"]),
                metrics={
                    "distance_from_subcentroid": float(row["distance_from_subcentroid"]),
                    "distance_zscore": float(row["distance_zscore"]),
                    "centrality": float(row["centrality"]),
                    "tfidf": float(row["tfidf"]),
                    "global_frequency": float(row["global_frequency"]),
                },
                n_articles=n_articles,
                article_titles=titles,
                article_dois=dois,
                citations=citations,
                rationale=(
                    f"frequency={int(row['frequency'])} in cluster, "
                    f"{int(row['global_frequency'])} in corpus, "
                    f"z={row['distance_zscore']:.2f}, tfidf={row['tfidf']:.4f}"
                ),
            )
        )
    return gaps


def detect_conceptual_combinations(
    space: KeywordSpace,
    cluster_articles: pd.DataFrame,
    cluster_id: int,
    tfidf: TfidfSalience,
    config: GapConfig,
) -> list[Gap]:
    """Type 2: established keyword pairs that never co-occur.

    Both members must occur at least twice in the cluster, must never appear
    together in one record, and their cosine similarity must fall inside
    ``config.combination_similarity_range``.
    """
    frame = space.to_frame()
    eligible = candidate_mask(space.keywords, config) & (frame["frequency"].to_numpy() >= 2)
    if eligible.sum() < 2:
        return []

    observed_pairs: set[tuple[str, str]] = set()
    for keywords in cluster_articles["keywords_normalized"]:
        unique = sorted(set(keywords))
        observed_pairs.update(combinations(unique, 2))

    members = [space.keywords[i] for i in np.flatnonzero(eligible)]
    low, high = config.combination_similarity_range
    similarity = cosine_similarity(space.embeddings)
    frequency = dict(zip(space.keywords, frame["frequency"].to_numpy()))

    scored: list[dict[str, Any]] = []
    for first, second in combinations(members, 2):
        if tuple(sorted((first, second))) in observed_pairs:
            continue
        i, j = space.index[first], space.index[second]
        sim = float(similarity[i, j])
        if not low < sim < high:
            continue
        mean_tfidf = (tfidf.get(first) + tfidf.get(second)) / 2.0
        if mean_tfidf < config.min_tfidf:
            continue
        scored.append(
            {
                "members": (first, second),
                "similarity": sim,
                "frequency_1": int(frequency[first]),
                "frequency_2": int(frequency[second]),
                "mean_tfidf": mean_tfidf,
                "score": score_combination(
                    sim,
                    int(frequency[first]),
                    int(frequency[second]),
                    mean_tfidf,
                    config.combination_weights,
                    config.combination_similarity_optimum,
                ),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return [
        Gap(
            gap_type="Conceptual Combination",
            keyword=COMBINATION_SEPARATOR.join(item["members"]),
            cluster=int(cluster_id),
            score=float(item["score"]),
            frequency=int(item["frequency_1"] + item["frequency_2"]),
            metrics={
                "similarity": item["similarity"],
                "frequency_1": float(item["frequency_1"]),
                "frequency_2": float(item["frequency_2"]),
                "mean_tfidf": item["mean_tfidf"],
            },
            members=item["members"],
            rationale=(
                f"semantically related (similarity={item['similarity']:.2f}) but never "
                f"co-occurring; mean tfidf={item['mean_tfidf']:.4f}"
            ),
        )
        for item in scored[: config.max_gaps_per_cluster]
    ]


def detect_cross_cluster_concepts(
    space: KeywordSpace,
    all_articles: pd.DataFrame,
    cluster_id: int,
    tfidf: TfidfSalience,
    global_frequency: Mapping[str, int],
    core_terms: Iterable[str],
    config: GapConfig,
) -> list[Gap]:
    """Type 3: keywords used in other clusters but absent from this one.

    The candidate must be rare in the rest of the corpus (at most
    ``config.max_frequency_other_clusters`` records) and cover less than
    ``config.max_presence_ratio`` of the corpus.
    """
    if "cluster" not in all_articles.columns:
        raise ValueError("cross-cluster detection needs a 'cluster' column")
    n_articles = len(all_articles)
    if n_articles == 0:
        return []

    others = all_articles[all_articles["cluster"] != cluster_id]
    counter: Counter = Counter()
    for keywords in others["keywords_normalized"]:
        for keyword in dict.fromkeys(keywords):
            counter[keyword] += 1

    present = set(space.keywords)
    core = {normalize_keyword(t) for t in core_terms}
    stop = {normalize_keyword(t) for t in config.extra_stop_keywords}
    tfidf_max = tfidf.max_score

    scored: list[dict[str, Any]] = []
    for keyword, frequency_other in counter.items():
        if keyword in present or keyword in core or keyword in stop:
            continue
        if config.gap_token_count is not None and len(keyword.split()) != config.gap_token_count:
            continue
        if frequency_other > config.max_frequency_other_clusters:
            continue
        if config.filter_methodological and is_methodological_term(keyword, config):
            continue
        presence_ratio = global_frequency.get(keyword, 0) / n_articles
        if presence_ratio > config.max_presence_ratio:
            continue
        keyword_tfidf = tfidf.get(keyword)
        if keyword_tfidf < config.min_tfidf:
            continue
        scored.append(
            {
                "keyword": keyword,
                "frequency_other": int(frequency_other),
                "presence_ratio": presence_ratio,
                "tfidf": keyword_tfidf,
                "score": score_cross_cluster(
                    int(frequency_other),
                    presence_ratio,
                    keyword_tfidf,
                    tfidf_max,
                    config.cross_cluster_weights,
                ),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    gaps: list[Gap] = []
    for item in scored[: config.max_gaps_per_cluster]:
        count, titles, dois, citations = _article_evidence(item["keyword"], others)
        gaps.append(
            Gap(
                gap_type="Cross-Cluster Concept",
                keyword=item["keyword"],
                cluster=int(cluster_id),
                score=float(item["score"]),
                frequency=item["frequency_other"],
                metrics={
                    "frequency_other_clusters": float(item["frequency_other"]),
                    "presence_ratio": item["presence_ratio"],
                    "tfidf": item["tfidf"],
                },
                n_articles=count,
                article_titles=titles,
                article_dois=dois,
                citations=citations,
                rationale=(
                    f"absent from cluster {cluster_id}, present in "
                    f"{item['frequency_other']} record(s) elsewhere "
                    f"({item['presence_ratio']:.1%} of corpus), tfidf={item['tfidf']:.4f}"
                ),
            )
        )
    return gaps


def detect_peripheral_keywords(
    space: KeywordSpace,
    articles: pd.DataFrame,
    tfidf: TfidfSalience,
    core_terms: Iterable[str],
    config: GapConfig,
) -> list[Gap]:
    """Percentile rule of the keywords-first mode.

    A keyword is a candidate when its distance from its own cluster centroid
    exceeds the cluster-specific ``config.periphery_percentile`` and its
    frequency does not exceed ``config.max_keyword_frequency``.  The threshold
    is computed inside each cluster, so clusters of different semantic density
    are judged on their own scale; the share of flagged keywords is therefore
    approximately ``100 - periphery_percentile`` per cent by construction.
    """
    frame = space.to_frame()
    labels = space.subcluster_labels
    distances = space.subcluster_distances
    eligible = candidate_mask(space.keywords, config, core_terms)
    if config.max_keyword_frequency is not None:
        eligible &= frame["frequency"].to_numpy() <= config.max_keyword_frequency

    thresholds: dict[int, float] = {
        int(cluster): periphery_threshold(distances[labels == cluster], config.periphery_percentile)
        for cluster in np.unique(labels)
    }

    gaps: list[Gap] = []
    for position in np.flatnonzero(eligible):
        cluster = int(labels[position])
        distance = float(distances[position])
        if distance < thresholds[cluster]:
            continue
        keyword = space.keywords[position]
        count, titles, dois, citations = _article_evidence(keyword, articles)
        keyword_tfidf = tfidf.get(keyword)
        gaps.append(
            Gap(
                gap_type="Peripheral Keyword",
                keyword=keyword,
                cluster=cluster,
                score=distance,
                frequency=int(frame["frequency"].iloc[position]),
                metrics={
                    "distance_from_centroid": distance,
                    "cluster_threshold": thresholds[cluster],
                    "centrality": float(space.centrality[position]),
                    "tfidf": keyword_tfidf,
                    "citations": float(citations) if citations is not None else float("nan"),
                },
                n_articles=count,
                article_titles=titles,
                article_dois=dois,
                citations=citations,
                rationale=(
                    f"distance {distance:.4f} exceeds the cluster-{cluster} "
                    f"p{config.periphery_percentile:g} threshold "
                    f"{thresholds[cluster]:.4f}; frequency="
                    f"{int(frame['frequency'].iloc[position])}"
                ),
            )
        )

    gaps.sort(key=lambda gap: gap.score, reverse=True)
    return gaps


def deduplicate_gaps(gaps: Sequence[Gap]) -> list[Gap]:
    """Keep the highest-scoring occurrence of every keyword across clusters."""
    best: dict[tuple[str, str], Gap] = {}
    for gap in gaps:
        key = (gap.gap_type, gap.keyword)
        if key not in best or gap.score > best[key].score:
            best[key] = gap
    return sorted(best.values(), key=lambda gap: gap.score, reverse=True)


def gaps_to_frame(gaps: Sequence[Gap]) -> pd.DataFrame:
    """Flatten gaps into a table, expanding ``metrics`` into columns."""
    if not gaps:
        return pd.DataFrame(
            columns=[
                "gap_type", "keyword", "cluster", "cluster_label", "score",
                "frequency", "n_articles", "citations", "rationale",
                "article_titles", "article_dois",
            ]
        )
    rows: list[dict[str, Any]] = []
    for gap in gaps:
        row: dict[str, Any] = {
            "gap_type": gap.gap_type,
            "keyword": gap.keyword,
            "cluster": gap.cluster,
            "cluster_label": gap.cluster_label,
            "score": gap.score,
            "frequency": gap.frequency,
            "n_articles": gap.n_articles,
            "citations": gap.citations,
            "rationale": gap.rationale,
            "article_titles": " | ".join(gap.article_titles),
            "article_dois": " | ".join(d for d in gap.article_dois if d),
        }
        row.update({f"metric_{k}": v for k, v in gap.metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["gap_type", "score"], ascending=[True, False], ignore_index=True
    )
