"""Configuration objects for EmbedResearchGaps.

All tunable parameters live here so that a run is fully described by a single
serialisable object.  Defaults reproduce the settings reported in the two
conference papers that established the method:

* Frankowski, Wisniewska & Matysik (2026a), PACIS -- keywords-first mode.
* Frankowski, Wisniewska & Matysik (2026b), AMCIS -- articles-first mode.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

DistanceMetric = Literal["cosine", "euclidean"]
KSelection = Literal["auto", "elbow", "silhouette", "fixed"]

#: Multi-word methodological phrases filtered out of gap candidates.
DEFAULT_METHODOLOGICAL_PATTERNS: tuple[str, ...] = (
    "equation model", "factor analysis", "regression", "anova", "manova", "ancova",
    "mediation", "moderation", "path analysis", "multilevel model", "hierarchical model",
    "survey method", "questionnaire", "experiment", "quasi-experiment",
    "case study", "interview", "focus group", "content analysis", "thematic analysis",
    "grounded theory", "literature review", "meta-analysis", "bibliometric",
    "longitudinal", "cross-sectional", "panel data", "cronbach", "composite reliability",
    "convergent validity", "discriminant validity", "measurement model",
    "sample size", "sampling method", "likert scale", "semantic differential",
    "conceptual framework", "research agenda", "future research", "systematic review",
    "qualitative", "quantitative", "mixed method", "configurational",
    "fuzzy set", "fuzzy-set", "necessary condition", "feature selection",
    "structural equation", "multi-group", "robustness check", "instrumental variable",
    "difference-in-differences", "propensity score", "text mining", "topic model",
)

#: Single-token methodological tool names filtered out of gap candidates.
DEFAULT_METHODOLOGICAL_TOOLS: frozenset[str] = frozenset({
    "sem", "pls", "cfa", "efa", "hlm", "tam", "spss", "amos", "lisrel",
    "mplus", "smartpls", "stata", "nvivo",
})


@dataclass(frozen=True)
class EmergingWeights:
    """Weights of equation (7) -- Type 1, Emerging Concepts.

    ``S_em = distance * d + centrality * c + tfidf * t + rarity * r`` where
    every component is min-max normalised within the candidate set.
    """

    distance: float = 0.25
    centrality: float = 0.15
    tfidf: float = 0.30
    rarity: float = 0.30

    def __post_init__(self) -> None:
        total = self.distance + self.centrality + self.tfidf + self.rarity
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"EmergingWeights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class CombinationWeights:
    """Weights of equation (9) -- Type 2, Conceptual Combinations."""

    similarity: float = 0.40
    frequency: float = 0.30
    tfidf: float = 0.30

    def __post_init__(self) -> None:
        total = self.similarity + self.frequency + self.tfidf
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"CombinationWeights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class CrossClusterWeights:
    """Weights of equation (10) -- Type 3, Cross-Cluster Concepts."""

    frequency: float = 0.25
    specificity: float = 0.35
    tfidf: float = 0.40

    def __post_init__(self) -> None:
        total = self.frequency + self.specificity + self.tfidf
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"CrossClusterWeights must sum to 1.0, got {total}")


@dataclass
class GapConfig:
    """Thresholds shared by both pipeline modes."""

    #: Percentile of the within-cluster distance distribution above which a
    #: keyword counts as semantically peripheral (keywords-first mode).
    periphery_percentile: float = 95.0
    #: Maximum number of gaps reported per gap type per cluster.
    max_gaps_per_cluster: int = 3
    #: Minimum mean TF-IDF salience a candidate must reach.
    min_tfidf: float = 0.005
    #: Candidate keywords must occur at most this many times.  ``None``
    #: activates the proportional rule of the PACIS paper:
    #: ``max(1, round(n_articles / 50))``.
    max_keyword_frequency: int | None = 1
    #: Restrict gap candidates to keywords of exactly this many tokens.
    #: ``None`` disables the filter; ``2`` reproduces the AMCIS setting.
    gap_token_count: int | None = 2
    #: A keyword appearing in more than this share of the corpus is treated as
    #: a core-domain term and never reported as a gap.
    core_domain_share: float = 0.50
    #: Upper bound on the corpus presence ratio for cross-cluster candidates.
    max_presence_ratio: float = 0.10
    #: Cross-cluster candidates may occur at most this often in other clusters.
    max_frequency_other_clusters: int = 3
    #: Admissible cosine-similarity window for conceptual combinations.
    combination_similarity_range: tuple[float, float] = (0.25, 0.65)
    #: Similarity treated as maximally "combinable" inside that window.
    combination_similarity_optimum: float = 0.45
    #: Drop duplicated keywords that surface in several clusters, keeping the
    #: highest-scoring occurrence.
    deduplicate_cross_cluster: bool = True
    #: Remove methodological terms from candidates.
    filter_methodological: bool = True
    #: Extra domain-specific stop keywords.
    extra_stop_keywords: tuple[str, ...] = ()
    methodological_patterns: tuple[str, ...] = DEFAULT_METHODOLOGICAL_PATTERNS
    methodological_tools: frozenset[str] = DEFAULT_METHODOLOGICAL_TOOLS
    emerging_weights: EmergingWeights = field(default_factory=EmergingWeights)
    combination_weights: CombinationWeights = field(default_factory=CombinationWeights)
    cross_cluster_weights: CrossClusterWeights = field(default_factory=CrossClusterWeights)

    def __post_init__(self) -> None:
        if not 0.0 < self.periphery_percentile < 100.0:
            raise ValueError("periphery_percentile must lie in (0, 100)")
        if self.max_gaps_per_cluster < 1:
            raise ValueError("max_gaps_per_cluster must be >= 1")
        lo, hi = self.combination_similarity_range
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError("combination_similarity_range must satisfy 0 <= lo < hi <= 1")
        if self.gap_token_count is not None and self.gap_token_count < 1:
            raise ValueError("gap_token_count must be >= 1 or None")


@dataclass
class ClusteringConfig:
    """Clustering of articles and of keywords."""

    #: Fixed number of clusters; ``None`` triggers automatic selection.
    n_clusters: int | None = None
    k_selection: KSelection = "auto"
    k_min: int = 2
    k_max: int = 10
    #: With ``k_selection='auto'`` the elbow is preferred whenever the best
    #: silhouette stays below this value, otherwise the silhouette wins.
    silhouette_floor: float = 0.25
    n_init: int = 10
    random_state: int = 42
    distance_metric: DistanceMetric = "cosine"
    #: Number of keyword sub-clusters inside an article cluster.  ``None``
    #: uses ``min(5, max(2, int(sqrt(n_keywords))))`` as in the source code.
    n_keyword_subclusters: int | None = None

    def __post_init__(self) -> None:
        if self.k_min < 2:
            raise ValueError("k_min must be >= 2")
        if self.k_max < self.k_min:
            raise ValueError("k_max must be >= k_min")
        if self.n_clusters is not None and self.n_clusters < 2:
            raise ValueError("n_clusters must be >= 2 or None")


@dataclass
class EncoderConfig:
    """Which embedding model to use and how to call it."""

    #: Registered encoder name, e.g. ``sbert``, ``openai``, ``nomic``,
    #: ``jina``, ``cohere``, ``tfidf-svd`` or ``precomputed``.
    name: str = "tfidf-svd"
    model: str | None = None
    #: Environment variable holding the API key; never store keys in configs.
    api_key_env: str | None = None
    batch_size: int = 96
    normalize: bool = True
    #: Output dimensionality for encoders that support it.
    dimensions: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _describe_extra(value: Any) -> Any:
    """Compact, JSON-safe description of one ``EncoderConfig.extra`` entry.

    Scalars and short strings pass through unchanged; mappings, arrays and
    long sequences are summarised, because they can hold whole embedding
    matrices that have no place in a saved configuration file.
    """
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, dict):
        return f"<mapping of {len(value)} entries>"
    if hasattr(value, "shape"):
        return f"<array of shape {tuple(value.shape)}>"
    try:
        length = len(value)
    except TypeError:
        return f"<{type(value).__name__}>"
    if length > 16:
        return f"<sequence of {length} items>"
    return [_describe_extra(item) for item in value]


@dataclass
class RunConfig:
    """Top-level configuration of a pipeline run."""

    #: Number of highest-ranked articles forming the analysis corpus.
    top_n_articles: int = 50
    #: Natural-language description of the research problem.  When given and
    #: the corpus carries text, articles are ranked by cosine distance to it.
    problem_description: str | None = None
    article_encoder: EncoderConfig = field(default_factory=EncoderConfig)
    keyword_encoder: EncoderConfig = field(default_factory=EncoderConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    gaps: GapConfig = field(default_factory=GapConfig)
    random_state: int = 42
    #: Keep the citation column as a ranking signal (keywords-first mode).
    use_citations: bool = True

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready view of the configuration.

        Bulk payloads carried in ``EncoderConfig.extra`` -- a mapping of
        precomputed vectors, for instance -- are replaced by a one-line
        descriptor, so that a saved configuration stays readable and small
        while still recording what was supplied.
        """

        def _default(obj: Any) -> Any:
            if isinstance(obj, (frozenset, set)):
                return sorted(obj)
            if isinstance(obj, tuple):
                return list(obj)
            raise TypeError(type(obj))

        payload = asdict(self)
        for key in ("article_encoder", "keyword_encoder"):
            extra = payload.get(key, {}).get("extra")
            if isinstance(extra, dict):
                payload[key]["extra"] = {
                    name: _describe_extra(value) for name, value in extra.items()
                }
        return json.loads(json.dumps(payload, default=_default))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def replace(self, **changes: Any) -> "RunConfig":
        return dataclasses.replace(self, **changes)
