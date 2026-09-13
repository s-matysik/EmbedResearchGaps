"""The two operating modes of EmbedResearchGaps.

``keywords_first``
    Cluster the author keywords of the analysis corpus directly and flag the
    semantic periphery of each keyword cluster (PACIS 2026 procedure).

``articles_first``
    Cluster the articles first, then build a keyword sub-space inside every
    article cluster and apply the three-type gap typology (AMCIS 2026
    procedure).

Both modes share corpus preparation: optional semantic ranking against a
natural-language problem description, truncation to the top ``N`` records,
corpus-level TF-IDF salience, document frequencies and the core-domain filter.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .clustering import (
    ClusterAssignment,
    centroid_similarity_matrix,
    fit_kmeans,
    rank_by_similarity,
)
from .config import RunConfig
from .corpus import Corpus
from .encoders import Encoder, get_encoder
from .gaps import (
    COMBINATION_SEPARATOR,
    Gap,
    deduplicate_gaps,
    detect_conceptual_combinations,
    detect_cross_cluster_concepts,
    detect_emerging_concepts,
    detect_peripheral_keywords,
    gaps_to_frame,
)
from .keywords import KeywordSpace, build_keyword_space, default_subcluster_count
from .results import PipelineResult
from .text import (
    TfidfSalience,
    compute_tfidf_salience,
    core_domain_terms,
    keyword_frequencies,
    normalize_keyword,
)

__all__ = [
    "LOW_SEPARATION_SILHOUETTE",
    "cluster_separation_warning",
    "keywords_first",
    "articles_first",
    "run",
    "rank_corpus",
    "default_cluster_label",
]

ClusterLabeler = Callable[[int, pd.DataFrame, Sequence[str]], str]


@dataclass
class _Prepared:
    """Corpus and corpus-level statistics shared by both modes."""

    corpus: Corpus
    frame: pd.DataFrame
    tfidf: TfidfSalience
    global_frequency: Counter
    core_terms: set[str]
    ranking: dict[str, object]
    spelling: dict[str, str]


def rank_corpus(
    corpus: Corpus,
    problem_description: str,
    encoder: Encoder,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank records by cosine distance to a natural-language problem statement.

    Returns the ordering (most similar first) and the distance of every
    record in original order, following equation (1) of the source papers.
    """
    texts = list(corpus.frame["combined_text"])
    matrix = encoder.encode([problem_description] + texts)
    return rank_by_similarity(matrix[0], matrix[1:])


def _resolve_frequency_threshold(config: RunConfig, n_articles: int) -> int | None:
    """Apply the proportional frequency rule when no explicit value is set."""
    if config.gaps.max_keyword_frequency is not None:
        return config.gaps.max_keyword_frequency
    return max(1, round(n_articles / 50))


def _prepare(corpus: Corpus, config: RunConfig) -> _Prepared:
    frame = corpus.frame.copy()
    ranking: dict[str, object] = {"ranked": False}

    if config.problem_description:
        encoder = get_encoder(config.article_encoder)
        order, distances = rank_corpus(corpus, config.problem_description, encoder)
        frame["query_distance"] = distances
        frame = frame.iloc[order].reset_index(drop=True)
        frame["rank"] = np.arange(1, len(frame) + 1)
        ranking = {
            "ranked": True,
            "encoder": encoder.describe(),
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances)),
            "cutoff_distance": float(frame["query_distance"].iloc[
                min(config.top_n_articles, len(frame)) - 1
            ]),
        }

    if config.top_n_articles and len(frame) > config.top_n_articles:
        frame = frame.head(config.top_n_articles).reset_index(drop=True)

    analysis = Corpus(
        frame=frame,
        source_path=corpus.source_path,
        column_map=dict(corpus.column_map),
    )
    keyword_lists = analysis.keyword_lists
    tfidf = compute_tfidf_salience(keyword_lists)
    frequencies = keyword_frequencies(keyword_lists)
    core = core_domain_terms(frequencies, len(frame), config.gaps.core_domain_share)
    return _Prepared(
        analysis, frame, tfidf, frequencies, core, ranking, analysis.spelling
    )


def _capitalise(text: str) -> str:
    """Upper-case the first character only, leaving acronyms untouched."""
    return text[:1].upper() + text[1:]


def _display_keyword(keyword: str, spelling: Mapping[str, str]) -> str:
    """Render a keyword (or a conceptual combination) in its source spelling."""
    parts = str(keyword).split(COMBINATION_SEPARATOR)
    return COMBINATION_SEPARATOR.join(
        _capitalise(spelling.get(part, part)) for part in parts
    )


def _add_display_column(frame: pd.DataFrame, spelling: Mapping[str, str]) -> pd.DataFrame:
    """Insert ``keyword_display`` next to ``keyword``."""
    if frame.empty:
        return frame
    frame = frame.copy()
    frame.insert(
        frame.columns.get_loc("keyword") + 1,
        "keyword_display",
        [_display_keyword(k, spelling) for k in frame["keyword"]],
    )
    return frame


def default_cluster_label(
    cluster_id: int,
    cluster_articles: pd.DataFrame,
    core_terms: Sequence[str] = (),
    n_terms: int = 2,
    spelling: Mapping[str, str] | None = None,
) -> str:
    """Label a cluster by its most frequent non-core author keywords.

    ``spelling`` maps normalised keywords back to the spelling the authors
    used, so that acronyms survive in the label.  Deterministic and offline;
    pass a ``cluster_labeler`` to the pipeline to label clusters with an LLM
    instead, as in the source papers.
    """
    core = {normalize_keyword(t) for t in core_terms}
    counter: Counter = Counter()
    for keywords in cluster_articles["keywords_normalized"]:
        for keyword in dict.fromkeys(keywords):
            if keyword and keyword not in core:
                counter[keyword] += 1
    top = [kw for kw, _ in counter.most_common(n_terms)]
    if not top:
        return f"Cluster {cluster_id}"
    spelling = spelling or {}
    label = " & ".join(spelling.get(kw, kw) for kw in top)
    return label[:1].upper() + label[1:]


def _encode_all_keywords(
    corpus: Corpus,
    config: RunConfig,
) -> tuple[dict[str, list[float]], Encoder]:
    """Embed every unique keyword once so clusters can share the vectors."""
    encoder = get_encoder(config.keyword_encoder)
    unique = corpus.unique_keywords()
    matrix = encoder.encode(unique)
    return {kw: matrix[i].tolist() for i, kw in enumerate(unique)}, encoder


def _keyword_table(
    space: KeywordSpace,
    tfidf: TfidfSalience,
    global_frequency: Mapping[str, int],
    core_terms: set[str],
    cluster_column: str = "cluster",
) -> pd.DataFrame:
    frame = space.to_frame().rename(columns={"subcluster": cluster_column})
    frame["tfidf"] = [tfidf.get(kw) for kw in frame["keyword"]]
    frame["global_frequency"] = [int(global_frequency.get(kw, 0)) for kw in frame["keyword"]]
    frame["is_core_domain"] = frame["keyword"].isin(core_terms)
    return frame


#: Silhouette below which a partition is reported as poorly separated.  The
#: value is the conventional boundary for "no substantial structure" in
#: Kaufman and Rousseeuw's reading of the coefficient; on author-keyword
#: corpora real partitions routinely fall below it, which is exactly the
#: situation a user must be told about rather than left to infer.
LOW_SEPARATION_SILHOUETTE = 0.10


def cluster_separation_warning(
    quality: Mapping[str, float],
    n_clusters: int,
    space: str,
) -> str | None:
    """A warning string when a partition has no substantial cluster structure.

    Returns ``None`` when the silhouette is at or above
    :data:`LOW_SEPARATION_SILHOUETTE`, and otherwise a message naming the
    measured value.  Callers put it in ``PipelineResult.diagnostics['warnings']``
    and it is also raised as a :class:`UserWarning`, because a candidate list
    read off a partition this weak is one draw from a broad distribution: see
    :func:`~embedresearchgaps.consensus.consensus_gaps`.
    """
    silhouette = quality.get("silhouette")
    if silhouette is None or silhouette >= LOW_SEPARATION_SILHOUETTE:
        return None
    return (
        f"LOW_CLUSTER_SEPARATION: silhouette of the {space} partition is "
        f"{silhouette:.3f} over {n_clusters} clusters, below the "
        f"{LOW_SEPARATION_SILHOUETTE:g} threshold for substantial structure. "
        "Candidates read off a single partition this weak are unstable across "
        "initialisations; pool across seeds with consensus_gaps() and report "
        "per-candidate support instead of a single-run ranking."
    )


def _emit_warnings(diagnostics: dict[str, Any], messages: Sequence[str | None]) -> None:
    """Record non-empty warnings in ``diagnostics`` and raise them once each."""
    collected = [message for message in messages if message]
    diagnostics["warnings"] = collected
    for message in collected:
        warnings.warn(message, UserWarning, stacklevel=3)


def keywords_first(
    corpus: Corpus,
    config: RunConfig | None = None,
    cluster_labeler: ClusterLabeler | None = None,
) -> PipelineResult:
    """Mode A: cluster author keywords directly, then flag the periphery.

    Steps
    -----
    1. Optionally rank records against ``config.problem_description`` and keep
       the top ``config.top_n_articles``.
    2. Embed every unique author keyword and cluster the keyword space with
       k-means, choosing k by elbow/silhouette unless fixed in the config.
    3. Flag keywords whose distance from their cluster centroid exceeds the
       cluster-specific percentile and whose frequency is low.
    4. Attach source articles and their citation counts to every candidate.

    Notes
    -----
    Candidates are ranked by centroid distance.  Citations are reported as a
    relevance signal and, unless ``config.use_citations`` is false, used as
    the tie-breaker; they are never used to exclude a candidate, because a
    recent paper has few citations by construction.
    """
    config = config or RunConfig()
    prepared = _prepare(corpus, config)
    gap_config = config.gaps
    threshold = _resolve_frequency_threshold(config, len(prepared.frame))
    if gap_config.max_keyword_frequency != threshold:
        gap_config = dataclasses.replace(gap_config, max_keyword_frequency=threshold)

    encoder = get_encoder(config.keyword_encoder)
    space = build_keyword_space(
        prepared.corpus.keyword_lists, encoder, config.clustering.distance_metric
    )
    if len(space) < 4:
        raise ValueError(
            f"keywords-first mode needs at least 4 unique keywords, got {len(space)}"
        )
    space.assignment = fit_kmeans(space.embeddings, config.clustering)

    gaps = detect_peripheral_keywords(
        space, prepared.frame, prepared.tfidf, prepared.core_terms, gap_config
    )
    if config.use_citations:
        gaps.sort(
            key=lambda gap: (gap.score, gap.citations if gap.citations is not None else -1.0),
            reverse=True,
        )

    labels = space.subcluster_labels
    cluster_labels: dict[int, str] = {}
    for cluster in sorted({int(c) for c in labels}):
        members = [space.keywords[i] for i in np.flatnonzero(labels == cluster)]
        member_articles = prepared.frame[
            prepared.frame["keywords_normalized"].apply(
                lambda kws: any(kw in members for kw in kws)
            )
        ]
        cluster_labels[cluster] = (
            cluster_labeler(cluster, member_articles, members)
            if cluster_labeler
            else _capitalise(
                " & ".join(
                    prepared.spelling.get(kw, kw)
                    for kw, _ in Counter(
                        kw for kw in members if kw not in prepared.core_terms
                    ).most_common(2)
                )
            ) or f"Cluster {cluster}"
        )
    for gap in gaps:
        gap.cluster_label = cluster_labels.get(gap.cluster)

    keywords_table = _keyword_table(
        space, prepared.tfidf, prepared.global_frequency, prepared.core_terms
    )
    keywords_table["cluster_label"] = keywords_table["cluster"].map(cluster_labels)

    assignment = space.assignment
    diagnostics = {
        "n_clusters": int(assignment.n_clusters),
        "cluster_sizes": assignment.sizes(),
        "cluster_quality": assignment.quality,
        "k_selection": assignment.selection.to_dict() if assignment.selection else None,
        "distance_metric": config.clustering.distance_metric,
        "frequency_threshold": threshold,
        "periphery_percentile": gap_config.periphery_percentile,
        "peripheral_share": float(len(gaps) / len(space)) if len(space) else 0.0,
        "n_core_domain_terms": len(prepared.core_terms),
        "core_domain_terms": sorted(prepared.core_terms),
        "tfidf_match_statistics": prepared.tfidf.match_statistics(),
        "corpus": prepared.corpus.summary(),
        "ranking": prepared.ranking,
        "keyword_encoder": encoder.describe(),
    }
    _emit_warnings(diagnostics, [
        cluster_separation_warning(
            space.assignment.quality, space.assignment.n_clusters, "keyword"
        ),
    ])

    return PipelineResult(
        mode="keywords_first",
        config=config,
        articles=prepared.frame,
        keywords=keywords_table,
        gaps=_add_display_column(gaps_to_frame(gaps), prepared.spelling),
        gap_objects=gaps,
        cluster_labels=cluster_labels,
        centroid_similarity=centroid_similarity_matrix(assignment.centroids),
        diagnostics=diagnostics,
        embeddings={"keywords": space.embeddings},
        keyword_index=list(space.keywords),
    )


def articles_first(
    corpus: Corpus,
    config: RunConfig | None = None,
    cluster_labeler: ClusterLabeler | None = None,
) -> PipelineResult:
    """Mode B: cluster articles, then mine keyword sub-spaces for three gap types.

    Steps
    -----
    1. Optionally rank records against ``config.problem_description`` and keep
       the top ``config.top_n_articles``.
    2. Embed title+abstract of each record and cluster the articles.
    3. Inside every article cluster, build a keyword sub-space, sub-cluster it
       and run the three detectors (emerging, combination, cross-cluster).
    4. Optionally de-duplicate keywords that surface in several clusters.
    """
    config = config or RunConfig()
    prepared = _prepare(corpus, config)
    frame = prepared.frame

    article_encoder = get_encoder(config.article_encoder)
    article_matrix = article_encoder.encode(list(frame["combined_text"]))
    assignment: ClusterAssignment = fit_kmeans(article_matrix, config.clustering)
    frame = frame.copy()
    frame["cluster"] = assignment.labels
    frame["distance_from_centroid"] = assignment.distances

    keyword_cache, keyword_encoder = _encode_all_keywords(prepared.corpus, config)

    all_gaps: list[Gap] = []
    keyword_tables: list[pd.DataFrame] = []
    cluster_labels: dict[int, str] = {}
    skipped: dict[int, str] = {}

    for cluster in range(assignment.n_clusters):
        cluster_articles = frame[frame["cluster"] == cluster].reset_index(drop=True)
        cluster_labels[cluster] = (
            cluster_labeler(cluster, cluster_articles, [])
            if cluster_labeler
            else default_cluster_label(
                cluster, cluster_articles, sorted(prepared.core_terms),
                spelling=prepared.spelling,
            )
        )
        if cluster_articles.empty:
            skipped[cluster] = "no articles"
            continue

        keyword_lists = list(cluster_articles["keywords_normalized"])
        if sum(len(k) for k in keyword_lists) < 3:
            skipped[cluster] = "fewer than 3 author keywords"
            continue

        space = build_keyword_space(
            keyword_lists,
            keyword_encoder,
            config.clustering.distance_metric,
            cache=keyword_cache,
        )
        space.fit_subclusters(
            config.clustering,
            config.clustering.n_keyword_subclusters or default_subcluster_count(len(space)),
        )

        cluster_gaps: list[Gap] = []
        cluster_gaps += detect_emerging_concepts(
            space, cluster_articles, cluster, prepared.tfidf,
            prepared.global_frequency, prepared.core_terms, config.gaps,
        )
        cluster_gaps += detect_conceptual_combinations(
            space, cluster_articles, cluster, prepared.tfidf, config.gaps
        )
        cluster_gaps += detect_cross_cluster_concepts(
            space, frame, cluster, prepared.tfidf,
            prepared.global_frequency, prepared.core_terms, config.gaps,
        )
        for gap in cluster_gaps:
            gap.cluster_label = cluster_labels[cluster]
        all_gaps += cluster_gaps

        table = _keyword_table(
            space, prepared.tfidf, prepared.global_frequency,
            prepared.core_terms, cluster_column="subcluster",
        )
        table.insert(0, "article_cluster", cluster)
        table["cluster_label"] = cluster_labels[cluster]
        keyword_tables.append(table)

    n_before = len(all_gaps)
    if config.gaps.deduplicate_cross_cluster:
        all_gaps = deduplicate_gaps(all_gaps)

    keywords_table = (
        pd.concat(keyword_tables, ignore_index=True)
        if keyword_tables
        else pd.DataFrame(columns=["article_cluster", "keyword", "frequency"])
    )

    diagnostics = {
        "n_clusters": int(assignment.n_clusters),
        "cluster_sizes": assignment.sizes(),
        "cluster_quality": assignment.quality,
        "k_selection": assignment.selection.to_dict() if assignment.selection else None,
        "distance_metric": config.clustering.distance_metric,
        "gaps_before_deduplication": n_before,
        "gaps_after_deduplication": len(all_gaps),
        "skipped_clusters": skipped,
        "n_core_domain_terms": len(prepared.core_terms),
        "core_domain_terms": sorted(prepared.core_terms),
        "tfidf_match_statistics": prepared.tfidf.match_statistics(),
        "corpus": prepared.corpus.summary(),
        "ranking": prepared.ranking,
        "article_encoder": article_encoder.describe(),
        "keyword_encoder": keyword_encoder.describe(),
        "n_unique_keywords_encoded": len(keyword_cache),
    }
    _emit_warnings(diagnostics, [
        cluster_separation_warning(assignment.quality, assignment.n_clusters, "article"),
    ])

    return PipelineResult(
        mode="articles_first",
        config=config,
        articles=frame,
        keywords=keywords_table,
        gaps=_add_display_column(gaps_to_frame(all_gaps), prepared.spelling),
        gap_objects=all_gaps,
        cluster_labels=cluster_labels,
        centroid_similarity=centroid_similarity_matrix(assignment.centroids),
        diagnostics=diagnostics,
        embeddings={
            "articles": article_matrix,
            "keywords": np.asarray([keyword_cache[kw] for kw in keyword_cache]),
        },
        keyword_index=list(keyword_cache),
    )


def run(
    corpus: Corpus,
    mode: str = "articles_first",
    config: RunConfig | None = None,
    cluster_labeler: ClusterLabeler | None = None,
) -> PipelineResult:
    """Dispatch to :func:`keywords_first` or :func:`articles_first` by name."""
    if mode == "keywords_first":
        return keywords_first(corpus, config, cluster_labeler)
    if mode == "articles_first":
        return articles_first(corpus, config, cluster_labeler)
    raise ValueError(
        f"unknown mode {mode!r}; expected 'keywords_first' or 'articles_first'"
    )
