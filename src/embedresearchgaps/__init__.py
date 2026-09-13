"""EmbedResearchGaps -- embedding-driven identification of research gaps.

Two operating modes over the same author-keyword semantic space:

>>> from embedresearchgaps import load_csv, RunConfig, keywords_first, articles_first
>>> corpus = load_csv("scopus.csv")                       # doctest: +SKIP
>>> config = RunConfig(top_n_articles=50)                 # doctest: +SKIP
>>> mode_a = keywords_first(corpus, config)               # doctest: +SKIP
>>> mode_b = articles_first(corpus, config)               # doctest: +SKIP

``keywords_first`` clusters the author keywords directly and flags the
semantic periphery of each keyword cluster.  ``articles_first`` clusters the
articles first, then mines a keyword sub-space inside every article cluster
for emerging concepts, unrealised conceptual combinations and cross-cluster
concepts.

The package is the gap-detection companion to EmbedSLR, which performs the
embedding-based screening step of a systematic literature review.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .clustering import (
    ClusterAssignment,
    KSelectionResult,
    centroid_similarity_matrix,
    fit_kmeans,
    rank_by_similarity,
    select_k,
)
from .config import (
    ClusteringConfig,
    CombinationWeights,
    CrossClusterWeights,
    EmergingWeights,
    EncoderConfig,
    GapConfig,
    RunConfig,
)
from .consensus import ConsensusResult, DEFAULT_SEEDS, consensus_gaps, jaccard
from .corpus import Corpus, from_dataframe, load_csv
from .encoders import Encoder, available_encoders, get_encoder, register_encoder
from .gaps import (
    Gap,
    GAP_TYPES,
    deduplicate_gaps,
    detect_conceptual_combinations,
    detect_cross_cluster_concepts,
    detect_emerging_concepts,
    detect_peripheral_keywords,
    gaps_to_frame,
    score_combination,
    score_cross_cluster,
    score_emerging,
)
from .keywords import KeywordSpace, build_keyword_space
from .pipelines import articles_first, keywords_first, run
from .results import PipelineResult
from .text import (
    compute_tfidf_salience,
    core_domain_terms,
    is_methodological_term,
    keyword_frequencies,
    normalize_keyword,
)

__all__ = [
    "__version__",
    # corpus
    "Corpus", "load_csv", "from_dataframe",
    # configuration
    "RunConfig", "GapConfig", "ClusteringConfig", "EncoderConfig",
    "EmergingWeights", "CombinationWeights", "CrossClusterWeights",
    # encoders
    "Encoder", "get_encoder", "register_encoder", "available_encoders",
    # keyword space and clustering
    "KeywordSpace", "build_keyword_space", "ClusterAssignment", "KSelectionResult",
    "select_k", "fit_kmeans", "rank_by_similarity", "centroid_similarity_matrix",
    # gaps
    "Gap", "GAP_TYPES", "gaps_to_frame", "deduplicate_gaps",
    "score_emerging", "score_combination", "score_cross_cluster",
    "detect_emerging_concepts", "detect_conceptual_combinations",
    "detect_cross_cluster_concepts", "detect_peripheral_keywords",
    # text utilities
    "normalize_keyword", "is_methodological_term", "keyword_frequencies",
    "core_domain_terms", "compute_tfidf_salience",
    # pipelines and results
    "keywords_first", "articles_first", "run", "PipelineResult",
    # stability-aware consensus
    "consensus_gaps", "ConsensusResult", "DEFAULT_SEEDS", "jaccard",
]
