"""Run both EmbedResearchGaps modes on a Scopus corpus.

Embeddings are computed once per corpus with ``text-embedding-3-large`` and
cached to ``cache/<name>_embeddings.npz``; both modes then read the cached
vectors through the ``precomputed`` backend, so a re-run costs nothing and is
bit-for-bit reproducible.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "embedresearchgaps/src")

from embedresearchgaps import (  # noqa: E402
    ClusteringConfig,
    EncoderConfig,
    GapConfig,
    RunConfig,
    articles_first,
    keywords_first,
    load_csv,
)
from embedresearchgaps.encoders import get_encoder  # noqa: E402
from embedresearchgaps.viz import generate_figures  # noqa: E402

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1024

PROBLEMS: dict[str, str] = {
    "management": (
        "How do dynamic capabilities enable the digital transformation of "
        "established firms, and which organisational mechanisms mediate the "
        "relationship between the adoption of digital technologies and firm "
        "performance?"
    ),
    "finance": (
        "How does fintech credit affect financial inclusion and credit risk in "
        "banking systems, and which mechanisms link digital lending "
        "technologies to borrower and bank outcomes?"
    ),
    "gamification": (
        "How does the use of gamification in marketing influence consumer "
        "engagement and brand loyalty?"
    ),
}

DOMAINS: dict[str, str] = {
    "management": "strategic management and organisation studies",
    "finance": "economics, finance and banking",
    "gamification": "marketing",
}


def embed_corpus(name: str, corpus, problem: str) -> dict[str, list[float]]:
    """Embed every article text, every unique keyword and the problem statement."""
    cache_path = Path("cache") / f"{name}_embeddings.npz"
    texts = (
        [problem]
        + list(corpus.frame["combined_text"])
        + corpus.unique_keywords()
    )
    texts = list(dict.fromkeys(texts))

    if cache_path.exists():
        stored = np.load(cache_path, allow_pickle=True)
        cached = {str(k): v for k, v in zip(stored["texts"], stored["vectors"])}
        missing = [text for text in texts if text not in cached]
        if not missing:
            print(f"  cache hit: {len(cached)} vectors", flush=True)
            return {text: cached[text] for text in texts}
    else:
        cached, missing = {}, texts

    print(f"  embedding {len(missing)} texts with {EMBEDDING_MODEL}", flush=True)
    encoder = get_encoder(
        EncoderConfig(
            name="openai",
            model=EMBEDDING_MODEL,
            api_key_env="OPENAI",
            dimensions=EMBEDDING_DIMENSIONS,
            batch_size=96,
        )
    )
    matrix = encoder.encode(missing)
    cached.update({text: matrix[i] for i, text in enumerate(missing)})

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(cached)
    np.savez_compressed(
        cache_path,
        texts=np.array(keys, dtype=object),
        vectors=np.asarray([cached[k] for k in keys], dtype=np.float32),
    )
    return {text: cached[text] for text in texts}


def build_config(problem: str, vectors: dict[str, list[float]], top_n: int) -> RunConfig:
    encoder = EncoderConfig(
        name="precomputed", normalize=True, extra={"vectors": vectors}
    )
    return RunConfig(
        top_n_articles=top_n,
        problem_description=problem,
        article_encoder=encoder,
        keyword_encoder=encoder,
        clustering=ClusteringConfig(k_min=2, k_max=10, silhouette_floor=0.25, random_state=42),
        gaps=GapConfig(max_gaps_per_cluster=3, periphery_percentile=95.0),
        random_state=42,
    )


def run_corpus(name: str, top_n: int = 50) -> dict[str, object]:
    print(f"[{name}]", flush=True)
    corpus = load_csv(f"corpora/{name}.csv")
    print(f"  corpus: {json.dumps(corpus.summary(), default=str)}", flush=True)
    problem = PROBLEMS[name]
    vectors = embed_corpus(name, corpus, problem)
    config = build_config(problem, vectors, top_n)

    out = Path("results") / name
    summaries: dict[str, object] = {"corpus": corpus.summary(), "domain": DOMAINS[name]}
    for mode, runner in (("articles_first", articles_first), ("keywords_first", keywords_first)):
        result = runner(corpus, config)
        result.save(out, prefix=mode)
        generate_figures(result, out, prefix=f"{mode}_fig")
        summaries[mode] = result.summary()
        print(
            f"  {mode}: k={result.n_clusters} gaps={result.n_gaps} "
            f"{json.dumps(result.gap_type_counts())}",
            flush=True,
        )
        if mode == "keywords_first":
            print(
                f"    peripheral share={result.diagnostics['peripheral_share']:.4f} "
                f"freq threshold={result.diagnostics['frequency_threshold']}",
                flush=True,
            )

    out.mkdir(parents=True, exist_ok=True)
    (out / "case_summary.json").write_text(
        json.dumps(summaries, indent=2, default=str), encoding="utf-8"
    )
    return summaries


if __name__ == "__main__":
    for corpus_name in sys.argv[1:] or ["management", "finance", "gamification"]:
        run_corpus(corpus_name)
