"""Sensitivity of both modes to the analysis-corpus size N and to the seed.

Reports, per corpus and per N: the selected number of clusters, the number of
candidates per gap type, and the Jaccard overlap of the candidate keyword set
with the N = 50 reference run.  Seed stability is measured as the Jaccard
overlap across five k-means seeds at fixed N.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, "embedresearchgaps/src")
sys.path.append(os.getcwd())

from embedresearchgaps import ClusteringConfig, articles_first, keywords_first, load_csv  # noqa: E402
from run_case import PROBLEMS, build_config, embed_corpus  # noqa: E402

N_VALUES = (30, 50, 100, 150)
SEEDS = (42, 7, 123, 2026, 31337)
CORPORA = ("management", "finance", "gamification")


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else float("nan")


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    size_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []

    for name in CORPORA:
        corpus = load_csv(f"corpora/{name}.csv")
        problem = PROBLEMS[name]
        vectors = embed_corpus(name, corpus, problem)

        for mode, runner in (("articles_first", articles_first), ("keywords_first", keywords_first)):
            reference: set[str] | None = None
            for n in N_VALUES:
                result = runner(corpus, build_config(problem, vectors, n))
                keywords = set(result.gaps["keyword"].astype(str)) if not result.gaps.empty else set()
                if n == 50:
                    reference = keywords
                size_rows.append(
                    {
                        "corpus": name,
                        "mode": mode,
                        "n_articles": n,
                        "k": result.n_clusters,
                        "n_keywords_in_space": int(result.keywords["keyword"].nunique()),
                        "n_gaps": result.n_gaps,
                        **{f"n_{k.lower().replace(' ', '_').replace('-', '_')}": v
                           for k, v in result.gap_type_counts().items()},
                        "silhouette": result.diagnostics.get("cluster_quality", {}).get("silhouette"),
                        "candidates": " | ".join(sorted(keywords)),
                    }
                )
            for row in size_rows:
                if row["corpus"] == name and row["mode"] == mode and reference is not None:
                    row["jaccard_vs_n50"] = jaccard(
                        set(str(row["candidates"]).split(" | ")) - {""}, reference
                    )

            per_seed: dict[int, set[str]] = {}
            for seed in SEEDS:
                config = build_config(problem, vectors, 50)
                config = dataclasses.replace(
                    config,
                    clustering=ClusteringConfig(
                        k_min=2, k_max=10, silhouette_floor=0.25, random_state=seed
                    ),
                    random_state=seed,
                )
                result = runner(corpus, config)
                per_seed[seed] = (
                    set(result.gaps["keyword"].astype(str)) if not result.gaps.empty else set()
                )
            overlaps = [jaccard(per_seed[a], per_seed[b]) for a, b in combinations(SEEDS, 2)]
            seed_rows.append(
                {
                    "corpus": name,
                    "mode": mode,
                    "n_seeds": len(SEEDS),
                    "mean_pairwise_jaccard": sum(overlaps) / len(overlaps),
                    "min_pairwise_jaccard": min(overlaps),
                    "max_pairwise_jaccard": max(overlaps),
                    "mean_n_gaps": sum(len(v) for v in per_seed.values()) / len(per_seed),
                }
            )
            print(f"  {name}/{mode}: seed stability "
                  f"{seed_rows[-1]['mean_pairwise_jaccard']:.3f}", flush=True)

    return pd.DataFrame(size_rows), pd.DataFrame(seed_rows)


if __name__ == "__main__":
    out = Path("results/sensitivity")
    out.mkdir(parents=True, exist_ok=True)
    sizes, seeds = run()
    sizes.to_csv(out / "corpus_size_sensitivity.csv", index=False)
    seeds.to_csv(out / "seed_stability.csv", index=False)
    print(sizes.drop(columns=["candidates"]).to_string(index=False))
    print(seeds.to_string(index=False))
