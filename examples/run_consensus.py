"""Consensus candidates pooled over k-means seeds and analysis-corpus sizes.

Reports, per corpus and mode, how many candidates survive at each support
level and how the pairwise overlap splits between the seed effect (run pairs
that share a corpus size) and the corpus-size effect (pairs that do not).
Uses the cached embeddings, so it makes no API calls.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, "embedresearchgaps/src")
sys.path.append(os.getcwd())

from embedresearchgaps import consensus_gaps, load_csv  # noqa: E402
from run_case import PROBLEMS, build_config, embed_corpus  # noqa: E402

SEEDS = (42, 7, 123, 2026, 31337)
SIZES = (30, 50, 100, 150)


def run(name: str) -> dict[str, object]:
    corpus = load_csv(f"corpora/{name}.csv")
    problem = PROBLEMS[name]
    vectors = embed_corpus(name, corpus, problem)
    config = build_config(problem, vectors, 50)
    out = Path("results") / name
    summaries: dict[str, object] = {}

    for mode in ("articles_first", "keywords_first"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            seeds_only = consensus_gaps(corpus, mode, config, seeds=SEEDS)
            both = consensus_gaps(corpus, mode, config, seeds=SEEDS, top_n=SIZES)
        both.save(out, prefix=f"{mode}_consensus_seeds_sizes")
        seeds_only.save(out, prefix=f"{mode}_consensus_seeds")
        summaries[mode] = {
            "seeds_only": seeds_only.summary(),
            "seeds_and_sizes": both.summary(),
        }
        s = both.stability
        print(
            f"  {mode}: {s['n_runs']} runs ({s['n_seeds']} seeds x "
            f"{s['n_corpus_sizes']} sizes)\n"
            f"    mean pairwise Jaccard {s['mean_pairwise_jaccard']:.3f} "
            f"(same size {s['mean_jaccard_same_size']:.3f}, "
            f"across sizes {s['mean_jaccard_across_sizes']:.3f})\n"
            f"    candidates pooled {len(both.gaps)}; "
            f"support >= 0.5: {int((both.gaps['support'] >= 0.5).sum())}; "
            f"unanimous: {int((both.gaps['support'] == 1.0).sum())}\n"
            f"    seeds only: {len(seeds_only.gaps)} pooled, "
            f"{int((seeds_only.gaps['support'] >= 0.5).sum())} at support >= 0.5, "
            f"mean J {seeds_only.stability['mean_pairwise_jaccard']:.3f}",
            flush=True,
        )
        top = both.gaps.nlargest(5, ["support", "mean_score"])[
            ["keyword", "gap_type", "support", "mean_score"]
        ]
        print("    top by support:", "; ".join(
            f"{r.keyword} ({r.support:.2f})" for r in top.itertuples()
        ), flush=True)

    (out / "consensus_summary.json").write_text(
        json.dumps(summaries, indent=2, default=str), encoding="utf-8"
    )
    return summaries


if __name__ == "__main__":
    for corpus_name in sys.argv[1:] or ["management", "finance"]:
        print(f"[{corpus_name}]", flush=True)
        run(corpus_name)
