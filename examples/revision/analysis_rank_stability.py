"""Rank-aware stability of the candidate list (reviewer 2, point 10).

Jaccard overlap treats a shortlist as a set, but the manuscript proposes it as
a reading order, so the reviewer is right that set overlap is the wrong
instrument.  This script re-runs both modes across five k-means seeds and
reports, in addition to Jaccard: rank-biased overlap (p = 0.9, top-weighted),
top-10 overlap, and Kendall's tau on the items two runs share.

The offline tfidf-svd backend is used so the measurement needs no credentials
and no hosted endpoint, and can be rerun by anyone from the deposited corpora.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "repo/current/src")

from embedresearchgaps import (  # noqa: E402
    ClusteringConfig, EncoderConfig, GapConfig, RunConfig,
    articles_first, keywords_first, load_csv,
)

PROBLEMS = {
    "management": ("How do dynamic capabilities enable the digital transformation of "
                   "established firms, and which organisational mechanisms mediate the "
                   "relationship between the adoption of digital technologies and firm "
                   "performance?"),
    "finance": ("How does fintech credit affect financial inclusion and credit risk in "
                "banking systems, and which mechanisms link digital lending "
                "technologies to borrower and bank outcomes?"),
    "gamification": ("How does the use of gamification in marketing influence consumer "
                     "engagement and brand loyalty?"),
}
SEEDS = (42, 7, 123, 2024, 99)


def rbo(left: list[str], right: list[str], p: float = 0.9) -> float:
    """Rank-biased overlap, the top-weighted similarity of two rankings."""
    if not left or not right:
        return 0.0
    depth = max(len(left), len(right))
    seen_l: set[str] = set()
    seen_r: set[str] = set()
    total = 0.0
    for d in range(1, depth + 1):
        if d <= len(left):
            seen_l.add(left[d - 1])
        if d <= len(right):
            seen_r.add(right[d - 1])
        total += (len(seen_l & seen_r) / d) * (p ** (d - 1))
    return (1 - p) * total


def kendall_on_shared(left: list[str], right: list[str]) -> float:
    shared = set(left) & set(right)
    if len(shared) < 3:
        return float("nan")
    a = [left.index(k) for k in shared]
    b = [right.index(k) for k in shared]
    return float(stats.kendalltau(a, b).statistic)


def ranked(result) -> list[str]:
    frame = result.gaps
    if frame.empty:
        return []
    column = "score" if "score" in frame.columns else frame.columns[-1]
    ordered = frame.sort_values(column, ascending=False)
    keywords = ordered["keyword"] if "keyword" in ordered else ordered.index.to_series()
    return list(dict.fromkeys(keywords.tolist()))


def run(corpus: str, mode: str) -> dict:
    data = load_csv(f"repo/current/data/{corpus}.csv")
    encoder = EncoderConfig(name="tfidf-svd")
    lists: list[list[str]] = []
    for seed in SEEDS:
        config = RunConfig(
            top_n_articles=50, problem_description=PROBLEMS[corpus],
            article_encoder=encoder, keyword_encoder=encoder,
            clustering=ClusteringConfig(random_state=seed), gaps=GapConfig(),
            random_state=seed,
        )
        function = articles_first if mode == "articles_first" else keywords_first
        lists.append(ranked(function(data, config)))

    jac, rbos, taus, top10 = [], [], [], []
    for a, b in itertools.combinations(lists, 2):
        sa, sb = set(a), set(b)
        jac.append(len(sa & sb) / len(sa | sb) if sa | sb else 0.0)
        rbos.append(rbo(a, b))
        taus.append(kendall_on_shared(a, b))
        ta, tb = set(a[:10]), set(b[:10])
        top10.append(len(ta & tb) / 10 if ta and tb else 0.0)
    return {
        "mean_list_length": float(np.mean([len(x) for x in lists])),
        "jaccard_mean": float(np.mean(jac)),
        "rbo_mean": float(np.mean(rbos)), "rbo_min": float(np.min(rbos)),
        "top10_mean": float(np.mean(top10)),
        "kendall_mean": float(np.nanmean(taus)),
        "kendall_n_pairs": int(np.sum(~np.isnan(taus))),
    }


if __name__ == "__main__":
    out = {}
    for corpus in ("management", "finance", "gamification"):
        for mode in ("articles_first", "keywords_first"):
            out[f"{corpus}|{mode}"] = run(corpus, mode)
            print(f"{corpus}|{mode}", json.dumps(out[f"{corpus}|{mode}"]), flush=True)
    Path("rev/rank_stability.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    rows = [{"corpus": k.split("|")[0], "mode": k.split("|")[1],
             "len": round(v["mean_list_length"], 1), "jaccard": round(v["jaccard_mean"], 3),
             "RBO": round(v["rbo_mean"], 3), "top10": round(v["top10_mean"], 2),
             "kendall": round(v["kendall_mean"], 3)} for k, v in out.items()]
    print(pd.DataFrame(rows).to_string(index=False))
