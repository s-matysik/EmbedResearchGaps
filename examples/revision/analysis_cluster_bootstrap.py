"""Cluster-robust intervals for the candidate-versus-control contrast.

Keywords are nested within article clusters, so a bootstrap that resamples
keywords treats correlated observations as independent.  This script resamples
article clusters instead, which is the cluster-robust analogue requested in
review, and reports the resulting 95 % intervals beside the point estimate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("paper/results")
RNG = np.random.default_rng(7)


def keyword_level(annotations: pd.DataFrame) -> pd.DataFrame:
    return (
        annotations.assign(is_gap=annotations["label"].eq("GAP").astype(float))
        .groupby(["keyword", "keyword_set"], as_index=False)
        .agg(gap=("is_gap", "mean"), nov=("novelty", "mean"))
    )


def run(corpus: str, mode: str, n_boot: int = 4000) -> dict:
    frame = keyword_level(
        pd.read_csv(RESULTS / corpus / f"{mode}_validation_annotations_disjoint.csv")
    )
    keywords = pd.read_csv(RESULTS / corpus / f"{mode}_keywords.csv")
    column = "article_cluster" if "article_cluster" in keywords.columns else "cluster"
    home = keywords.groupby("keyword")[column].agg(lambda s: s.value_counts().idxmax())
    merged = (frame.merge(home.rename("clu"), left_on="keyword", right_index=True, how="left")
                   .dropna(subset=["clu"]))
    clusters = sorted(merged["clu"].unique())

    deltas = []
    for _ in range(n_boot):
        draw = pd.concat([merged[merged.clu == c]
                          for c in RNG.choice(clusters, size=len(clusters), replace=True)])
        selected = draw[draw.keyword_set == "SELECTED"]["gap"]
        control = draw[draw.keyword_set == "CONTROL"]["gap"]
        if len(selected) and len(control):
            deltas.append((selected.mean() - control.mean()) * 100)

    observed = (merged[merged.keyword_set == "SELECTED"]["gap"].mean()
                - merged[merged.keyword_set == "CONTROL"]["gap"].mean()) * 100
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {"n_clusters": len(clusters), "delta_pp": float(observed),
            "cluster_ci": [float(low), float(high)]}


if __name__ == "__main__":
    out = {f"{c}|{m}": run(c, m)
           for c in ("management", "finance", "gamification")
           for m in ("articles_first", "keywords_first")}
    Path("paper/results/revision/cluster_bootstrap.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    for key, rec in out.items():
        print(f"{key:32s} delta={rec['delta_pp']:+6.1f} pp  "
              f"cluster-robust CI [{rec['cluster_ci'][0]:+.0f}, {rec['cluster_ci'][1]:+.0f}]")
