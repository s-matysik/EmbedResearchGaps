"""How much does the composite score depend on its weights? (reviewer 2, point 6)

The emerging-concept score is a weighted sum of four min-max normalised
components, with weights inherited from the source procedure and never tuned.
The reviewer asks what they buy. Since the weights affect the ORDER of the
candidate list and not its membership -- the filters decide membership -- the
question is answered by rescoring one fixed candidate set under alternative
weightings and measuring how far the ranking moves.

Four alternatives are compared with the published weighting
(0.25 distance, 0.15 centrality, 0.30 TF-IDF, 0.30 rarity): equal weights, and
three leave-one-out variants with the remaining weights renormalised.  Reported
per corpus: Kendall's tau-b and rank-biased overlap against the published
order, and the overlap of the top five.

The run is offline: the scored candidate tables deposited with the release
already carry every component, so nothing is re-embedded and no credentials are
needed.

Output: rev/weight_ablation.json, rev/weight_ablation.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "repo/current/src")

RESULTS = Path("repo/current/paper/results")
PUBLISHED = {"distance": 0.25, "centrality": 0.15, "tfidf": 0.30, "rarity": 0.30}
COMPONENTS = list(PUBLISHED)


def variants() -> dict[str, dict[str, float]]:
    out = {"published": dict(PUBLISHED),
           "equal": {c: 0.25 for c in COMPONENTS}}
    for dropped in COMPONENTS:
        kept = {c: w for c, w in PUBLISHED.items() if c != dropped}
        total = sum(kept.values())
        out[f"without_{dropped}"] = {c: w / total for c, w in kept.items()}
    return out


def minmax(values: pd.Series) -> pd.Series:
    span = values.max() - values.min()
    if span == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.min()) / span


def rbo(left: list[str], right: list[str], p: float = 0.9) -> float:
    if not left or not right:
        return 0.0
    seen_l: set[str] = set()
    seen_r: set[str] = set()
    total = 0.0
    for d in range(1, max(len(left), len(right)) + 1):
        if d <= len(left):
            seen_l.add(left[d - 1])
        if d <= len(right):
            seen_r.add(right[d - 1])
        total += (len(seen_l & seen_r) / d) * (p ** (d - 1))
    return (1 - p) * total


def components_for(frame: pd.DataFrame) -> pd.DataFrame | None:
    """The four normalised components of the emerging-concept score."""
    emerging = frame[frame.gap_type == "Emerging Concept"].copy()
    needed = {"metric_distance_zscore", "metric_centrality", "metric_tfidf",
              "metric_global_frequency"}
    if emerging.empty or not needed <= set(emerging.columns):
        return None
    # Mirrors gaps.py exactly: the distance term is the z-scored distance put
    # through min-max, and rarity is one minus the min-max of the corpus-wide
    # document frequency.
    out = pd.DataFrame(index=emerging.index)
    out["keyword"] = emerging["keyword"]
    out["distance"] = minmax(emerging["metric_distance_zscore"].astype(float))
    out["centrality"] = minmax(emerging["metric_centrality"].astype(float))
    out["tfidf"] = minmax(emerging["metric_tfidf"].astype(float))
    out["rarity"] = 1.0 - minmax(emerging["metric_global_frequency"].astype(float))
    return out


def main() -> None:
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for corpus in ("management", "finance", "gamification"):
        frame = pd.read_csv(RESULTS / corpus / "articles_first_gaps.csv")
        comp = components_for(frame)
        if comp is None:
            print(f"{corpus}: components unavailable in the deposited table", flush=True)
            continue
        orders: dict[str, list[str]] = {}
        scores: dict[str, pd.Series] = {}
        for name, weights in variants().items():
            score = sum(comp[c] * w for c, w in weights.items())
            scores[name] = score
            orders[name] = list(comp.assign(s=score).sort_values(
                "s", ascending=False)["keyword"])
        base = orders["published"]
        for name in orders:
            if name == "published":
                continue
            shared = [k for k in base if k in orders[name]]
            tau = stats.kendalltau(
                [base.index(k) for k in shared],
                [orders[name].index(k) for k in shared],
            ).statistic if len(shared) >= 3 else float("nan")
            row = {
                "corpus": corpus, "variant": name,
                "n_candidates": len(base),
                "kendall_tau": round(float(tau), 3),
                "rbo": round(rbo(base, orders[name]), 3),
                "top5_overlap": round(
                    len(set(base[:5]) & set(orders[name][:5])) / 5, 2),
                "top1_same": base[0] == orders[name][0],
            }
            rows.append(row)
        summary[corpus] = {"n_emerging": len(base), "published_top5": base[:5]}
    table = pd.DataFrame(rows)
    table.to_csv("rev/weight_ablation.csv", index=False)
    aggregate = table.groupby("variant").agg(
        kendall_min=("kendall_tau", "min"), kendall_max=("kendall_tau", "max"),
        rbo_min=("rbo", "min"), rbo_max=("rbo", "max"),
        top5_min=("top5_overlap", "min"), top5_max=("top5_overlap", "max"),
    ).round(3)
    Path("rev/weight_ablation.json").write_text(json.dumps(
        {"per_corpus": summary, "aggregate": aggregate.to_dict(orient="index")}, indent=2))
    print(table.to_string(index=False), flush=True)
    print("\n" + aggregate.to_string(), flush=True)


if __name__ == "__main__":
    main()
