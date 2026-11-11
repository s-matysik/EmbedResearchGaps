"""The decisive baseline: does the ranking beat a random pick of rare keywords?

A "rarest k" list is ill-defined here, because far more than k keywords occur
exactly once and the tie is broken arbitrarily.  The comparator that answers
the reviewers' question -- is Mode B essentially a rarity filter? -- is a
random draw of k keywords from the rare stratum the filters admit, scored on
the same panel judgements.  If the ranking carries information beyond rarity,
it should sit in the upper tail of that distribution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("repo/current/paper/results")
RNG = np.random.default_rng(20260911)
N_DRAWS = 5000


def keyword_level(annotations: pd.DataFrame) -> pd.DataFrame:
    return (
        annotations.assign(is_gap=annotations["label"].eq("GAP").astype(float))
        .groupby(["keyword", "keyword_set"], as_index=False)
        .agg(gap_share=("is_gap", "mean"), novelty=("novelty", "mean"))
    )


def stratum_draws(pool: pd.DataFrame, k: int, column: str) -> np.ndarray:
    values = pool[column].to_numpy()
    if values.size < k:
        return np.array([])
    return np.array([RNG.choice(values, size=k, replace=False).mean()
                     for _ in range(N_DRAWS)])


def main() -> dict:
    out = {}
    for corpus in ("management", "finance", "gamification"):
        for mode in ("articles_first", "keywords_first"):
            path = RESULTS / corpus / f"{mode}_validation_annotations_disjoint.csv"
            if not path.exists():
                continue
            frame = keyword_level(pd.read_csv(path))
            keywords = pd.read_csv(RESULTS / corpus / f"{mode}_keywords.csv")
            column = "global_frequency" if "global_frequency" in keywords else "frequency"
            meta = keywords.groupby("keyword").agg(df=(column, "max"), tok=("n_tokens", "max"))
            merged = frame.merge(meta, left_on="keyword", right_index=True, how="left").dropna(subset=["df"])

            selected = merged[merged.keyword_set == "SELECTED"]
            k = len(selected)
            if k < 3:
                continue
            ceiling = selected["df"].max()      # the frequency ceiling the run applied
            # Rare stratum: every annotated keyword the frequency ceiling admits,
            # with the same multi-token rule the filters impose.
            pool = merged[(merged["df"] <= ceiling) & (merged["tok"] >= 2)]
            record = {"k": int(k), "frequency_ceiling": float(ceiling),
                      "pool_size": int(len(pool))}
            for metric, scale in (("gap_share", 100.0), ("novelty", 1.0)):
                observed = float(selected[metric].mean() * scale)
                draws = stratum_draws(pool, k, metric) * scale
                record[f"{metric}_ranking"] = observed
                if draws.size:
                    record[f"{metric}_rare_random_mean"] = float(draws.mean())
                    record[f"{metric}_rare_random_ci"] = [float(np.percentile(draws, 2.5)),
                                                          float(np.percentile(draws, 97.5))]
                    record[f"{metric}_percentile"] = float((draws < observed).mean() * 100)
            out[f"{corpus}|{mode}"] = record
    return out


if __name__ == "__main__":
    results = main()
    Path("rev/rarity_matched.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = []
    for key, rec in results.items():
        corpus, mode = key.split("|")
        rows.append({
            "corpus": corpus, "mode": mode, "k": rec["k"],
            "ceiling": rec["frequency_ceiling"], "pool": rec["pool_size"],
            "gap_ranking": round(rec["gap_share_ranking"], 1),
            "gap_rare_random": round(rec.get("gap_share_rare_random_mean", float("nan")), 1),
            "gap_rare_ci": [round(x, 1) for x in rec.get("gap_share_rare_random_ci", [float("nan")] * 2)],
            "gap_pctile": round(rec.get("gap_share_percentile", float("nan"))),
            "nov_ranking": round(rec["novelty_ranking"], 2),
            "nov_rare_random": round(rec.get("novelty_rare_random_mean", float("nan")), 2),
            "nov_pctile": round(rec.get("novelty_percentile", float("nan"))),
        })
    print(pd.DataFrame(rows).to_string(index=False))
