"""Construct validity, power and agreement, from the deposited annotations.

Answers, in order: how far Mode B reduces to a rarity filter (R2.7), what the
realised design can detect and whether the null is an equivalence result
(R2.9), and agreement on the ordinal Novelty Score with an interval (R2.11).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path("repo/current/paper/results")
CORPORA = ("management", "finance", "gamification")
MODES = ("articles_first", "keywords_first")
RNG = np.random.default_rng(20260911)

#: Smallest difference in GAP rate we regard as practically relevant.  A
#: shortlist is read by an expert; a 10 pp difference on a 25-item shortlist
#: moves between two and three items, which is the smallest change that would
#: alter how the list is used.  Declared before the tests are run.
SESOI_GAP_PP = 10.0
SESOI_NOVELTY = 0.5


def keyword_level(annotations: pd.DataFrame) -> pd.DataFrame:
    return (
        annotations.assign(is_gap=annotations["label"].eq("GAP").astype(float))
        .groupby(["keyword", "keyword_set"], as_index=False)
        .agg(gap_share=("is_gap", "mean"), novelty=("novelty", "mean"))
    )


def tost(a: np.ndarray, b: np.ndarray, margin: float) -> dict:
    """Two one-sided tests for equivalence of means within +/- margin."""
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    df = (se ** 4) / ((a.var(ddof=1) / a.size) ** 2 / (a.size - 1)
                      + (b.var(ddof=1) / b.size) ** 2 / (b.size - 1))
    t_lower = (diff + margin) / se
    t_upper = (diff - margin) / se
    p_lower = stats.t.sf(t_lower, df)          # H0: diff <= -margin
    p_upper = stats.t.cdf(t_upper, df)         # H0: diff >= +margin
    p = max(p_lower, p_upper)
    return {"diff": float(diff), "margin": float(margin), "p_tost": float(p),
            "equivalent": bool(p < 0.05), "se": float(se), "df": float(df)}


def mde(a: np.ndarray, b: np.ndarray, power: float = 0.80, alpha: float = 0.05) -> float:
    """Minimal detectable difference of means at the realised sizes and spread."""
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)


def krippendorff_ordinal(matrix: np.ndarray) -> float:
    """Ordinal Krippendorff's alpha for a units x raters matrix with NaNs."""
    values = matrix[~np.isnan(matrix)]
    levels = np.unique(values)
    if levels.size < 2:
        return float("nan")

    counts = {v: float((values == v).sum()) for v in levels}
    n_total = values.size

    def delta(v1: float, v2: float) -> float:
        lo, hi = (v1, v2) if v1 <= v2 else (v2, v1)
        inner = [counts[v] for v in levels if lo <= v <= hi]
        return (sum(inner) - (counts[lo] + counts[hi]) / 2.0) ** 2

    observed = 0.0
    pairable = 0
    for row in matrix:
        present = row[~np.isnan(row)]
        m = present.size
        if m < 2:
            continue
        pairable += m
        for i in range(m):
            for j in range(m):
                if i != j:
                    observed += delta(present[i], present[j]) / (m - 1)

    expected = 0.0
    for v1 in levels:
        for v2 in levels:
            if v1 != v2:
                expected += counts[v1] * counts[v2] * delta(v1, v2)
    expected /= (n_total - 1)
    return float(1 - (observed / expected)) if expected else float("nan")


def alpha_with_ci(annotations: pd.DataFrame, n_boot: int = 400) -> dict:
    wide = annotations.pivot_table(index="keyword", columns="annotator",
                                   values="novelty", aggfunc="mean")
    matrix = wide.to_numpy(dtype=float)
    point = krippendorff_ordinal(matrix)
    draws = []
    for _ in range(n_boot):
        rows = RNG.integers(0, matrix.shape[0], matrix.shape[0])
        value = krippendorff_ordinal(matrix[rows])
        if np.isfinite(value):
            draws.append(value)
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan)
    return {"alpha_ordinal": point, "ci": [float(lo), float(hi)],
            "n_units": int(matrix.shape[0]), "n_raters": int(matrix.shape[1])}


def main() -> dict:
    out: dict[str, dict] = {}
    for corpus in CORPORA:
        for mode in MODES:
            path = RESULTS / corpus / f"{mode}_validation_annotations_disjoint.csv"
            if not path.exists():
                continue
            annotations = pd.read_csv(path)
            frame = keyword_level(annotations)
            selected = frame[frame.keyword_set == "SELECTED"]
            control = frame[frame.keyword_set == "CONTROL"]
            k = len(selected)

            keywords = pd.read_csv(RESULTS / corpus / f"{mode}_keywords.csv")
            column = "global_frequency" if "global_frequency" in keywords else "frequency"
            freq = keywords.groupby("keyword")[column].max()
            tokens = keywords.groupby("keyword")["n_tokens"].max()

            population = frame.merge(freq.rename("df"), left_on="keyword",
                                     right_index=True, how="left")
            rarest = set(population.nsmallest(k, "df").head(k)["keyword"])
            candidates = set(selected["keyword"])
            cand_freq = population[population.keyword.isin(candidates)]["df"]
            cand_tok = (frame[frame.keyword_set == "SELECTED"]
                        .merge(tokens.rename("tok"), left_on="keyword",
                               right_index=True, how="left")["tok"])

            record = {
                "k": int(k),
                "overlap_with_rarest_k": len(candidates & rarest) / k,
                "share_singleton_candidates": float((cand_freq <= 1).mean()),
                "share_multitoken_candidates": float((cand_tok >= 2).mean()),
                "median_candidate_df": float(cand_freq.median()),
                "median_population_df": float(population["df"].median()),
            }

            for metric, scale, margin in (("gap_share", 100.0, SESOI_GAP_PP),
                                          ("novelty", 1.0, SESOI_NOVELTY)):
                a = selected[metric].to_numpy() * scale
                b = control[metric].to_numpy() * scale
                record[f"{metric}_mde_80"] = mde(a, b)
                record[f"{metric}_tost"] = tost(a, b, margin)

            record["agreement_novelty"] = alpha_with_ci(annotations)
            out[f"{corpus}|{mode}"] = record
    return out


if __name__ == "__main__":
    results = main()
    Path("rev/validity.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = []
    for key, rec in results.items():
        corpus, mode = key.split("|")
        rows.append({
            "corpus": corpus, "mode": mode, "k": rec["k"],
            "in_rarest_k": round(rec["overlap_with_rarest_k"], 2),
            "singleton": round(rec["share_singleton_candidates"], 2),
            "multitoken": round(rec["share_multitoken_candidates"], 2),
            "MDE_gap_pp": round(rec["gap_share_mde_80"], 1),
            "TOST_gap_p": round(rec["gap_share_tost"]["p_tost"], 3),
            "MDE_nov": round(rec["novelty_mde_80"], 2),
            "TOST_nov_p": round(rec["novelty_tost"]["p_tost"], 3),
            "alpha_ord": round(rec["agreement_novelty"]["alpha_ordinal"], 3),
            "alpha_ci": [round(x, 2) for x in rec["agreement_novelty"]["ci"]],
        })
    print(pd.DataFrame(rows).to_string(index=False))
