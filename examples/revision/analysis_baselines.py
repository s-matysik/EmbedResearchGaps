"""Reviewer-requested baselines, computed from the deposited panel annotations.

The panel annotated *every* author keyword of each analysis corpus, not only
the candidates, so any subset of the same size can be scored against the same
judgements without a single new model call.  That makes the comparison the
reviewers ask for -- ranking versus random selection, versus a
frequency-ordered list, versus the filters alone -- a re-analysis rather than
a new experiment.

Keyword is the unit of analysis throughout, matching Section 3.3: the five
provider judgements of one keyword are aggregated before any arm is compared.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("repo/current/paper/results")
CORPORA = ("management", "finance", "gamification")
MODES = ("articles_first", "keywords_first")
RNG = np.random.default_rng(20260911)
N_BOOT = 10_000


def keyword_level(annotations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the five provider judgements of each keyword."""
    return (
        annotations.assign(is_gap=annotations["label"].eq("GAP").astype(float))
        .groupby(["keyword", "keyword_set"], as_index=False)
        .agg(gap_share=("is_gap", "mean"), novelty=("novelty", "mean"),
             n_judgements=("is_gap", "size"))
    )


def boot_ci(values: np.ndarray, statistic=np.mean, n: int = N_BOOT) -> tuple[float, float]:
    draws = RNG.choice(values, size=(n, values.size), replace=True)
    return tuple(np.percentile(statistic(draws, axis=1), [2.5, 97.5]))


def diff_ci(a: np.ndarray, b: np.ndarray, n: int = N_BOOT) -> tuple[float, float, float]:
    """Difference of means a - b with a percentile bootstrap interval."""
    da = RNG.choice(a, size=(n, a.size), replace=True).mean(axis=1)
    db = RNG.choice(b, size=(n, b.size), replace=True).mean(axis=1)
    delta = da - db
    return float(a.mean() - b.mean()), *np.percentile(delta, [2.5, 97.5])


def permutation_p(a: np.ndarray, b: np.ndarray, n: int = N_BOOT) -> float:
    observed = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    count = 0
    for _ in range(n):
        RNG.shuffle(pool)
        count += abs(pool[:a.size].mean() - pool[a.size:].mean()) >= observed - 1e-12
    return (count + 1) / (n + 1)


def load(corpus: str, mode: str):
    base = RESULTS / corpus / f"{mode}_validation_annotations_disjoint.csv"
    if not base.exists():
        return None
    annotations = pd.read_csv(base)
    keywords = pd.read_csv(RESULTS / corpus / f"{mode}_keywords.csv")
    frequency_column = "global_frequency" if "global_frequency" in keywords else "frequency"
    frequency = (keywords.groupby("keyword")[frequency_column].max()
                 if "keyword" in keywords else None)
    tokens = keywords.groupby("keyword")["n_tokens"].max() if "n_tokens" in keywords else None
    return keyword_level(annotations), frequency, tokens


def random_baseline(population: pd.DataFrame, size: int, column: str, n: int = 2000):
    """Distribution of a subset statistic under random selection of `size` keywords."""
    values = population[column].to_numpy()
    draws = np.array([RNG.choice(values, size=size, replace=False).mean() for _ in range(n)])
    return draws


def analyse() -> dict:
    out: dict[str, dict] = {}
    for corpus in CORPORA:
        for mode in MODES:
            loaded = load(corpus, mode)
            if loaded is None:
                continue
            frame, frequency, tokens = loaded
            selected = frame[frame["keyword_set"] == "SELECTED"]
            control = frame[frame["keyword_set"] == "CONTROL"]
            if selected.empty or control.empty:
                continue
            k = len(selected)
            population = frame  # candidates + control = every annotated keyword

            record: dict[str, object] = {
                "n_candidates": int(k),
                "n_control": int(len(control)),
                "n_population": int(len(population)),
            }

            for metric, scale in (("gap_share", 100.0), ("novelty", 1.0)):
                arm = selected[metric].to_numpy() * scale
                ctrl = control[metric].to_numpy() * scale
                delta, lo, hi = diff_ci(arm, ctrl)
                record[f"{metric}_candidates"] = float(arm.mean())
                record[f"{metric}_control"] = float(ctrl.mean())
                record[f"{metric}_delta"] = delta
                record[f"{metric}_ci"] = [float(lo), float(hi)]
                record[f"{metric}_p_perm"] = permutation_p(arm, ctrl, n=5000)

                # --- baselines of the same size, scored on the same judgements ---
                if frequency is not None:
                    merged = population.merge(
                        frequency.rename("df"), left_on="keyword", right_index=True, how="left"
                    ).fillna({"df": 0})
                    rarest = merged.nsmallest(k, "df", keep="all").head(k)
                    commonest = merged.nlargest(k, "df", keep="all").head(k)
                    record[f"{metric}_rarest_k"] = float(rarest[metric].mean() * scale)
                    record[f"{metric}_commonest_k"] = float(commonest[metric].mean() * scale)

                draws = random_baseline(population, k, metric) * scale
                record[f"{metric}_random_mean"] = float(draws.mean())
                record[f"{metric}_random_ci"] = [float(np.percentile(draws, 2.5)),
                                                 float(np.percentile(draws, 97.5))]
                record[f"{metric}_random_percentile_of_ranking"] = float(
                    (draws < arm.mean()).mean() * 100
                )

            # --- rarity as a covariate (R2 point 8) ---
            if frequency is not None:
                merged = population.merge(
                    frequency.rename("df"), left_on="keyword", right_index=True, how="left"
                ).dropna(subset=["df"])
                if len(merged) > 10:
                    from scipy import stats
                    rho, p = stats.spearmanr(merged["novelty"], merged["df"])
                    record["rarity_spearman_rho"] = float(rho)
                    record["rarity_spearman_p"] = float(p)
                    record["rarity_n"] = int(len(merged))
                    rho_g, p_g = stats.spearmanr(merged["gap_share"], merged["df"])
                    record["rarity_gap_rho"] = float(rho_g)
                    record["rarity_gap_p"] = float(p_g)
            if tokens is not None:
                merged = population.merge(
                    tokens.rename("tok"), left_on="keyword", right_index=True, how="left"
                ).dropna(subset=["tok"])
                if len(merged) > 10:
                    from scipy import stats
                    rho_t, p_t = stats.spearmanr(merged["novelty"], merged["tok"])
                    record["tokens_spearman_rho"] = float(rho_t)
                    record["tokens_spearman_p"] = float(p_t)

            out[f"{corpus}|{mode}"] = record
    return out


if __name__ == "__main__":
    results = analyse()
    Path("rev").mkdir(exist_ok=True)
    Path("rev/baselines.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = []
    for key, rec in results.items():
        corpus, mode = key.split("|")
        rows.append({
            "corpus": corpus, "mode": mode, "k": rec["n_candidates"],
            "gap_rank": round(rec["gap_share_candidates"], 1),
            "gap_ctrl": round(rec["gap_share_control"], 1),
            "gap_random": round(rec["gap_share_random_mean"], 1),
            "gap_rarest": round(rec.get("gap_share_rarest_k", float("nan")), 1),
            "gap_common": round(rec.get("gap_share_commonest_k", float("nan")), 1),
            "nov_rank": round(rec["novelty_candidates"], 2),
            "nov_ctrl": round(rec["novelty_control"], 2),
            "nov_random": round(rec["novelty_random_mean"], 2),
            "nov_rarest": round(rec.get("novelty_rarest_k", float("nan")), 2),
            "pct_vs_random": round(rec["gap_share_random_percentile_of_ranking"], 0),
            "rarity_rho": round(rec.get("rarity_spearman_rho", float("nan")), 2),
        })
    table = pd.DataFrame(rows)
    table.to_csv("rev/baselines_table.csv", index=False)
    print(table.to_string(index=False))
