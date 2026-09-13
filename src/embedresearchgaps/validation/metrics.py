"""Validation metrics: label rates, novelty, annotator agreement, expert concordance."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .protocol import LABELS, Annotation

__all__ = [
    "annotations_to_frame",
    "label_rates",
    "mean_novelty",
    "per_keyword_consensus",
    "fleiss_kappa",
    "krippendorff_alpha_ordinal",
    "pairwise_agreement",
    "SetComparison",
    "compare_keyword_sets",
    "expert_llm_concordance",
]


def annotations_to_frame(annotations: Sequence[Annotation]) -> pd.DataFrame:
    """Long-format table with one row per (annotator, keyword) judgement."""
    if not annotations:
        return pd.DataFrame(
            columns=["annotator", "keyword_set", "keyword", "label", "novelty", "rationale"]
        )
    return pd.DataFrame(
        [
            {
                "annotator": a.annotator,
                "keyword_set": a.keyword_set,
                "keyword": a.keyword,
                "label": a.label,
                "novelty": a.novelty,
                "rationale": a.rationale,
            }
            for a in annotations
        ]
    )


def label_rates(frame: pd.DataFrame) -> dict[str, float]:
    """Share of judgements carrying each label.

    The GAP rate is the number of GAP labels divided by the total number of
    (annotator, keyword) judgements -- the definition used in the PACIS paper.
    """
    if frame.empty:
        return {label: 0.0 for label in LABELS}
    total = len(frame)
    counts = frame["label"].value_counts()
    return {label: float(counts.get(label, 0)) / total for label in LABELS}


def mean_novelty(frame: pd.DataFrame) -> dict[str, float]:
    """Mean, standard deviation and quartiles of the Novelty Score."""
    if frame.empty:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    novelty = frame["novelty"].astype(float)
    return {
        "mean": float(novelty.mean()),
        "std": float(novelty.std(ddof=1)) if len(novelty) > 1 else 0.0,
        "median": float(novelty.median()),
        "q25": float(novelty.quantile(0.25)),
        "q75": float(novelty.quantile(0.75)),
        "share_ge_5": float((novelty >= 5).mean()),
        "share_ge_8": float((novelty >= 8).mean()),
        "n": int(len(novelty)),
    }


def per_keyword_consensus(frame: pd.DataFrame) -> pd.DataFrame:
    """Majority label, unanimity and mean novelty per keyword."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "keyword", "keyword_set", "n_annotators", "gap_votes", "gap_share",
                "majority_label", "unanimous", "mean_novelty", "std_novelty",
            ]
        )
    rows: list[dict[str, Any]] = []
    group_columns = ["keyword_set", "keyword"] if frame["keyword_set"].any() else ["keyword"]
    for key, group in frame.groupby(group_columns, dropna=False):
        keys = key if isinstance(key, tuple) else (key,)
        counts = group["label"].value_counts()
        rows.append(
            {
                "keyword_set": keys[0] if len(group_columns) == 2 else "",
                "keyword": keys[-1],
                "n_annotators": int(group["annotator"].nunique()),
                "gap_votes": int(counts.get("GAP", 0)),
                "gap_share": float(counts.get("GAP", 0) / len(group)),
                "majority_label": str(counts.idxmax()),
                "unanimous": bool(len(counts) == 1),
                "mean_novelty": float(group["novelty"].mean()),
                "std_novelty": float(group["novelty"].std(ddof=1)) if len(group) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["gap_share", "mean_novelty"], ascending=False, ignore_index=True
    )


def _label_count_matrix(frame: pd.DataFrame) -> np.ndarray:
    """``(n_keywords, n_labels)`` matrix of label counts per keyword."""
    pivot = (
        frame.pivot_table(
            index="keyword", columns="label", values="novelty", aggfunc="count", fill_value=0
        )
        .reindex(columns=list(LABELS), fill_value=0)
        .to_numpy(dtype=float)
    )
    return pivot


def fleiss_kappa(frame: pd.DataFrame) -> float:
    """Fleiss' kappa over the nominal label assignments.

    Returns ``nan`` when fewer than two annotators rated every keyword, or
    when all annotators agree on a single label for the whole set (in which
    case chance agreement is 1 and kappa is undefined).
    """
    matrix = _label_count_matrix(frame)
    if matrix.size == 0:
        return float("nan")
    n_raters = matrix.sum(axis=1)
    if n_raters.min() < 2 or not np.allclose(n_raters, n_raters[0]):
        # Unequal rater counts: restrict to keywords rated by the mode count.
        mode = stats.mode(n_raters, keepdims=False).mode
        keep = n_raters == mode
        if keep.sum() < 2 or mode < 2:
            return float("nan")
        matrix, n_raters = matrix[keep], n_raters[keep]
    n = float(n_raters[0])
    proportions = matrix.sum(axis=0) / (len(matrix) * n)
    agreement = ((matrix**2).sum(axis=1) - n) / (n * (n - 1))
    p_bar = float(agreement.mean())
    p_expected = float((proportions**2).sum())
    if np.isclose(p_expected, 1.0):
        return float("nan")
    return (p_bar - p_expected) / (1.0 - p_expected)


def krippendorff_alpha_ordinal(frame: pd.DataFrame) -> float:
    """Krippendorff's alpha for the ordinal Novelty Score.

    Uses the ``krippendorff`` package when installed; otherwise returns
    ``nan`` rather than substituting a different coefficient.
    """
    try:
        import krippendorff
    except ImportError:  # pragma: no cover - optional dependency
        return float("nan")
    wide = frame.pivot_table(
        index="annotator", columns="keyword", values="novelty", aggfunc="mean"
    )
    if wide.shape[0] < 2 or wide.shape[1] < 2:
        return float("nan")
    data = wide.to_numpy(dtype=float)
    if len(np.unique(data[~np.isnan(data)])) < 2:
        # Every annotator gave every keyword the same score: alpha is
        # undefined because the value domain has a single point.
        return float("nan")
    try:
        return float(
            krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")
        )
    except ValueError:
        return float("nan")


def pairwise_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    """Label agreement and novelty correlation for every annotator pair."""
    annotators = sorted(frame["annotator"].unique())
    rows: list[dict[str, Any]] = []
    for first, second in itertools.combinations(annotators, 2):
        left = frame[frame["annotator"] == first].set_index("keyword")
        right = frame[frame["annotator"] == second].set_index("keyword")
        shared = left.index.intersection(right.index)
        if len(shared) == 0:
            continue
        label_match = float((left.loc[shared, "label"] == right.loc[shared, "label"]).mean())
        if len(shared) > 2:
            rho, p_value = stats.spearmanr(
                left.loc[shared, "novelty"].astype(float),
                right.loc[shared, "novelty"].astype(float),
            )
        else:
            rho, p_value = float("nan"), float("nan")
        rows.append(
            {
                "annotator_a": first,
                "annotator_b": second,
                "n_keywords": int(len(shared)),
                "label_agreement": label_match,
                "novelty_spearman": float(rho),
                "novelty_p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class SetComparison:
    """SELECTED vs ALL comparison, the headline validation result."""

    selected_rates: dict[str, float]
    all_rates: dict[str, float]
    selected_novelty: dict[str, float]
    all_novelty: dict[str, float]
    gap_rate_delta_pp: float
    explored_rate_delta_pp: float
    novelty_increase_pct: float
    novelty_test: dict[str, float] = field(default_factory=dict)
    n_selected_keywords: int = 0
    n_all_keywords: int = 0
    n_annotators: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_annotators": self.n_annotators,
            "n_selected_keywords": self.n_selected_keywords,
            "n_all_keywords": self.n_all_keywords,
            "gap_rate_selected": self.selected_rates.get("GAP", float("nan")),
            "gap_rate_all": self.all_rates.get("GAP", float("nan")),
            "gap_rate_delta_pp": self.gap_rate_delta_pp,
            "explored_rate_selected": self.selected_rates.get("EXPLORED", float("nan")),
            "explored_rate_all": self.all_rates.get("EXPLORED", float("nan")),
            "explored_rate_delta_pp": self.explored_rate_delta_pp,
            "mean_novelty_selected": self.selected_novelty.get("mean", float("nan")),
            "mean_novelty_all": self.all_novelty.get("mean", float("nan")),
            "novelty_increase_pct": self.novelty_increase_pct,
            "selected_rates": self.selected_rates,
            "all_rates": self.all_rates,
            "selected_novelty": self.selected_novelty,
            "all_novelty": self.all_novelty,
            "novelty_test": self.novelty_test,
        }


def compare_keyword_sets(
    selected: pd.DataFrame,
    everything: pd.DataFrame,
) -> SetComparison:
    """Compare annotations of the SELECTED and ALL keyword sets.

    ``selected`` holds judgements of the keywords the pipeline flagged,
    ``everything`` judgements of the unfiltered corpus keywords.  Novelty
    distributions are compared with a Welch t-test and a Mann-Whitney U test,
    the two tests reported in the PACIS paper.
    """
    selected_rates, all_rates = label_rates(selected), label_rates(everything)
    selected_novelty, all_novelty = mean_novelty(selected), mean_novelty(everything)

    test: dict[str, float] = {}
    if not selected.empty and not everything.empty:
        left = selected["novelty"].astype(float).to_numpy()
        right = everything["novelty"].astype(float).to_numpy()
        t_stat, t_p = stats.ttest_ind(left, right, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(left, right, alternative="two-sided")
        pooled = np.sqrt((np.var(left, ddof=1) + np.var(right, ddof=1)) / 2) if len(left) > 1 and len(right) > 1 else np.nan
        test = {
            "welch_t": float(t_stat),
            "welch_p": float(t_p),
            "mannwhitney_u": float(u_stat),
            "mannwhitney_p": float(u_p),
            "cohens_d": float((left.mean() - right.mean()) / pooled) if pooled and pooled > 0 else float("nan"),
        }

    base = all_novelty.get("mean", float("nan"))
    return SetComparison(
        selected_rates=selected_rates,
        all_rates=all_rates,
        selected_novelty=selected_novelty,
        all_novelty=all_novelty,
        gap_rate_delta_pp=100.0 * (selected_rates["GAP"] - all_rates["GAP"]),
        explored_rate_delta_pp=100.0 * (selected_rates["EXPLORED"] - all_rates["EXPLORED"]),
        novelty_increase_pct=(
            100.0 * (selected_novelty.get("mean", np.nan) - base) / base
            if base and not np.isnan(base) and base != 0
            else float("nan")
        ),
        novelty_test=test,
        n_selected_keywords=int(selected["keyword"].nunique()) if not selected.empty else 0,
        n_all_keywords=int(everything["keyword"].nunique()) if not everything.empty else 0,
        n_annotators=int(
            pd.concat([selected, everything])["annotator"].nunique()
            if not (selected.empty and everything.empty)
            else 0
        ),
    )


def expert_llm_concordance(
    llm_frame: pd.DataFrame,
    expert_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Agreement between human experts and the LLM panel.

    ``expert_frame`` must carry ``expert``, ``keyword``, ``novelty`` and
    optionally ``label``.  Reports the Spearman correlation between mean
    expert and mean LLM novelty on the shared keywords, the share of keywords
    on which the majority labels coincide, and the spread among experts and
    among models.
    """
    required = {"expert", "keyword", "novelty"}
    missing = required - set(expert_frame.columns)
    if missing:
        raise ValueError(f"expert_frame is missing column(s) {sorted(missing)}")

    llm_mean = llm_frame.groupby("keyword")["novelty"].mean()
    expert_mean = expert_frame.assign(
        novelty=expert_frame["novelty"].astype(float)
    ).groupby("keyword")["novelty"].mean()
    shared = llm_mean.index.intersection(expert_mean.index)
    if len(shared) < 3:
        raise ValueError(
            f"need at least 3 shared keywords for a correlation, got {len(shared)}"
        )
    rho, p_value = stats.spearmanr(
        expert_mean.loc[shared].astype(float), llm_mean.loc[shared].astype(float)
    )

    result: dict[str, Any] = {
        "n_shared_keywords": int(len(shared)),
        "n_experts": int(expert_frame["expert"].nunique()),
        "n_llm_annotators": int(llm_frame["annotator"].nunique()),
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
        "mean_novelty_expert": float(expert_mean.loc[shared].mean()),
        "mean_novelty_llm": float(llm_mean.loc[shared].mean()),
        "expert_dispersion": float(
            expert_frame.groupby("keyword")["novelty"].std(ddof=1).mean()
        ),
        "llm_dispersion": float(llm_frame.groupby("keyword")["novelty"].std(ddof=1).mean()),
    }

    if "label" in expert_frame.columns:
        expert_majority = (
            expert_frame.groupby("keyword")["label"]
            .agg(lambda s: s.value_counts().idxmax())
            .loc[shared]
        )
        llm_majority = (
            llm_frame.groupby("keyword")["label"]
            .agg(lambda s: s.value_counts().idxmax())
            .loc[shared]
        )
        result["label_agreement"] = float((expert_majority == llm_majority).mean())
        result["expert_gap_share"] = float((expert_majority == "GAP").mean())
        result["llm_gap_share"] = float((llm_majority == "GAP").mean())
    return result
