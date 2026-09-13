"""Recompute the validation comparisons against a *disjoint* control set.

The first study annotated the candidate set and, as the control, every unique
author keyword of the same analysis corpus -- a set that contains the
candidates. A Welch or Mann-Whitney statistic on such nested samples has no
defined null distribution, because every candidate contributes to both arms.

The annotations already on disk cover the whole keyword population, so the
corrected comparison needs no new model calls: the control arm is simply
restricted to the keywords the pipeline did *not* select. Every reported
p-value is recomputed, and a permutation test on the difference of means is
added, which assumes neither normality nor equal variance.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "embedresearchgaps/src")
sys.path.append(os.getcwd())

from embedresearchgaps.validation import (  # noqa: E402
    compare_keyword_sets,
    fleiss_kappa,
    krippendorff_alpha_ordinal,
    per_keyword_consensus,
)
from embedresearchgaps.validation.metrics import (  # noqa: E402
    expert_llm_concordance,
    pairwise_agreement,
)

CORPORA = ("management", "finance", "gamification")
MODES = ("articles_first", "keywords_first")
HELD_OUT = {"openai:gpt-4o", "google:gemini-2.5-flash"}


def recompute(name: str) -> dict[str, object]:
    out = Path("results") / name
    old = json.loads((out / "validation_summary.json").read_text())
    summaries: dict[str, object] = {
        "corpus": old.get("corpus"), "domain": old.get("domain"),
        "control": "disjoint (candidates removed from the control arm)",
    }

    for mode in MODES:
        frame = pd.read_csv(out / f"{mode}_validation_annotations.csv")
        candidates = sorted(set(frame.loc[frame["keyword_set"] == "SELECTED", "keyword"]))
        selected = frame[frame["keyword_set"] == "SELECTED"].copy()
        control = frame[
            (frame["keyword_set"] == "ALL") & (~frame["keyword"].isin(candidates))
        ].copy()
        control["keyword_set"] = "CONTROL"
        nested = int(frame.loc[frame["keyword_set"] == "ALL", "keyword"].isin(candidates).sum())

        comparison = compare_keyword_sets(selected, control)
        held = selected[selected["annotator"].isin(HELD_OUT)]
        rest = selected[~selected["annotator"].isin(HELD_OUT)]
        proxy = held.rename(columns={"annotator": "expert"})[
            ["expert", "keyword", "novelty", "label"]
        ]
        leave_out = expert_llm_concordance(rest, proxy) if not proxy.empty else {}

        entry = {
            "annotators": sorted(frame["annotator"].unique()),
            "n_annotations": int(len(selected) + len(control)),
            "n_candidates": len(candidates),
            "n_control": int(control["keyword"].nunique()),
            "n_candidate_judgements_removed_from_control": nested,
            "comparison": comparison.to_dict(),
            "reliability": {
                "fleiss_kappa_selected": fleiss_kappa(selected),
                "krippendorff_alpha_selected": krippendorff_alpha_ordinal(selected),
                "fleiss_kappa_control": fleiss_kappa(control),
                "krippendorff_alpha_control": krippendorff_alpha_ordinal(control),
            },
            "leave_out": leave_out,
        }
        summaries[mode] = entry

        pd.concat([selected, control], ignore_index=True).to_csv(
            out / f"{mode}_validation_annotations_disjoint.csv", index=False
        )
        per_keyword_consensus(pd.concat([selected, control], ignore_index=True)).to_csv(
            out / f"{mode}_validation_consensus_disjoint.csv", index=False
        )
        pairwise_agreement(selected).to_csv(
            out / f"{mode}_validation_agreement_disjoint.csv", index=False
        )

        test = comparison.novelty_test
        print(
            f"  {mode}: {len(candidates)} candidates vs {entry['n_control']} control "
            f"(removed {nested} nested judgements)\n"
            f"    GAP {comparison.selected_rates['GAP']:.3f} vs "
            f"{comparison.all_rates['GAP']:.3f} "
            f"({comparison.gap_rate_delta_pp:+.1f} pp) | "
            f"novelty {comparison.selected_novelty['mean']:.2f} vs "
            f"{comparison.all_novelty['mean']:.2f} "
            f"({comparison.novelty_increase_pct:+.1f}%)\n"
            f"    Welch p={test['welch_p']:.3f} | MWU p={test['mannwhitney_p']:.3f} | "
            f"permutation p={test['permutation_p']:.3f} | d={test['cohens_d']:+.3f}",
            flush=True,
        )

    (out / "validation_summary_disjoint.json").write_text(
        json.dumps(summaries, indent=2, default=str), encoding="utf-8"
    )
    return summaries


def recompute_matched(name: str) -> dict[str, object] | None:
    """Redo the eligible-pool contrasts as two fully disjoint comparisons.

    SELECTED vs ELIGIBLE was already disjoint: the eligible pool is defined as
    the filter-passing keywords the ranking did *not* select. The second
    contrast was not -- the eligible pool is a subset of the keyword
    population it was compared against -- so it is recomputed as ELIGIBLE vs
    NOT-ELIGIBLE, the keywords that fail at least one filter. The two
    contrasts then decompose the pipeline cleanly: the filters are what
    separates ELIGIBLE from NOT-ELIGIBLE, and the embedding-based ranking is
    what separates SELECTED from ELIGIBLE.
    """
    path = Path("results") / name / "matched_arm_summary.json"
    if not path.exists():
        return None
    eligible = pd.read_csv(Path("results") / name / "eligible_arm_annotations.csv")
    frame = pd.read_csv(Path("results") / name / "articles_first_validation_annotations.csv")
    candidates = sorted(set(frame.loc[frame["keyword_set"] == "SELECTED", "keyword"]))
    selected = frame[frame["keyword_set"] == "SELECTED"]
    population = frame[
        (frame["keyword_set"] == "ALL") & (~frame["keyword"].isin(candidates))
    ]
    pool = eligible[~eligible["keyword"].isin(candidates)]
    pool_keywords = set(pool["keyword"])
    not_eligible = population[~population["keyword"].isin(pool_keywords)]

    payload = {
        "n_eligible_annotated": int(pool["keyword"].nunique()),
        "n_not_eligible": int(not_eligible["keyword"].nunique()),
        "selected_vs_eligible": compare_keyword_sets(selected, pool).to_dict(),
        "eligible_vs_not_eligible": compare_keyword_sets(pool, not_eligible).to_dict(),
    }
    (Path("results") / name / "matched_arm_summary_disjoint.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    for key in ("selected_vs_eligible", "eligible_vs_not_eligible"):
        entry = payload[key]
        print(
            f"  {key}: GAP {entry['gap_rate_selected']:.3f} vs {entry['gap_rate_all']:.3f} "
            f"({entry['gap_rate_delta_pp']:+.1f} pp) | novelty "
            f"{entry['novelty_increase_pct']:+.1f}% | "
            f"permutation p={entry['novelty_test']['permutation_p']:.3f}",
            flush=True,
        )
    return payload


if __name__ == "__main__":
    for corpus_name in sys.argv[1:] or list(CORPORA):
        print(f"[{corpus_name}]", flush=True)
        recompute(corpus_name)
        recompute_matched(corpus_name)
