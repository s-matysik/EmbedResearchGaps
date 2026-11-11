"""Concordance between a student panel and a model panel on one keyword set.

Reviewers 1, 2 and 3 all asked for human assessment of the candidates.  Besides
the two-expert pilot of the antecedent study (S8b), one further human arm
exists and is reported here: between 3 and 17 June 2026, 23 undergraduates
rated the keyword set produced for the research problem "How do social media
notifications affect users' attention and task performance?", using the same
1-10 Novelty Score and the same five labels as the model panel of S8.  The
raters were 19 second-year students of psychology and communication in
business and 4 students of computer science, whose year of study the form did
not record.  The topic was chosen because it is one these raters know from
daily use, which is the condition under which a non-specialist rating is
informative at all.

The same keyword set had been rated by a single model (Claude Opus 4.5) on
24 May 2026 under the same rubric, two weeks before the survey opened, which
makes the two arms comparable at the keyword level: per-keyword mean Novelty Score, majority label, and the rank
correlation between the two.

The analysis is descriptive.  Raters are undergraduates, not domain experts;
the arm is a second pilot, not the calibrated expert study the paper names as
future work.

Inputs:  rev/pilot/student_ratings_long.csv   (tidied survey responses)
         rev/pilot/model_ratings.csv          (the single-model arm)
Outputs: rev/pilot/student_pilot.csv, rev/pilot/student_pilot.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

OUT = Path("rev/pilot")

#: Spellings differ between the survey and the model table; both are mapped to
#: a single comparison key so that no keyword is silently dropped.
ALIASES = {
    "delay of gratification": "delay of gratification / delay discounting",
    "delay of gratification / delay discounting": "delay of gratification / delay discounting",
    "history of mtbi": "history of mtbi",
    "history of mtbi / mtbi": "history of mtbi",
    "phubbing": "phubbing",
    "phubbing / phubbing": "phubbing",
}


def canonical(name: str) -> str:
    key = str(name).strip().lower()
    return ALIASES.get(key, key)


def main() -> None:
    students = pd.read_csv(OUT / "student_ratings_long.csv")
    # one rater wrote ",6"; read it as 6 rather than discard the judgement
    students["novelty"] = pd.to_numeric(
        students["novelty"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    students = students.dropna(subset=["novelty"])
    students["key"] = students["keyword"].map(canonical)

    model = pd.read_csv(OUT / "model_ratings.csv")
    model["key"] = model["keyword"].map(canonical)
    # "GAP (graniczne)" is a borderline GAP in the source document; it counts as GAP
    model["label"] = model["label"].str.replace(r"\s*\(graniczne\)", "", regex=True)

    per_keyword = (
        students.groupby("key")
        .agg(n_raters=("rater", "nunique"),
             student_mean=("novelty", "mean"),
             student_sd=("novelty", "std"),
             student_gap_share=("label", lambda s: (s == "GAP").mean() * 100))
        .reset_index()
    )
    per_keyword["student_majority"] = (
        students.groupby("key")["label"].agg(lambda s: s.value_counts().idxmax()).values
    )
    merged = per_keyword.merge(
        model[["key", "novelty", "label"]].rename(
            columns={"novelty": "model_novelty", "label": "model_label"}),
        on="key", how="inner")
    merged = merged.sort_values("student_mean", ascending=False)
    merged.to_csv(OUT / "student_pilot.csv", index=False)

    rho, p_rho = stats.spearmanr(merged["student_mean"], merged["model_novelty"])
    agreement = float((merged["student_majority"] == merged["model_label"]).mean())
    model_gap = merged["model_label"] == "GAP"
    summary = {
        "n_raters": int(students["rater"].nunique()),
        "n_keywords_compared": int(len(merged)),
        "n_judgements": int(len(students)),
        "programmes": students.groupby("rater")["programme"].first().value_counts().to_dict(),
        "spearman_rho": round(float(rho), 3),
        "spearman_p": round(float(p_rho), 4),
        "label_agreement": round(agreement, 3),
        "student_gap_share_overall": round(float((students["label"] == "GAP").mean() * 100), 1),
        "model_gap_share": round(float(model_gap.mean() * 100), 1),
        "student_mean_on_model_gaps": round(float(merged.loc[model_gap, "student_mean"].mean()), 2),
        "student_mean_on_model_non_gaps": round(
            float(merged.loc[~model_gap, "student_mean"].mean()), 2),
        "student_gap_share_on_model_gaps": round(
            float(merged.loc[model_gap, "student_gap_share"].mean()), 1),
        "student_gap_share_on_model_non_gaps": round(
            float(merged.loc[~model_gap, "student_gap_share"].mean()), 1),
    }
    gap_scores = merged.loc[model_gap, "student_mean"]
    rest_scores = merged.loc[~model_gap, "student_mean"]
    u, p_u = stats.mannwhitneyu(gap_scores, rest_scores, alternative="two-sided")
    summary["mannwhitney_u"] = float(u)
    summary["mannwhitney_p"] = round(float(p_u), 4)

    (OUT / "student_pilot.json").write_text(json.dumps(summary, indent=2))
    print(merged.round(2).to_string(index=False), flush=True)
    print("\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
