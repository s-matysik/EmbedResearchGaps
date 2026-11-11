"""Audit a ranking produced outside this package (supplementary S17).

The baseline of Section 3.4 is a frequency-ordered shortlist: the k rarest
author keywords that pass the token-count rule, ordered by document frequency.
It is written to a CSV exactly as another tool would export one, read back
through ``load_external_ranking``, and scored against the same disjoint
control and the same deposited panel judgements as the package's own
candidates.  No model is called: the panel annotated every author keyword of
each corpus, so any shortlist of the same size can be scored on the judgements
already on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "repo/current/src")

from embedresearchgaps import load_csv  # noqa: E402
from embedresearchgaps.external import load_external_ranking  # noqa: E402

RESULTS = Path("repo/current/paper/results")
RNG = np.random.default_rng(11)


def keyword_level(annotations: pd.DataFrame) -> pd.DataFrame:
    return (
        annotations.assign(is_gap=annotations["label"].eq("GAP").astype(float))
        .groupby("keyword", as_index=False)
        .agg(gap=("is_gap", "mean"), novelty=("novelty", "mean"))
    )


def audit(corpus_name: str, mode: str = "articles_first") -> dict:
    corpus = load_csv(f"repo/current/data/{corpus_name}.csv")
    annotations = pd.read_csv(RESULTS / corpus_name / f"{mode}_validation_annotations_disjoint.csv")
    judged = keyword_level(annotations)
    ours = pd.read_csv(RESULTS / corpus_name / f"{mode}_gaps.csv")
    k = ours["keyword"].nunique()

    meta = (pd.read_csv(RESULTS / corpus_name / f"{mode}_keywords.csv")
            .groupby("keyword").agg(df=("global_frequency", "max"), tok=("n_tokens", "max")))
    pool = meta[meta["tok"] >= 2].sort_values("df")
    external_file = Path("rev/external_frequency_ranking.csv")
    (pool.head(k).reset_index()
         .assign(score=lambda f: 1.0 / (1.0 + f["df"]))
         .rename(columns={"keyword": "keyword"})[["keyword", "score"]]
         .to_csv(external_file, index=False))

    external = load_external_ranking(external_file, corpus=corpus)
    keywords = [k_.lower() for k_ in external.gaps["keyword"]]

    scored = judged[judged["keyword"].str.lower().isin(keywords)]
    control = judged[~judged["keyword"].str.lower().isin(keywords)]
    ours_kw = {k_.lower() for k_ in ours["keyword"]}
    ours_scored = judged[judged["keyword"].str.lower().isin(ours_kw)]

    return {
        "corpus": corpus_name,
        "k_requested": int(k),
        "k_loaded": int(len(external.gaps)),
        "mode_of_result": external.mode,
        "n_scored": int(len(scored)),
        "external_gap_rate": float(scored["gap"].mean() * 100),
        "external_novelty": float(scored["novelty"].mean()),
        "control_gap_rate": float(control["gap"].mean() * 100),
        "package_gap_rate": float(ours_scored["gap"].mean() * 100),
        "package_novelty": float(ours_scored["novelty"].mean()),
        "overlap_with_package": float(
            len(set(keywords) & ours_kw) / len(set(keywords) | ours_kw)
        ),
    }


if __name__ == "__main__":
    out = [audit(name) for name in ("management", "finance", "gamification")]
    Path("rev/external_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    frame = pd.DataFrame(out)
    print(frame[["corpus", "k_requested", "k_loaded", "mode_of_result", "n_scored",
                 "external_gap_rate", "package_gap_rate", "control_gap_rate",
                 "overlap_with_package"]].round(2).to_string(index=False))
