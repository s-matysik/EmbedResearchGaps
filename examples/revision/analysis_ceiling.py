"""How much does the candidate set depend on the frequency ceiling?

Reviewers 1 and 2 both observe that the emerging-concept detector is gated by a
document-frequency ceiling, and ask what the ranking contributes once that
filter has done its work.  The reported runs held the ceiling at one article.
This script varies it -- 1, 2, 3 articles, and the proportional rule
max(1, round(N/50)) the source procedure offers -- with everything else fixed,
and reports how the candidate set changes.

The offline tfidf-svd backend is used so the measurement needs no credentials
and can be rerun from the deposited corpora; absolute counts therefore differ
from the hosted-backend runs of Section 3, and only the comparison across
ceilings is of interest.

Output: rev/ceiling_sensitivity.csv, rev/ceiling_sensitivity.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "repo/current/src")

from embedresearchgaps import (  # noqa: E402
    ClusteringConfig, EncoderConfig, GapConfig, RunConfig, articles_first, load_csv,
)

DATA = Path("repo/current/data")
RESULTS = Path("repo/current/paper/results")
CEILINGS = {"1 (reported)": 1, "2": 2, "3": 3, "proportional max(1, round(N/50))": None}


def run(corpus: str, ceiling: int | None) -> set[str]:
    problem = json.loads(
        (RESULTS / corpus / "articles_first_run_config.json").read_text()
    )["problem_description"]
    data = load_csv(DATA / f"{corpus}.csv")
    config = RunConfig(
        top_n_articles=50,
        problem_description=problem,
        article_encoder=EncoderConfig(name="tfidf-svd"),
        keyword_encoder=EncoderConfig(name="tfidf-svd"),
        clustering=ClusteringConfig(random_state=42),
        gaps=GapConfig(max_keyword_frequency=ceiling),
    )
    result = articles_first(data, config)
    return set(result.gaps[result.gaps.gap_type == "Emerging Concept"]["keyword"])


def main() -> None:
    rows = []
    for corpus in ("management", "finance", "gamification"):
        sets = {label: run(corpus, value) for label, value in CEILINGS.items()}
        reference = sets["1 (reported)"]
        for label, found in sets.items():
            union = reference | found
            rows.append({
                "corpus": corpus,
                "ceiling": label,
                "n_emerging": len(found),
                "shared_with_reported": len(reference & found),
                "jaccard_vs_reported": round(
                    len(reference & found) / len(union), 3) if union else 0.0,
            })
            print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    table.to_csv("rev/ceiling_sensitivity.csv", index=False)
    Path("rev/ceiling_sensitivity.json").write_text(
        json.dumps(table.to_dict(orient="records"), indent=2))
    print("\n" + table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
