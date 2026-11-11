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

#: Analysis-corpus sizes at which the ceiling is varied.  N matters because the
#: proportional rule the source procedure offers is max(1, round(N / 50)): at
#: N = 50 it equals the fixed ceiling of one article, so the two rules can only
#: be told apart at a larger N.
SIZES = (50, 200)

#: ``None`` is NOT the proportional rule in this mode.  ``articles_first``
#: passes the configured value straight to the detectors, where ``None`` means
#: "no frequency ceiling at all"; only ``keywords_first`` resolves ``None``
#: through ``_resolve_frequency_threshold``.  The variants below are therefore
#: named by the threshold each one actually applies.
def variants(n_articles: int) -> dict[str, int | None]:
    proportional = max(1, round(n_articles / 50))
    return {
        "1 (reported)": 1,
        "2": 2,
        "3": 3,
        f"proportional max(1, round(N/50)) = {proportional}": proportional,
        "none (no frequency ceiling)": None,
    }


def run(corpus: str, ceiling: int | None, top_n: int) -> set[str]:
    problem = json.loads(
        (RESULTS / corpus / "articles_first_run_config.json").read_text()
    )["problem_description"]
    data = load_csv(DATA / f"{corpus}.csv")
    config = RunConfig(
        top_n_articles=top_n,
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
        for top_n in SIZES:
            table = variants(top_n)
            sets = {label: run(corpus, value, top_n) for label, value in table.items()}
            reference = sets["1 (reported)"]
            for label, found in sets.items():
                union = reference | found
                rows.append({
                    "corpus": corpus,
                    "top_n": top_n,
                    "ceiling": label,
                    "effective_threshold": (
                        "none" if table[label] is None else table[label]),
                    "n_emerging": len(found),
                    "shared_with_reference": len(reference & found),
                    "jaccard_vs_reference": round(
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
