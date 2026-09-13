"""Frequency-matched validation arm.

The full-corpus control set answers "are the candidates better than an
arbitrary author keyword?".  It cannot separate two contributions that the
pipeline makes at once: the lexical filters (rare, multi-word, not a method
term) and the embedding-based periphery criterion applied on top of them.

This script adds a third arm, ELIGIBLE: the keywords that pass every lexical
and frequency filter but that the ranking did *not* select.  Comparing
SELECTED against ELIGIBLE isolates the contribution of the semantic
criterion; comparing ELIGIBLE against the full corpus isolates the
contribution of the filters.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, "embedresearchgaps/src")
sys.path.append(os.getcwd())

from embedresearchgaps import articles_first, load_csv  # noqa: E402
from embedresearchgaps.gaps import candidate_mask  # noqa: E402
from embedresearchgaps.text import core_domain_terms  # noqa: E402
from embedresearchgaps.validation import (  # noqa: E402
    annotations_to_frame,
    compare_keyword_sets,
    fleiss_kappa,
    krippendorff_alpha_ordinal,
    label_rates,
    mean_novelty,
    selected_and_all_keywords,
)
from run_case import DOMAINS, PROBLEMS, build_config, embed_corpus  # noqa: E402
from run_validation import build_panel  # noqa: E402

SAMPLE = 60
SEED = 2026


def eligible_pool(result, config) -> list[str]:
    """Keywords that survive every filter but were not selected."""
    counts: Counter = Counter()
    for keywords in result.articles["keywords_normalized"]:
        counts.update(set(keywords))
    n_articles = len(result.articles)
    threshold = config.gaps.max_keyword_frequency
    if threshold is None:
        threshold = max(1, round(n_articles / 50))
    core = core_domain_terms(counts, n_articles, config.gaps.core_domain_share)

    keywords = [kw for kw, freq in counts.items() if freq <= threshold]
    mask = candidate_mask(keywords, config.gaps, core)
    pool = [kw for kw, keep in zip(keywords, mask) if keep]

    selected, _ = selected_and_all_keywords(result)
    return sorted(set(pool) - set(selected))


def run(name: str, top_n: int = 50) -> dict[str, object]:
    print(f"[{name}]", flush=True)
    corpus = load_csv(f"corpora/{name}.csv")
    problem = PROBLEMS[name]
    vectors = embed_corpus(name, corpus, problem)
    config = build_config(problem, vectors, top_n)
    result = articles_first(corpus, config)

    pool = eligible_pool(result, config)
    sample = sorted(random.Random(SEED).sample(pool, min(SAMPLE, len(pool))))
    print(f"  eligible pool: {len(pool)} keywords, annotating {len(sample)}", flush=True)

    collected: list = []
    failures: dict[str, str] = {}
    for annotator in build_panel():
        try:
            collected.extend(annotator.annotate(sample, problem, DOMAINS[name], "ELIGIBLE"))
        except Exception as exc:  # noqa: BLE001
            failures[f"{annotator.name}/ELIGIBLE"] = f"{type(exc).__name__}: {exc}"
    eligible_frame = annotations_to_frame(collected)
    if eligible_frame.empty:
        raise RuntimeError(f"every annotator failed on the eligible arm: {failures}")

    out = Path("results") / name
    eligible_frame.to_csv(out / "eligible_arm_annotations.csv", index=False)

    previous = pd.read_csv(out / "articles_first_validation_annotations.csv")
    selected_frame = previous[previous["keyword_set"] == "SELECTED"]
    control_frame = previous[previous["keyword_set"] == "ALL"]

    payload = {
        "n_eligible_pool": len(pool),
        "n_eligible_annotated": len(sample),
        "failures": failures,
        "annotators": sorted(eligible_frame["annotator"].unique()),
        "rates": {
            "SELECTED": label_rates(selected_frame),
            "ELIGIBLE": label_rates(eligible_frame),
            "ALL": label_rates(control_frame),
        },
        "novelty": {
            "SELECTED": mean_novelty(selected_frame),
            "ELIGIBLE": mean_novelty(eligible_frame),
            "ALL": mean_novelty(control_frame),
        },
        "selected_vs_eligible": compare_keyword_sets(selected_frame, eligible_frame).to_dict(),
        "eligible_vs_all": compare_keyword_sets(eligible_frame, control_frame).to_dict(),
        "reliability_eligible": {
            "fleiss_kappa": fleiss_kappa(eligible_frame),
            "krippendorff_alpha": krippendorff_alpha_ordinal(eligible_frame),
        },
    }
    (out / "matched_arm_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    sve = payload["selected_vs_eligible"]
    eva = payload["eligible_vs_all"]
    print(f"  SELECTED vs ELIGIBLE: GAP {sve['gap_rate_selected']:.3f} vs "
          f"{sve['gap_rate_all']:.3f} ({sve['gap_rate_delta_pp']:+.1f} pp), "
          f"novelty {sve['novelty_increase_pct']:+.1f}%, "
          f"Welch p={sve['novelty_test'].get('welch_p'):.3g}", flush=True)
    print(f"  ELIGIBLE vs ALL:      GAP {eva['gap_rate_selected']:.3f} vs "
          f"{eva['gap_rate_all']:.3f} ({eva['gap_rate_delta_pp']:+.1f} pp), "
          f"novelty {eva['novelty_increase_pct']:+.1f}%, "
          f"Welch p={eva['novelty_test'].get('welch_p'):.3g}", flush=True)
    return payload


if __name__ == "__main__":
    for corpus_name in sys.argv[1:] or ["management", "finance"]:
        run(corpus_name)
