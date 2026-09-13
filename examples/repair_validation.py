"""Complete the validation studies whose control arm lost one annotator.

The first pass ran Gemini with an 8k output-token budget, which truncated its
reply on the largest control batches and cost the control arm one annotator
while the candidate arm kept all five.  An asymmetric panel is not a valid
comparison, so this script re-runs that annotator alone with a 16k budget and
recomputes every report over the union of old and new judgements.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "embedresearchgaps/src")
sys.path.append(os.getcwd())

from embedresearchgaps import articles_first, keywords_first, load_csv  # noqa: E402
from embedresearchgaps.validation import (  # noqa: E402
    Annotation,
    ChatAnnotator,
    selected_and_all_keywords,
    validate_gaps,
)
from run_case import DOMAINS, PROBLEMS, build_config, embed_corpus  # noqa: E402

MISSING_PROVIDER = ("google", "gemini-2.5-flash", "GOOGLE")


def to_annotations(frame: pd.DataFrame, keyword_set: str) -> list[Annotation]:
    subset = frame[frame["keyword_set"] == keyword_set]
    return [
        Annotation(
            keyword=str(row["keyword"]),
            label=str(row["label"]),
            novelty=float(row["novelty"]),
            annotator=str(row["annotator"]),
            rationale=str(row.get("rationale", "") or ""),
            keyword_set=keyword_set,
        )
        for _, row in subset.iterrows()
    ]


def repair(name: str, top_n: int = 50) -> dict[str, object]:
    print(f"[{name}]", flush=True)
    corpus = load_csv(f"corpora/{name}.csv")
    problem, domain = PROBLEMS[name], DOMAINS[name]
    vectors = embed_corpus(name, corpus, problem)
    config = build_config(problem, vectors, top_n)
    out = Path("results") / name

    reference = articles_first(corpus, config)
    _, control = selected_and_all_keywords(reference)

    provider, model, key_env = MISSING_PROVIDER
    annotator = ChatAnnotator(
        provider, model=model, key_env=key_env, batch_size=15, max_output_tokens=16384
    )
    fresh = annotator.annotate(control, problem, domain, "ALL")
    print(f"  {annotator.name}: {len(fresh)} control judgements recovered", flush=True)

    summaries: dict[str, object] = {}
    for mode, runner in (("articles_first", articles_first), ("keywords_first", keywords_first)):
        path = out / f"{mode}_validation_annotations.csv"
        previous = pd.read_csv(path)
        already = set(previous[previous["keyword_set"] == "ALL"]["annotator"].unique())
        control_annotations = to_annotations(previous, "ALL")
        if annotator.name not in already:
            control_annotations += fresh
        result = reference if mode == "articles_first" else runner(corpus, config)
        selected, _ = selected_and_all_keywords(result)

        report = validate_gaps(
            result, [], problem, domain,
            selected=selected, all_keywords=control,
            all_annotations=control_annotations,
            selected_annotations=to_annotations(previous, "SELECTED"),
        )
        report.save(out, prefix=f"{mode}_validation")
        payload = report.summary()
        summaries[mode] = payload
        comparison = payload["comparison"]
        print(
            f"  {mode}: annotators SELECTED="
            f"{report.annotations[report.annotations['keyword_set'] == 'SELECTED']['annotator'].nunique()}"
            f" ALL="
            f"{report.annotations[report.annotations['keyword_set'] == 'ALL']['annotator'].nunique()}"
            f"  GAP {comparison['gap_rate_selected']:.3f} vs {comparison['gap_rate_all']:.3f}"
            f" ({comparison['gap_rate_delta_pp']:+.1f} pp);"
            f" novelty {comparison['mean_novelty_selected']:.2f} vs "
            f"{comparison['mean_novelty_all']:.2f}"
            f" ({comparison['novelty_increase_pct']:+.1f}%);"
            f" Welch p={comparison['novelty_test']['welch_p']:.3g}",
            flush=True,
        )

    (out / "validation_summary.json").write_text(
        json.dumps(summaries, indent=2, default=str), encoding="utf-8"
    )
    return summaries


if __name__ == "__main__":
    for corpus_name in sys.argv[1:] or ["management", "finance", "gamification"]:
        repair(corpus_name)
