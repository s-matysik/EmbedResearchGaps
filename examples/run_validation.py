"""Multi-model LLM validation of the case-study gap candidates.

Five providers annotate, at temperature 0, both the keywords the pipeline
selected (SELECTED) and the most frequent author keywords of the same analysis
corpus (ALL).  The report contains label rates, Novelty Score distributions,
inter-annotator reliability and the SELECTED-vs-ALL comparison.

A leave-out check re-uses two of the five models as held-out annotators and
correlates them against the remaining three, exercising the same concordance
code path that a human expert study uses.  It is a model-versus-model check,
not a substitute for expert judgement.
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
    build_expert_sheet,
    expert_llm_concordance,
    get_annotator,
    selected_and_all_keywords,
    validate_gaps,
    write_expert_sheet,
)
from embedresearchgaps.viz import plot_validation_comparison  # noqa: E402
from run_case import DOMAINS, PROBLEMS, build_config, embed_corpus  # noqa: E402

#: Provider, model and the environment variable holding its key.
#:
#: Moonshot (Kimi) was probed and excluded: its current models reject
#: ``temperature=0`` ("only 1 is allowed for this model"), which would break
#: the determinism requirement of the annotation protocol.
PANEL: list[tuple[str, str, str]] = [
    ("openai", "gpt-4o", "OPENAI"),
    ("anthropic", "claude-sonnet-4-6", "ANTROPIC"),
    ("deepseek", "deepseek-v4-pro", "DEEPSEEK"),
    ("xai", "grok-4.5", "XAI"),
    ("google", "gemini-2.5-flash", "GOOGLE"),
]

HELD_OUT = {"openai:gpt-4o", "google:gemini-2.5-flash"}


def build_panel() -> list:
    return [
        get_annotator(provider, model=model, key_env=key_env, batch_size=25)
        for provider, model, key_env in PANEL
    ]


def leave_out_concordance(annotations: pd.DataFrame) -> dict[str, object]:
    """Correlate the held-out annotators against the remaining panel."""
    selected = annotations[annotations["keyword_set"] == "SELECTED"]
    held = selected[selected["annotator"].isin(HELD_OUT)]
    rest = selected[~selected["annotator"].isin(HELD_OUT)]
    if held.empty or rest.empty:
        return {"error": "not enough annotators for a leave-out check"}
    proxy = held.rename(columns={"annotator": "expert"})[
        ["expert", "keyword", "novelty", "label"]
    ]
    report = expert_llm_concordance(rest, proxy)
    report["held_out_annotators"] = sorted(held["annotator"].unique())
    report["panel_annotators"] = sorted(rest["annotator"].unique())
    report["note"] = (
        "model-versus-model leave-out check; not a human expert study"
    )
    return report


def validate(name: str, top_n: int = 50, max_all: int | None = None) -> dict[str, object]:
    print(f"[{name}]", flush=True)
    corpus = load_csv(f"corpora/{name}.csv")
    problem = PROBLEMS[name]
    vectors = embed_corpus(name, corpus, problem)
    config = build_config(problem, vectors, top_n)

    out = Path("results") / name
    out.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}

    # Both modes analyse the same corpus, so they share one control set.
    # Annotating it once keeps the baseline identical across modes.
    reference = articles_first(corpus, config)
    _, control = selected_and_all_keywords(reference, max_all)
    print(f"  control set: {len(control)} author keywords", flush=True)
    control_annotations: list = []
    control_failures: dict[str, str] = {}
    for annotator in build_panel():
        try:
            control_annotations.extend(
                annotator.annotate(control, problem, DOMAINS[name], "ALL")
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not silenced
            control_failures[f"{annotator.name}/ALL"] = f"{type(exc).__name__}: {exc}"
    print(f"  control annotations: {len(control_annotations)}"
          f"{' failures: ' + json.dumps(control_failures) if control_failures else ''}",
          flush=True)

    for mode, runner in (("articles_first", articles_first), ("keywords_first", keywords_first)):
        result = reference if mode == "articles_first" else runner(corpus, config)
        selected, _ = selected_and_all_keywords(result, max_all)
        print(f"  {mode}: {len(selected)} selected vs {len(control)} control keywords", flush=True)

        report = validate_gaps(
            result, build_panel(), problem, DOMAINS[name],
            selected=selected, all_keywords=control,
            all_annotations=control_annotations,
        )
        report.failures.update(control_failures)
        report.save(out, prefix=f"{mode}_validation")
        plot_validation_comparison(report, out / f"{mode}_fig_validation.png")

        comparison = report.comparison
        summary: dict[str, object] = {
            "annotators": report.annotators,
            "failures": report.failures,
            "n_annotations": int(len(report.annotations)),
            "comparison": comparison.to_dict() if comparison else None,
            "reliability": report.reliability,
            "agreement": report.agreement.to_dict(orient="records"),
            "leave_out": leave_out_concordance(report.annotations),
        }
        summaries[mode] = summary
        if comparison:
            print(
                f"    GAP rate {comparison.selected_rates['GAP']:.3f} vs "
                f"{comparison.all_rates['GAP']:.3f} "
                f"({comparison.gap_rate_delta_pp:+.1f} pp); "
                f"novelty {comparison.selected_novelty['mean']:.2f} vs "
                f"{comparison.all_novelty['mean']:.2f} "
                f"({comparison.novelty_increase_pct:+.1f}%); "
                f"Welch p={comparison.novelty_test.get('welch_p'):.2e}",
                flush=True,
            )
        print(f"    reliability: {json.dumps(report.reliability)}", flush=True)
        if report.failures:
            print(f"    failures: {json.dumps(report.failures)}", flush=True)

        if mode == "articles_first":
            sheet = build_expert_sheet(result, problem)
            write_expert_sheet(
                sheet, out / "expert_sheet.xlsx", problem, ("expert_1", "expert_2")
            )
            sheet.to_csv(out / "expert_sheet.csv", index=False)

    (out / "validation_summary.json").write_text(
        json.dumps(summaries, indent=2, default=str), encoding="utf-8"
    )
    return summaries


if __name__ == "__main__":
    for corpus_name in sys.argv[1:] or ["management", "finance", "gamification"]:
        validate(corpus_name)
