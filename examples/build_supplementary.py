"""Assemble the supplementary material for the SoftwareX submission.

Produces one workbook with every table the manuscript refers to but cannot
contain, and a markdown companion that states the parameters, the formulae and
the provenance of each sheet.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "embedresearchgaps/src")
sys.path.append(os.getcwd())

OUT = Path("manuscript")
OUT.mkdir(exist_ok=True)
CASES = {"management": "Management", "finance": "Economics & finance",
         "gamification": "Gamification (conformance)"}

GAP_COLUMNS = [
    "gap_type", "keyword_display", "cluster_label", "score", "frequency",
    "n_articles", "citations", "rationale", "article_titles", "article_dois",
]


def gap_sheet(name: str, mode: str) -> pd.DataFrame:
    frame = pd.read_csv(f"results/{name}/{mode}_gaps.csv")
    columns = [c for c in GAP_COLUMNS if c in frame.columns]
    metrics = [c for c in frame.columns if c.startswith("metric_")]
    return frame[columns + metrics].sort_values(
        ["gap_type", "score"], ascending=[True, False], ignore_index=True
    )


def validation_sheet(name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    payload = json.loads(Path(f"results/{name}/validation_summary.json").read_text())
    for mode, entry in payload.items():
        comparison = entry.get("comparison") or {}
        test = comparison.get("novelty_test", {})
        rows.append({
            "corpus": CASES[name],
            "mode": mode,
            "n_candidates": comparison.get("n_selected_keywords"),
            "n_control": comparison.get("n_all_keywords"),
            "annotators": len(entry.get("annotators", [])),
            "n_judgements": entry.get("n_annotations"),
            "gap_rate_candidates": comparison.get("gap_rate_selected"),
            "gap_rate_control": comparison.get("gap_rate_all"),
            "gap_rate_delta_pp": comparison.get("gap_rate_delta_pp"),
            "novelty_candidates": comparison.get("mean_novelty_selected"),
            "novelty_control": comparison.get("mean_novelty_all"),
            "novelty_change_pct": comparison.get("novelty_increase_pct"),
            "welch_t": test.get("welch_t"),
            "welch_p": test.get("welch_p"),
            "mannwhitney_p": test.get("mannwhitney_p"),
            "cohens_d": test.get("cohens_d"),
            "fleiss_kappa": entry.get("reliability", {}).get("fleiss_kappa_selected"),
            "krippendorff_alpha": entry.get("reliability", {}).get(
                "krippendorff_alpha_selected"
            ),
            "leaveout_spearman_rho": (entry.get("leave_out") or {}).get("spearman_rho"),
            "leaveout_spearman_p": (entry.get("leave_out") or {}).get("spearman_p"),
            "leaveout_label_agreement": (entry.get("leave_out") or {}).get(
                "label_agreement"
            ),
            "annotator_failures": json.dumps(entry.get("failures") or {}),
        })
    return pd.DataFrame(rows)


def matched_sheet() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in ("management", "finance"):
        path = Path(f"results/{name}/matched_arm_summary.json")
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for contrast in ("selected_vs_eligible", "eligible_vs_all"):
            entry = payload[contrast]
            test = entry.get("novelty_test", {})
            rows.append({
                "corpus": CASES[name],
                "contrast": contrast.replace("_", " "),
                "n_eligible_pool": payload["n_eligible_pool"],
                "n_eligible_annotated": payload["n_eligible_annotated"],
                "gap_rate_left": entry.get("gap_rate_selected"),
                "gap_rate_right": entry.get("gap_rate_all"),
                "gap_rate_delta_pp": entry.get("gap_rate_delta_pp"),
                "novelty_left": entry.get("mean_novelty_selected"),
                "novelty_right": entry.get("mean_novelty_all"),
                "novelty_change_pct": entry.get("novelty_increase_pct"),
                "welch_p": test.get("welch_p"),
                "mannwhitney_p": test.get("mannwhitney_p"),
                "cohens_d": test.get("cohens_d"),
            })
    return pd.DataFrame(rows)


def diagnostics_sheet() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in CASES:
        payload = json.loads(Path(f"results/{name}/case_summary.json").read_text())
        for mode in ("articles_first", "keywords_first"):
            entry = payload[mode]
            diagnostics = entry["diagnostics"]
            quality = diagnostics.get("cluster_quality", {})
            selection = diagnostics.get("k_selection") or {}
            rows.append({
                "corpus": CASES[name],
                "mode": mode,
                "n_articles": entry["n_articles"],
                "n_keywords": entry["n_keywords"],
                "k": entry["n_clusters"],
                "k_rule": selection.get("rule"),
                "k_elbow": selection.get("k_elbow"),
                "k_silhouette": selection.get("k_silhouette"),
                "silhouette": quality.get("silhouette"),
                "davies_bouldin": quality.get("davies_bouldin"),
                "calinski_harabasz": quality.get("calinski_harabasz"),
                "n_candidates": entry["n_gaps"],
                "gap_types": json.dumps(entry["gap_types"]),
                "cluster_sizes": json.dumps(diagnostics.get("cluster_sizes")),
                "frequency_threshold": diagnostics.get("frequency_threshold"),
                "peripheral_share": diagnostics.get("peripheral_share"),
                "n_core_domain_terms": diagnostics.get("n_core_domain_terms"),
                "core_domain_terms": ", ".join(diagnostics.get("core_domain_terms", [])),
                "gaps_before_dedup": diagnostics.get("gaps_before_deduplication"),
                "gaps_after_dedup": diagnostics.get("gaps_after_deduplication"),
                "tfidf_match_types": json.dumps(
                    diagnostics.get("tfidf_match_statistics")
                ),
                "article_encoder": json.dumps(diagnostics.get("article_encoder")),
            })
    return pd.DataFrame(rows)


def corpus_sheet() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in CASES:
        frame = pd.read_csv(f"corpora/{name}.csv")
        years = pd.to_numeric(frame["Year"], errors="coerce")
        citations = pd.to_numeric(frame["Cited by"], errors="coerce")
        rows.append({
            "corpus": CASES[name],
            "file": f"corpora/{name}.csv",
            "n_records": len(frame),
            "year_min": int(years.min()),
            "year_max": int(years.max()),
            "median_year": float(years.median()),
            "median_citations": float(citations.median()),
            "mean_citations": round(float(citations.mean()), 2),
            "n_with_abstract": int((frame["Abstract"].astype(str).str.len() > 40).sum()),
            "mean_keywords_per_record": round(
                float(frame["Author Keywords"].astype(str).str.count(";").add(1).mean()), 2
            ),
        })
    return pd.DataFrame(rows)


def main() -> dict[str, str]:
    sheets: dict[str, pd.DataFrame] = {
        "S1_corpora": corpus_sheet(),
        "S2_diagnostics": diagnostics_sheet(),
        "S3_validation": pd.concat([validation_sheet(n) for n in CASES], ignore_index=True),
        "S5_size_sensitivity": pd.read_csv(
            "results/sensitivity/corpus_size_sensitivity.csv"
        ).drop(columns=["candidates"], errors="ignore"),
        "S6_seed_stability": pd.read_csv("results/sensitivity/seed_stability.csv"),
    }
    matched = matched_sheet()
    if not matched.empty:
        sheets["S4_matched_arm"] = matched
    for name, pretty in CASES.items():
        for mode, tag in (("articles_first", "B"), ("keywords_first", "A")):
            sheets[f"S7_{name[:6]}_mode{tag}"] = gap_sheet(name, mode)

    ordered = dict(sorted(sheets.items()))
    path = OUT / "supplementary_material.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in ordered.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"{path}: {len(ordered)} sheets", flush=True)
    for name, frame in ordered.items():
        print(f"  {name}: {frame.shape[0]} rows x {frame.shape[1]} cols", flush=True)
    return {"workbook": str(path)}


if __name__ == "__main__":
    main()
