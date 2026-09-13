"""Expert validation: assessment sheets and concordance with the LLM panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..results import PipelineResult
from .metrics import expert_llm_concordance
from .protocol import LABELS, RUBRIC

__all__ = [
    "build_expert_sheet",
    "write_expert_sheet",
    "load_expert_sheet",
    "expert_report",
]

INSTRUCTIONS = (
    "For every keyword below, read the research problem, then fill in two "
    "columns: 'novelty' (integer 1-10) and 'label' (one of "
    f"{', '.join(LABELS)}). Leave 'comment' empty unless you want to justify "
    "a judgement. Work independently; do not consult the other reviewers or "
    "the automatically generated scores, which are deliberately omitted from "
    "this sheet."
)


def build_expert_sheet(
    result: PipelineResult,
    problem_description: str,
    include_scores: bool = False,
    shuffle_seed: int | None = 42,
) -> pd.DataFrame:
    """Blinded assessment sheet for one expert.

    Keywords of conceptual combinations are presented as the pair, exactly as
    the pipeline reports them.  Pipeline scores, gap types and cluster labels
    are withheld by default so that the expert judgement stays independent of
    the model output; rows are shuffled for the same reason.
    """
    if result.gaps.empty:
        raise ValueError("the run produced no candidate gaps to review")
    columns = ["keyword"]
    if include_scores:
        columns += ["gap_type", "cluster_label", "score"]
    sheet = result.gaps[columns].drop_duplicates("keyword").copy()
    if shuffle_seed is not None:
        sheet = sheet.sample(frac=1.0, random_state=shuffle_seed)
    sheet = sheet.reset_index(drop=True)
    sheet.insert(0, "item", range(1, len(sheet) + 1))
    sheet["novelty"] = ""
    sheet["label"] = ""
    sheet["comment"] = ""
    sheet.attrs["problem_description"] = problem_description
    return sheet


def write_expert_sheet(
    sheet: pd.DataFrame,
    path: str | Path,
    problem_description: str,
    experts: Sequence[str] = ("expert_1", "expert_2"),
) -> str:
    """Write an ``.xlsx`` workbook with the rubric and one tab per expert."""
    destination = Path(path)
    header = pd.DataFrame(
        {
            "section": ["Research problem", "Instructions", "Rubric"],
            "content": [problem_description, INSTRUCTIONS, RUBRIC],
        }
    )
    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        header.to_excel(writer, sheet_name="instructions", index=False)
        for expert in experts:
            sheet.assign(expert=expert).to_excel(writer, sheet_name=expert[:31], index=False)
    return str(destination)


def load_expert_sheet(path: str | Path, expert: str | None = None) -> pd.DataFrame:
    """Read completed expert sheets into the long format the metrics expect.

    Accepts a CSV or an ``.xlsx`` workbook whose tabs are experts.  Rows with
    an empty ``novelty`` cell are dropped, so a partially completed sheet
    contributes only the items the expert actually rated.
    """
    source = Path(path)
    if source.suffix.lower() in {".xlsx", ".xlsm"}:
        book = pd.read_excel(source, sheet_name=None)
        frames = [
            frame.assign(expert=frame.get("expert", pd.Series([name] * len(frame))))
            for name, frame in book.items()
            if name != "instructions" and "keyword" in frame.columns
        ]
        if not frames:
            raise ValueError(f"no expert tabs with a 'keyword' column in {source}")
        frame = pd.concat(frames, ignore_index=True)
    else:
        frame = pd.read_csv(source)
        if "expert" not in frame.columns:
            frame["expert"] = expert or source.stem

    if "novelty" not in frame.columns:
        raise ValueError("completed sheet must contain a 'novelty' column")
    frame = frame[frame["novelty"].notna() & (frame["novelty"].astype(str).str.strip() != "")]
    frame["novelty"] = pd.to_numeric(frame["novelty"], errors="coerce")
    frame = frame[frame["novelty"].between(1, 10)].copy()
    if "label" in frame.columns:
        frame["label"] = frame["label"].astype(str).str.strip().str.upper()
        frame.loc[~frame["label"].isin(LABELS), "label"] = pd.NA
    return frame[[c for c in ("expert", "keyword", "novelty", "label", "comment") if c in frame.columns]]


def expert_report(
    llm_annotations: pd.DataFrame,
    expert_frame: pd.DataFrame,
    keyword_set: str = "SELECTED",
) -> dict[str, Any]:
    """Concordance between experts and the LLM panel on one keyword set."""
    subset = (
        llm_annotations[llm_annotations["keyword_set"] == keyword_set]
        if "keyword_set" in llm_annotations.columns and keyword_set
        else llm_annotations
    )
    return expert_llm_concordance(subset, expert_frame)
