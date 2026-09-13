"""Orchestration of a multi-annotator validation study."""

from __future__ import annotations

import random
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..results import PipelineResult
from .annotators import Annotator
from .metrics import (
    SetComparison,
    annotations_to_frame,
    compare_keyword_sets,
    fleiss_kappa,
    krippendorff_alpha_ordinal,
    label_rates,
    mean_novelty,
    pairwise_agreement,
    per_keyword_consensus,
)
from .protocol import Annotation

__all__ = ["ValidationReport", "validate_gaps", "selected_and_all_keywords"]


def selected_and_control_keywords(
    result: PipelineResult,
    max_control: int | None = None,
    random_state: int = 42,
    exclude_selected: bool = True,
) -> tuple[list[str], list[str]]:
    """Split a run into the candidate set and a disjoint control set.

    SELECTED are the keywords the pipeline flagged as candidate gaps (members
    of a conceptual combination are expanded into individual keywords).
    CONTROL is every *other* unique author keyword of the analysis corpus,
    ordered by corpus-level document frequency.

    The two sets must be disjoint for a two-sample comparison to mean
    anything.  Earlier versions of this function returned the whole keyword
    population as the control, which nests the candidate set inside it: the
    candidates then contribute to both arms, the two samples are not
    independent, and a Welch or Mann-Whitney statistic computed on them has no
    defined null distribution.  Pass ``exclude_selected=False`` only to
    reproduce such a nested comparison deliberately.

    ``max_control`` bounds the annotation cost on large corpora. The cap is
    applied by drawing a *random* sample with ``random_state``, not by taking
    the head of the ordered list: a head would be either all-frequent or --
    where document frequencies tie at one, as they do for most author keywords
    -- an arbitrary subset of rare terms, and either choice makes the control
    set unrepresentative of the keyword population the pipeline filtered.
    """
    selected: list[str] = []
    for gap in result.gap_objects:
        if gap.members:
            selected.extend(gap.members)
        else:
            selected.append(gap.keyword)
    selected = list(dict.fromkeys(selected))

    counts: Counter = Counter()
    for keywords in result.articles["keywords_normalized"]:
        # list(...) not the dict itself: Counter.update on a mapping would
        # add its values (all None) instead of counting its keys.
        counts.update(list(dict.fromkeys(keywords)))
    population = [keyword for keyword, _ in counts.most_common()]
    if exclude_selected:
        flagged = set(selected)
        control = [keyword for keyword in population if keyword not in flagged]
    else:
        control = list(population)
    if not control:
        return selected, selected
    if max_control is not None and len(control) > max_control:
        control = sorted(random.Random(random_state).sample(control, max_control))
    return selected, control


def selected_and_all_keywords(
    result: PipelineResult,
    max_all: int | None = None,
    random_state: int = 42,
) -> tuple[list[str], list[str]]:
    """Deprecated alias returning the *nested* control set.

    Kept so that a study run with version 1.0.0 can be reproduced exactly.
    New code should call :func:`selected_and_control_keywords`, whose control
    set excludes the candidates.
    """
    warnings.warn(
        "selected_and_all_keywords returns a control set that contains the "
        "candidates, so the two arms are not independent; use "
        "selected_and_control_keywords instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return selected_and_control_keywords(
        result, max_control=max_all, random_state=random_state, exclude_selected=False
    )


@dataclass
class ValidationReport:
    """Annotations and every derived metric of one validation study."""

    annotations: pd.DataFrame
    consensus: pd.DataFrame
    agreement: pd.DataFrame
    comparison: SetComparison | None
    reliability: dict[str, float] = field(default_factory=dict)
    per_set: dict[str, dict[str, Any]] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    problem_description: str = ""
    domain: str = ""

    @property
    def annotators(self) -> list[str]:
        return sorted(self.annotations["annotator"].unique()) if not self.annotations.empty else []

    def summary(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "annotators": self.annotators,
            "n_annotations": int(len(self.annotations)),
            "per_set": self.per_set,
            "reliability": self.reliability,
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "unanimous_gaps": (
                self.consensus[
                    (self.consensus["gap_share"] == 1.0)
                    & (self.consensus["keyword_set"] != "ALL")
                ]["keyword"].tolist()
                if not self.consensus.empty
                else []
            ),
            "failures": self.failures,
        }

    def save(self, directory: str | Path, prefix: str = "validation") -> dict[str, str]:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}
        for name, frame in (
            ("annotations", self.annotations),
            ("consensus", self.consensus),
            ("agreement", self.agreement),
        ):
            path = out / f"{prefix}_{name}.csv"
            frame.to_csv(path, index=False)
            written[name] = str(path)
        import json

        path = out / f"{prefix}_summary.json"
        path.write_text(
            json.dumps(self.summary(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        written["summary"] = str(path)
        return written


def validate_gaps(
    result: PipelineResult,
    annotators: Sequence[Annotator],
    problem_description: str,
    domain: str = "",
    max_all_keywords: int | None = 60,
    selected: Sequence[str] | None = None,
    all_keywords: Sequence[str] | None = None,
    all_annotations: Sequence[Annotation] | None = None,
    selected_annotations: Sequence[Annotation] | None = None,
) -> ValidationReport:
    """Annotate the SELECTED and ALL keyword sets with every annotator.

    An annotator that raises (missing credential, transport error, malformed
    reply) is recorded in ``ValidationReport.failures`` and skipped; the study
    continues with the remaining annotators rather than aborting, and the
    report states explicitly which models contributed.

    Pass ``all_annotations`` to reuse judgements of the control set that were
    obtained earlier. Both modes of a run share one analysis corpus and hence
    one control set, so reusing it halves the annotation cost and makes the
    two modes comparable against an identical baseline.
    ``selected_annotations`` does the same for the candidate set, which lets a
    study that lost one annotator to a transport error be completed by
    re-running that annotator alone: supply the judgements already on disk,
    pass only the missing annotator in ``annotators``, and the report is
    recomputed over the union.
    """
    if not annotators and not (all_annotations and selected_annotations):
        raise ValueError(
            "no annotators supplied: pass at least one annotator, or supply "
            "both all_annotations and selected_annotations to recompute a "
            "report from judgements already collected"
        )
    if selected is None or all_keywords is None:
        auto_selected, auto_all = selected_and_all_keywords(result, max_all_keywords)
        selected = list(selected) if selected is not None else auto_selected
        all_keywords = list(all_keywords) if all_keywords is not None else auto_all
    if not selected:
        raise ValueError("the pipeline produced no candidate gaps to validate")

    collected: list[Annotation] = list(all_annotations or []) + list(
        selected_annotations or []
    )
    pending: list[tuple[str, Sequence[str]]] = []
    if not selected_annotations:
        pending.append(("SELECTED", selected))
    if not all_annotations:
        pending.append(("ALL", all_keywords))
    sets_to_annotate: tuple[tuple[str, Sequence[str]], ...] = tuple(pending)
    failures: dict[str, str] = {}
    for annotator in annotators:
        for keyword_set, keywords in sets_to_annotate:
            try:
                collected.extend(
                    annotator.annotate(
                        keywords, problem_description, domain or None, keyword_set
                    )
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not silenced
                failures[f"{annotator.name}/{keyword_set}"] = f"{type(exc).__name__}: {exc}"

    frame = annotations_to_frame(collected)
    if frame.empty:
        raise RuntimeError(f"every annotator failed: {failures}")

    selected_frame = frame[frame["keyword_set"] == "SELECTED"]
    all_frame = frame[frame["keyword_set"] == "ALL"]
    per_set = {
        name: {"label_rates": label_rates(subset), "novelty": mean_novelty(subset)}
        for name, subset in (("SELECTED", selected_frame), ("ALL", all_frame))
        if not subset.empty
    }
    comparison = (
        compare_keyword_sets(selected_frame, all_frame)
        if not selected_frame.empty and not all_frame.empty
        else None
    )
    reliability = {
        "fleiss_kappa_selected": fleiss_kappa(selected_frame),
        "krippendorff_alpha_selected": krippendorff_alpha_ordinal(selected_frame),
    }

    return ValidationReport(
        annotations=frame,
        consensus=per_keyword_consensus(frame),
        agreement=pairwise_agreement(selected_frame),
        comparison=comparison,
        reliability=reliability,
        per_set=per_set,
        failures=failures,
        problem_description=problem_description,
        domain=domain,
    )
