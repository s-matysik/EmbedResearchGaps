"""Validation of candidate gaps by LLM annotator panels and human experts."""

from __future__ import annotations

from .annotators import (
    Annotator,
    CallableAnnotator,
    ChatAnnotator,
    available_providers,
    get_annotator,
)
from .experts import (
    build_expert_sheet,
    expert_report,
    load_expert_sheet,
    write_expert_sheet,
)
from .metrics import (
    SetComparison,
    annotations_to_frame,
    compare_keyword_sets,
    permutation_mean_difference,
    expert_llm_concordance,
    fleiss_kappa,
    krippendorff_alpha_ordinal,
    label_rates,
    mean_novelty,
    pairwise_agreement,
    per_keyword_consensus,
)
from .protocol import LABELS, RUBRIC, Annotation, build_prompt, parse_annotation_payload
from .runner import (
    ValidationReport,
    selected_and_all_keywords,
    selected_and_control_keywords,
    validate_gaps,
)

__all__ = [
    "Annotator", "ChatAnnotator", "CallableAnnotator", "get_annotator", "available_providers",
    "Annotation", "LABELS", "RUBRIC", "build_prompt", "parse_annotation_payload",
    "validate_gaps", "ValidationReport", "selected_and_all_keywords",
    "selected_and_control_keywords",
    "annotations_to_frame", "label_rates", "mean_novelty", "per_keyword_consensus",
    "fleiss_kappa", "krippendorff_alpha_ordinal", "pairwise_agreement",
    "compare_keyword_sets", "SetComparison", "expert_llm_concordance",
    "permutation_mean_difference",
    "build_expert_sheet", "write_expert_sheet", "load_expert_sheet", "expert_report",
]
