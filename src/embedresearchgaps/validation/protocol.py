"""The annotation protocol used to validate candidate gaps.

The label set and the 1-10 Novelty Score reproduce the evaluation framework
of the PACIS/AMCIS papers verbatim, so that results obtained with this
package are comparable with the published figures.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "LABELS",
    "Annotation",
    "build_prompt",
    "SYSTEM_PROMPT",
    "RUBRIC",
    "parse_annotation_payload",
]

#: Admissible classification labels.
LABELS: tuple[str, ...] = ("GAP", "EXPLORED", "LOW IMPORTANCE", "METHOD", "NOISE")

SYSTEM_PROMPT = (
    "You are an experienced reviewer of systematic literature reviews. "
    "You classify candidate research-gap keywords strictly according to the "
    "rubric you are given and you answer with JSON only."
)

RUBRIC = """\
Novelty Score (1-10)
  1-2  The keyword, in the context of the research problem, points to an area
       that is already well described in the literature, OR creates no
       scientific value because it refers to technical aspects of an article
       (methodology, data format, software), OR points to an area that,
       although not well described, carries no significant research value.
  3-8  The area is described in the literature but not extensively; several
       topics requiring more detailed investigation can be identified and the
       area possesses scientific value.
  9    Described in the scientific literature to a minimal extent.
  10   Described to a minimal extent and with very high scientific potential.

Classification
  GAP             Valid research topic with remaining research potential in
                  the context of the stated research problem (Novelty 3-10).
  EXPLORED        Well-known area, field label, or a pure lexical variant of a
                  dominant construct (Novelty 1-2).
  LOW IMPORTANCE  Term that creates no scientific value in the context of the
                  research problem, either theoretically vacuous or
                  practically irrelevant (Novelty 1-2).
  METHOD          Pure methodological or analytical term: research technique,
                  statistical method, data-collection approach, software tool
                  (Novelty 1-2).
  NOISE           Nonsensical term, typo, or term with no conceivable
                  connection to the research-problem domain (Novelty 1).
"""

TASK = (
    "Evaluate the given keywords in the context of the above research problem. "
    "Assess whether each keyword represents a genuine research gap, an "
    "already well-explored area, a methodological term, or noise. Base your "
    "evaluation on the current state of the scientific literature related to "
    "the research problem."
)


@dataclass
class Annotation:
    """One (annotator, keyword) judgement."""

    keyword: str
    label: str
    novelty: float
    annotator: str
    rationale: str = ""
    keyword_set: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label = str(self.label).strip().upper()
        if self.label not in LABELS:
            raise ValueError(f"label {self.label!r} is not one of {LABELS}")
        self.novelty = float(self.novelty)
        if not 1.0 <= self.novelty <= 10.0:
            raise ValueError(f"novelty score {self.novelty} outside [1, 10]")

    @property
    def is_gap(self) -> bool:
        return self.label == "GAP"


def build_prompt(
    keywords: Sequence[str],
    problem_description: str,
    domain: str | None = None,
) -> str:
    """Compose the annotation prompt for one batch of keywords."""
    if not keywords:
        raise ValueError("no keywords to annotate")
    domain_line = f"Research domain: {domain}\n" if domain else ""
    numbered = "\n".join(f"{i + 1}. {kw}" for i, kw in enumerate(keywords))
    return (
        f"{domain_line}Research problem:\n{problem_description}\n\n"
        f"{TASK}\n\n{RUBRIC}\n"
        f"Keywords to evaluate:\n{numbered}\n\n"
        "Return JSON only, with this exact shape and one entry per keyword, "
        "in the same order:\n"
        '{"annotations": [{"keyword": "<keyword>", "novelty": <1-10>, '
        '"label": "GAP|EXPLORED|LOW IMPORTANCE|METHOD|NOISE", '
        '"rationale": "<one sentence>"}]}'
    )


#: One annotation object inside a reply, matched independently of its siblings.
_ENTRY_PATTERN = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the annotation payload out of a model reply.

    Chat models occasionally emit a reply that is *almost* JSON: an unescaped
    quotation mark inside a rationale, a trailing comma, a stray newline in a
    string.  Rejecting the whole batch over one malformed entry would discard
    24 valid judgements to lose 1, so the parse falls back to salvaging
    entries one by one and keeps those that stand on their own.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1]
        if stripped.lstrip().lower().startswith("json"):
            stripped = stripped.lstrip()[4:]
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in model reply: {text[:200]!r}")
    body = stripped[start : end + 1]
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass

    salvaged: list[dict[str, Any]] = []
    for match in _ENTRY_PATTERN.finditer(body):
        fragment = match.group(0)
        for candidate in (fragment, re.sub(r",(\s*[}\]])", r"\1", fragment)):
            try:
                entry = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict) and "keyword" in entry:
                salvaged.append(entry)
            break
    if not salvaged:
        raise ValueError(
            f"model reply is not valid JSON and no entry could be salvaged: "
            f"{text[:200]!r}"
        )
    return {"annotations": salvaged}


def parse_annotation_payload(
    text: str,
    keywords: Sequence[str],
    annotator: str,
    keyword_set: str = "",
) -> list[Annotation]:
    """Parse a model reply into :class:`Annotation` objects.

    Entries are matched to ``keywords`` case-insensitively; keywords the model
    omitted are skipped rather than guessed, so a partial reply produces
    fewer annotations instead of silently wrong ones.
    """
    payload = _extract_json(text)
    entries = payload.get("annotations", payload.get("results", []))
    if not isinstance(entries, list):
        raise ValueError("expected 'annotations' to be a list")

    wanted = {kw.lower(): kw for kw in keywords}
    annotations: list[Annotation] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_keyword = str(entry.get("keyword", "")).strip()
        keyword = wanted.get(raw_keyword.lower())
        if keyword is None:
            continue
        label = str(entry.get("label", "")).strip().upper().replace("_", " ")
        if label not in LABELS:
            continue
        annotations.append(
            Annotation(
                keyword=keyword,
                label=label,
                novelty=float(entry.get("novelty", entry.get("novelty_score", 1))),
                annotator=annotator,
                rationale=str(entry.get("rationale", ""))[:500],
                keyword_set=keyword_set,
                raw=entry,
            )
        )
    return annotations
