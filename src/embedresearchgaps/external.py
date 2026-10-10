"""Load a candidate ranking produced somewhere else.

The assessment, control-set and consensus machinery of this package does not
depend on how a candidate list was produced: it needs a set of keywords, a
corpus to draw a control from, and -- for the comparison tables -- a score per
candidate.  Until now the only way to obtain that input was to run one of the
two built-in modes, which made the package unable to assess a ranking from
VOSviewer, BERTopic, a competing implementation, or a human reviewer, and
unable to say whether its own ranking beats one of those on the same corpus.

:func:`load_external_ranking` closes that gap.  It reads a ranking from a CSV
or Excel export, a :class:`pandas.DataFrame`, or a bare sequence of keywords,
and returns the same :class:`~embedresearchgaps.results.PipelineResult` the
built-in modes return, with ``mode='external'``.  Everything downstream then
works unchanged::

    external = load_external_ranking("vosviewer_top50.csv", corpus=corpus)
    report = validate_gaps(external, annotators, problem_description=problem)
    selected, control = selected_and_control_keywords(external)
    overlap = jaccard(set(external.gaps["keyword"]), set(ours.gaps["keyword"]))

Two properties are worth stating because they bound what the comparison can
mean.  First, an external ranking carries no cluster structure of ours, so
``cluster`` is -1 and the cluster-dependent diagnostics are absent; metrics
that need a partition (silhouette, centroid similarity) are not computed and
are not faked.  Second, when a corpus is supplied the loader attaches *corpus*
evidence -- document frequency, source records, citations -- to each candidate,
which is what makes a like-for-like comparison with a built-in run possible;
without a corpus the candidates carry only what the file provided, and the
control-set functions have no population to draw from.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import RunConfig
from .corpus import Corpus
from .gaps import Gap, gaps_to_frame
from .results import PipelineResult
from .text import normalize_keyword

__all__ = ["load_external_ranking", "EXTERNAL_GAP_TYPE"]

#: ``gap_type`` assigned to candidates read from an external ranking that does
#: not name a type of its own.  It is deliberately *not* a member of
#: ``GAP_TYPES``: the four built-in types are defined by detectors in this
#: package, and labelling a foreign candidate with one of them would assert a
#: provenance the file does not support.
EXTERNAL_GAP_TYPE = "External Candidate"

#: Accepted spellings for the keyword column, in priority order.
_KEYWORD_ALIASES = ("keyword", "term", "label", "item", "candidate", "name")
#: Accepted spellings for the score column, in priority order.
_SCORE_ALIASES = ("score", "weight", "relevance", "value", "novelty", "strength")
#: Accepted spellings for an explicit rank column.
_RANK_ALIASES = ("rank", "position", "order", "place")
#: Accepted spellings for a type/category column.
_TYPE_ALIASES = ("gap_type", "type", "category", "class")


def _resolve(frame: pd.DataFrame, aliases: Sequence[str], explicit: str | None) -> str | None:
    """Pick a column by explicit name or by case-insensitive alias."""
    if explicit is not None:
        if explicit not in frame.columns:
            raise KeyError(
                f"column {explicit!r} is not in the ranking "
                f"(columns: {list(frame.columns)})"
            )
        return explicit
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    # Second pass on whole words inside compound headers, so that real exports
    # resolve without a column map: VOSviewer writes "Total link strength",
    # a spreadsheet of expert ratings writes "Mean novelty score".
    for alias in aliases:
        for name, original in lowered.items():
            if alias in re.split(r"[^a-z0-9]+", name):
                return original
    return None


def _read(source: Any) -> pd.DataFrame:
    """Normalise the accepted input shapes into a frame with a keyword column."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"ranking file not found: {path}")
        if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
            return pd.read_excel(path)
        if path.suffix.lower() in {".tsv", ".tab"}:
            return pd.read_csv(path, sep="\t")
        return pd.read_csv(path)
    if isinstance(source, Iterable):
        values = [str(x) for x in source]
        if not values:
            raise ValueError("the ranking is empty")
        return pd.DataFrame({"keyword": values})
    raise TypeError(
        "source must be a DataFrame, a path to a CSV/TSV/Excel export, "
        f"or a sequence of keywords, not {type(source).__name__}"
    )


def _corpus_evidence(corpus: Corpus | None) -> tuple[Counter, dict[str, dict[str, Any]]]:
    """Document frequency and per-keyword source records for the corpus."""
    if corpus is None:
        return Counter(), {}
    frame = corpus.frame
    counts: Counter = Counter()
    evidence: dict[str, dict[str, Any]] = {}
    has_citations = "citations" in frame.columns
    for row in frame.itertuples(index=False):
        keywords = list(dict.fromkeys(getattr(row, "keywords_normalized", []) or []))
        counts.update(keywords)
        citation = getattr(row, "citations", np.nan) if has_citations else np.nan
        for keyword in keywords:
            slot = evidence.setdefault(
                keyword, {"titles": [], "dois": [], "citations": []}
            )
            slot["titles"].append(str(getattr(row, "title", "")))
            slot["dois"].append(str(getattr(row, "doi", "")))
            if citation is not None and not pd.isna(citation):
                slot["citations"].append(float(citation))
    return counts, evidence


def load_external_ranking(
    source: Any,
    corpus: Corpus | None = None,
    *,
    keyword_column: str | None = None,
    score_column: str | None = None,
    rank_column: str | None = None,
    type_column: str | None = None,
    top_n: int | None = None,
    config: RunConfig | None = None,
    name: str = "external",
    normalize: bool = True,
) -> PipelineResult:
    """Read a ranking produced outside this package.

    Parameters
    ----------
    source:
        A path to a CSV, TSV or Excel export, a :class:`pandas.DataFrame`, or a
        plain sequence of keywords in rank order.
    corpus:
        The corpus the ranking refers to.  Supplying it attaches document
        frequency, source titles, DOIs and citation counts to every candidate
        and gives :func:`selected_and_control_keywords` a population to draw a
        control set from.  Without it the result still validates and still
        compares against another ranking, but no control set can be built.
    keyword_column, score_column, rank_column, type_column:
        Column names.  Each is resolved from a short alias table when omitted,
        so ordinary exports load without a column map.
    top_n:
        Keep only the first ``top_n`` rows after ordering.  Ordering follows
        the rank column when present, then the score column in descending
        order, and otherwise the order of the file.
    config:
        Recorded on the result so a deposited external run documents its own
        settings.  A default :class:`RunConfig` is used when omitted.
    name:
        Label for this ranking, written to ``diagnostics['external']['name']``
        and used in comparison tables.
    normalize:
        Apply :func:`~embedresearchgaps.text.normalize_keyword` to every
        keyword, which is required for the keywords to match corpus evidence.
        Disable only when the ranking is already normalised.

    Returns
    -------
    PipelineResult
        ``mode='external'``.  ``cluster`` is -1 for every candidate and no
        partition diagnostics are produced: an external ranking carries no
        clustering of ours, and inventing one would misreport its provenance.

    Raises
    ------
    ValueError
        When the ranking is empty, or when no keyword column can be resolved.
    """
    frame = _read(source)
    if frame.empty:
        raise ValueError("the ranking is empty")

    keyword_col = _resolve(frame, _KEYWORD_ALIASES, keyword_column)
    if keyword_col is None:
        raise ValueError(
            "no keyword column found; pass keyword_column= explicitly "
            f"(columns: {list(frame.columns)})"
        )
    score_col = _resolve(frame, _SCORE_ALIASES, score_column)
    rank_col = _resolve(frame, _RANK_ALIASES, rank_column)
    type_col = _resolve(frame, _TYPE_ALIASES, type_column)

    frame = frame[frame[keyword_col].notna()].copy()
    frame[keyword_col] = frame[keyword_col].astype(str).str.strip()
    frame = frame[frame[keyword_col] != ""]
    if frame.empty:
        raise ValueError("the ranking contains no non-empty keyword")

    if rank_col is not None:
        frame = frame.sort_values(rank_col, kind="mergesort")
    elif score_col is not None:
        frame = frame.sort_values(score_col, ascending=False, kind="mergesort")
    if top_n is not None:
        frame = frame.head(int(top_n))

    counts, evidence = _corpus_evidence(corpus)

    # Deduplicate first, so that the rank-derived score below runs over a
    # contiguous scale: dropping a duplicate must not leave a hole in it.
    # Positional access rather than itertuples, because itertuples renames
    # columns that are not valid identifiers and an exported ranking routinely
    # has headers like "Avg. citations" or "Total link strength".
    ordered: list[tuple[int, str]] = []
    seen: set[str] = set()
    duplicates = 0
    for position in range(1, len(frame) + 1):
        raw = str(frame[keyword_col].iloc[position - 1])
        keyword = normalize_keyword(raw) if normalize else raw
        if not keyword:
            continue
        if keyword in seen:
            duplicates += 1
            continue
        seen.add(keyword)
        ordered.append((position, keyword))

    gaps: list[Gap] = []
    unmatched: list[str] = []
    for index, (position, keyword) in enumerate(ordered, start=1):
        record = evidence.get(keyword)
        if corpus is not None and record is None:
            unmatched.append(keyword)
        if score_col is not None:
            value = frame[score_col].iloc[position - 1]
            score = float(value) if not pd.isna(value) else float("nan")
        else:
            # Without a score, rank order is the only information the file
            # carries: map it to a descending scale so that higher is better,
            # as it is for the built-in detectors.  The position in the file is
            # preserved separately in ``metrics``.
            score = 1.0 - (index - 1) / max(len(ordered), 1)
        gap_type = EXTERNAL_GAP_TYPE
        if type_col is not None:
            given = frame[type_col].iloc[position - 1]
            if not pd.isna(given) and str(given).strip():
                gap_type = str(given).strip()
        citations = (
            max(record["citations"]) if record and record["citations"] else None
        )
        gaps.append(
            Gap(
                gap_type=gap_type,
                keyword=keyword,
                cluster=-1,
                score=score,
                frequency=int(counts.get(keyword, 0)),
                rationale=f"rank {position} in external ranking {name!r}",
                metrics={"external_rank": float(position)},
                n_articles=len(record["titles"]) if record else 0,
                article_titles=tuple(record["titles"][:5]) if record else (),
                article_dois=tuple(record["dois"][:5]) if record else (),
                citations=citations,
                cluster_label=None,
            )
        )

    if not gaps:
        raise ValueError("the ranking produced no candidates")

    if corpus is not None:
        articles = corpus.frame.copy()
        keywords_frame = pd.DataFrame(
            {
                "keyword": list(counts.keys()),
                "frequency": list(counts.values()),
                "cluster": -1,
            }
        ).sort_values("frequency", ascending=False, ignore_index=True)
    else:
        articles = pd.DataFrame(columns=["title", "doi", "keywords_normalized"])
        keywords_frame = pd.DataFrame(
            {"keyword": [gap.keyword for gap in gaps], "frequency": 0, "cluster": -1}
        )

    return PipelineResult(
        mode="external",
        config=config if config is not None else RunConfig(),
        articles=articles,
        keywords=keywords_frame,
        gaps=gaps_to_frame(gaps),
        gap_objects=gaps,
        cluster_labels={},
        centroid_similarity=None,
        diagnostics={
            "external": {
                "name": name,
                "source": str(source) if isinstance(source, (str, Path)) else type(source).__name__,
                "rows_read": int(len(frame)),
                "candidates": len(gaps),
                "duplicates_dropped": duplicates,
                "keyword_column": keyword_col,
                "score_column": score_col,
                "rank_column": rank_col,
                "type_column": type_col,
                "normalized": bool(normalize),
                "corpus_attached": corpus is not None,
                "unmatched_in_corpus": unmatched,
            }
        },
    )
