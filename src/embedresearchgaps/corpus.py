"""Loading and normalising bibliographic corpora."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .text import normalize_keyword, split_author_keywords

__all__ = ["Corpus", "load_csv", "from_dataframe", "COLUMN_ALIASES"]

#: Canonical column name -> accepted source spellings (case-insensitive).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "title": ("title", "document title", "ti", "dc:title"),
    "abstract": ("abstract", "ab", "dc:description"),
    "author_keywords": (
        "author keywords", "authkeywords", "keywords", "author_keywords",
        "de", "id", "author-keywords",
    ),
    "year": ("year", "publication year", "py", "prism:coverdate", "cover date"),
    "citations": (
        "cited by", "citations", "times cited", "tc", "citedby-count",
        "cited-by", "citation count",
    ),
    "doi": ("doi", "di", "prism:doi"),
    "authors": ("authors", "author full names", "af", "au", "dc:creator"),
    "source": ("source title", "journal", "so", "prism:publicationname", "publication name"),
    "eid": ("eid", "scopus id", "uid", "id"),
}

_REQUIRED = ("title", "author_keywords")


def _resolve_columns(columns: Sequence[str], mapping: Mapping[str, str] | None) -> dict[str, str]:
    """Map canonical field names onto the actual column names of a frame."""
    lowered = {str(col).strip().lower(): str(col) for col in columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                resolved[canonical] = lowered[alias]
                break
    if mapping:
        for canonical, source in mapping.items():
            if source is not None:
                resolved[canonical] = source
    return resolved


@dataclass
class Corpus:
    """A normalised bibliographic corpus.

    The frame always carries the columns ``title``, ``abstract``,
    ``author_keywords`` (list of strings), ``keywords_normalized`` (list of
    normalised strings), ``combined_text``, ``year``, ``citations``, ``doi``,
    ``authors`` and ``source``; missing source fields are filled with empty
    values so that downstream code never has to branch on availability.
    """

    frame: pd.DataFrame
    source_path: str | None = None
    column_map: dict[str, str] = field(default_factory=dict)
    dropped_empty_text: int = 0
    dropped_empty_keywords: int = 0
    _spelling: dict[str, str] | None = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def n_articles(self) -> int:
        return len(self.frame)

    @property
    def keyword_lists(self) -> list[list[str]]:
        return list(self.frame["keywords_normalized"])

    @property
    def spelling(self) -> dict[str, str]:
        """Normalised keyword -> its most frequent spelling in the source data.

        Author keywords are normalised to lower case for matching, which
        destroys acronyms (``ESG``, ``SME``, ``AI``).  This map restores the
        spelling the authors actually used, so cluster labels and reported
        candidates read as they do in the literature.
        """
        if self._spelling is None:
            variants: dict[str, Counter] = {}
            for raw_list in self.frame["author_keywords"]:
                for raw in raw_list:
                    normalized = normalize_keyword(raw)
                    if not normalized:
                        continue
                    variants.setdefault(normalized, Counter())[" ".join(str(raw).split())] += 1
            self._spelling = {
                normalized: counter.most_common(1)[0][0]
                for normalized, counter in variants.items()
            }
        return self._spelling

    @property
    def has_abstracts(self) -> bool:
        return bool((self.frame["abstract"].astype(str).str.strip() != "").any())

    @property
    def has_citations(self) -> bool:
        return bool(self.frame["citations"].notna().any())

    def unique_keywords(self) -> list[str]:
        """Normalised keywords in order of first appearance."""
        seen: dict[str, None] = {}
        for keywords in self.keyword_lists:
            for keyword in keywords:
                seen.setdefault(keyword, None)
        return list(seen)

    def head(self, n: int) -> "Corpus":
        """A corpus restricted to the first ``n`` records, order preserved."""
        return Corpus(
            frame=self.frame.head(n).reset_index(drop=True),
            source_path=self.source_path,
            column_map=dict(self.column_map),
            dropped_empty_text=self.dropped_empty_text,
            dropped_empty_keywords=self.dropped_empty_keywords,
        )

    def subset(self, mask: Sequence[bool] | pd.Series) -> "Corpus":
        return Corpus(
            frame=self.frame.loc[np.asarray(mask)].reset_index(drop=True),
            source_path=self.source_path,
            column_map=dict(self.column_map),
        )

    def summary(self) -> dict[str, Any]:
        keyword_counts = [len(k) for k in self.keyword_lists]
        return {
            "n_articles": self.n_articles,
            "n_unique_keywords": len(self.unique_keywords()),
            "mean_keywords_per_article": float(np.mean(keyword_counts)) if keyword_counts else 0.0,
            "articles_without_keywords": int(sum(1 for c in keyword_counts if c == 0)),
            "has_abstracts": self.has_abstracts,
            "has_citations": self.has_citations,
            "year_range": (
                (int(self.frame["year"].min()), int(self.frame["year"].max()))
                if self.frame["year"].notna().any()
                else None
            ),
            "dropped_empty_text": self.dropped_empty_text,
            "dropped_empty_keywords": self.dropped_empty_keywords,
        }


def _coerce_year(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"(\d{4})", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def from_dataframe(
    frame: pd.DataFrame,
    column_map: Mapping[str, str] | None = None,
    require_keywords: bool = True,
    source_path: str | None = None,
) -> Corpus:
    """Normalise an arbitrary bibliographic frame into a :class:`Corpus`.

    Parameters
    ----------
    frame:
        Source records, one row per publication.
    column_map:
        Optional explicit mapping of canonical field names (``title``,
        ``abstract``, ``author_keywords``, ``year``, ``citations``, ``doi``,
        ``authors``, ``source``) onto column names of ``frame``.
    require_keywords:
        Drop records without any author keyword.  Records without title and
        abstract are always dropped because they cannot be embedded.
    """
    resolved = _resolve_columns(list(frame.columns), column_map)
    missing = [name for name in _REQUIRED if name not in resolved]
    if missing:
        raise ValueError(
            f"corpus is missing required column(s) {missing}; "
            f"available columns: {list(frame.columns)}"
        )

    out = pd.DataFrame(index=range(len(frame)))
    source = frame.reset_index(drop=True)
    out["title"] = source[resolved["title"]].astype(str).fillna("")
    out["abstract"] = (
        source[resolved["abstract"]].astype(str).fillna("")
        if "abstract" in resolved
        else ""
    )
    out["abstract"] = out["abstract"].replace({"nan": "", "None": ""})
    out["title"] = out["title"].replace({"nan": "", "None": ""})
    out["author_keywords"] = source[resolved["author_keywords"]].apply(split_author_keywords)
    out["keywords_normalized"] = out["author_keywords"].apply(
        lambda keywords: list(dict.fromkeys(
            kw for kw in (normalize_keyword(k) for k in keywords) if kw
        ))
    )
    out["combined_text"] = (out["title"].str.strip() + " " + out["abstract"].str.strip()).str.strip()
    out["year"] = _coerce_year(source[resolved["year"]]) if "year" in resolved else np.nan
    out["citations"] = (
        pd.to_numeric(source[resolved["citations"]], errors="coerce")
        if "citations" in resolved
        else np.nan
    )
    for optional in ("doi", "authors", "source", "eid"):
        out[optional] = (
            source[resolved[optional]].astype(str).replace({"nan": "", "None": ""})
            if optional in resolved
            else ""
        )

    before = len(out)
    out = out[out["combined_text"].str.strip() != ""].copy()
    dropped_text = before - len(out)

    dropped_keywords = 0
    if require_keywords:
        before = len(out)
        out = out[out["keywords_normalized"].apply(len) > 0].copy()
        dropped_keywords = before - len(out)

    out = out.reset_index(drop=True)
    return Corpus(
        frame=out,
        source_path=source_path,
        column_map=resolved,
        dropped_empty_text=dropped_text,
        dropped_empty_keywords=dropped_keywords,
    )


def load_csv(
    path: str | Path,
    column_map: Mapping[str, str] | None = None,
    require_keywords: bool = True,
    **read_csv_kwargs: Any,
) -> Corpus:
    """Load a Scopus/WoS style CSV export into a :class:`Corpus`.

    Any keyword argument is forwarded to :func:`pandas.read_csv`, so unusual
    separators or encodings can be handled by the caller.
    """
    read_csv_kwargs.setdefault("low_memory", False)
    frame = pd.read_csv(path, **read_csv_kwargs)
    return from_dataframe(
        frame,
        column_map=column_map,
        require_keywords=require_keywords,
        source_path=str(path),
    )
