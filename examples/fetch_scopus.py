"""Fetch Scopus corpora for the EmbedResearchGaps case studies.

Writes one CSV per corpus with the columns the package expects:
Title, Abstract, Author Keywords, Year, Cited by, DOI, Authors, Source title, EID.
Credentials are read from the environment; nothing is printed or stored.
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd
import requests

BASE = "https://api.elsevier.com/content/search/scopus"
HEADERS = {
    "X-ELS-APIKey": os.environ["SCOPUSAPI"].strip(),
    "X-ELS-Insttoken": os.environ["SCOPUSINSTTOKEN"].strip(),
    "Accept": "application/json",
}
PAGE = 25  # maximum page size of the COMPLETE view


def fetch(query: str, target: int, sort: str = "-citedby-count") -> pd.DataFrame:
    """Retrieve up to ``target`` records for ``query``, most cited first."""
    rows: list[dict[str, object]] = []
    start, total = 0, None
    while len(rows) < target:
        response = requests.get(
            BASE,
            headers=HEADERS,
            params={"query": query, "count": PAGE, "start": start,
                    "view": "COMPLETE", "sort": sort},
            timeout=(10, 60),
        )
        if response.status_code == 429:
            time.sleep(5)
            continue
        response.raise_for_status()
        results = response.json()["search-results"]
        if total is None:
            total = int(results["opensearch:totalResults"])
            print(f"  total available: {total}", flush=True)
        entries = results.get("entry") or []
        if not entries or "error" in entries[0]:
            break
        for entry in entries:
            keywords = entry.get("authkeywords") or ""
            rows.append(
                {
                    "Title": entry.get("dc:title") or "",
                    "Abstract": entry.get("dc:description") or "",
                    "Author Keywords": "; ".join(
                        part.strip() for part in keywords.split("|") if part.strip()
                    ),
                    "Year": (entry.get("prism:coverDate") or "")[:4],
                    "Cited by": entry.get("citedby-count") or 0,
                    "DOI": entry.get("prism:doi") or "",
                    "Authors": entry.get("dc:creator") or "",
                    "Source title": entry.get("prism:publicationName") or "",
                    "EID": entry.get("eid") or "",
                }
            )
        start += PAGE
        if total is not None and start >= min(total, 5000):
            break
        print(f"  fetched {len(rows)}", flush=True)
        time.sleep(0.35)
    return pd.DataFrame(rows)


CORPORA = {
    "management": (
        'TITLE-ABS-KEY("dynamic capabilit*" AND "digital transformation") '
        'AND DOCTYPE(ar) AND LANGUAGE(english) '
        'AND SUBJAREA(BUSI OR ECON OR DECI)',
        500,
    ),
    "finance": (
        'TITLE-ABS-KEY(("fintech" OR "financial technology") '
        'AND ("financial inclusion" OR "credit risk" OR "bank lending")) '
        'AND DOCTYPE(ar) AND LANGUAGE(english) '
        'AND SUBJAREA(ECON OR BUSI)',
        500,
    ),
    "gamification": (
        'TITLE-ABS-KEY(gamification AND marketing) '
        'AND DOCTYPE(ar) AND LANGUAGE(english)',
        300,
    ),
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(CORPORA)
    os.makedirs("corpora", exist_ok=True)
    for name in wanted:
        query, target = CORPORA[name]
        print(f"[{name}]", flush=True)
        frame = fetch(query, target)
        frame = frame[frame["Author Keywords"].str.strip() != ""].reset_index(drop=True)
        path = f"corpora/{name}.csv"
        frame.to_csv(path, index=False)
        print(f"  -> {path}: {len(frame)} records with author keywords", flush=True)
