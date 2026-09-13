# Corpora

The three Scopus exports analysed in the article, after dropping records
without author keywords. Regenerate them with `python examples/fetch_scopus.py`
(needs `SCOPUSAPI` and `SCOPUSINSTTOKEN`).

| File | Records | Scopus query (all with `DOCTYPE(ar) AND LANGUAGE(english)`) |
|------|---------|--------------------------------------------------------------|
| `management.csv` | 493 | `TITLE-ABS-KEY("dynamic capabilit*" AND "digital transformation") AND SUBJAREA(BUSI OR ECON OR DECI)` |
| `finance.csv` | 487 | `TITLE-ABS-KEY(("fintech" OR "financial technology") AND ("financial inclusion" OR "credit risk" OR "bank lending")) AND SUBJAREA(ECON OR BUSI)` |
| `gamification.csv` | 290 | `TITLE-ABS-KEY(gamification AND marketing)` |

Columns: `Title`, `Abstract`, `Author Keywords`, `Year`, `Cited by`, `DOI`,
`Authors`, `Source title`, `EID`. Bibliographic metadata is reproduced here for
reproducibility of the analysis; rights to the underlying records remain with
Elsevier.
