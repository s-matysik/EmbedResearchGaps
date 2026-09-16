# EmbedResearchGaps — handoff

State of the work as of 13 September 2026, written so that a second person can
pick it up in a fresh Claude Science session without reconstructing anything.

Everything below is either in the repository or derivable from it. Nothing
needed to continue lives only in the session that produced it.

---

## 1. Where the authoritative copy is

| | |
|---|---|
| Repository | https://github.com/s-matysik/EmbedResearchGaps |
| Release cited by the article | `v1.0.0` — https://github.com/s-matysik/EmbedResearchGaps/releases/tag/v1.0.0 |
| Commit the tag points at | `eed714ad7c` |
| Package version | 1.0.0 (`pyproject.toml`, `src/embedresearchgaps/__init__.py`, `CITATION.cff` all agree) |
| Test suite | 316 tests, 93 % statement coverage, runs offline |

The repository is the transfer medium. Claude Science artifact IDs are scoped
to the workspace that created them, so they are useless to a different account;
clone the repository instead, or upload
`EmbedResearchGaps_reproduction_bundle.tar.gz` (26 MB, attached alongside this
file) into the new session — the bundle is the repository plus the corpora,
every result table, the figures and the built submission documents.

```bash
git clone https://github.com/s-matysik/EmbedResearchGaps
cd EmbedResearchGaps && git checkout v1.0.0
```

## 2. Environment

Python 3.10–3.13. One conda environment is enough; CI covers all four versions.

```bash
pip install -e ".[dev,excel,agreement]"     # numpy, pandas, scipy, scikit-learn,
                                            # matplotlib, networkx, kneed, requests,
                                            # typer + pytest, openpyxl, krippendorff
pip install python-docx                     # only for build_docx.py / build_docx_supp.py
PYTHONPATH=src:tests python -m pytest tests/ -q      # expect 316 passed
```

`sentence-transformers` is optional and only needed for the `sbert` embedding
backend. The default backend, `tfidf-svd`, is offline and deterministic, so the
whole test suite and any smoke run work with no credentials at all.

## 3. Credentials the pipeline expects

Read from environment variables, never written to a config object, a result
file or a log. In Claude Science they go in Customize → Credentials under
exactly these names:

| Variable | Used for |
|---|---|
| `SCOPUSAPI`, `SCOPUSINSTTOKEN` | corpus retrieval (`fetch_scopus.py`) |
| `OPENAI` | `text-embedding-3-large` embeddings |
| `OPENAI`, `ANTROPIC`, `DEEPSEEK`, `XAI`, `GOOGLE` | the five-provider annotation panel |
| `GITHUBCLASSIC` | pushing releases (a classic PAT with `repo` scope) |

**The corpora are already in `data/`** — retrieval does not need to be
repeated, and re-running it would produce a different corpus as Scopus grows.
Embeddings are cached under `cache/` *within a session*, but that cache is
**not** shipped: it is regenerable and large. Consequently the first
`run_case.py` in a new checkout re-embeds through the OpenAI API and costs
money, while every later stage in the same session is free. Reading or citing
any reported number needs neither: all result tables are in `paper/results/`,
and `paper/results/manuscript_numbers_disjoint.csv` holds every figure quoted
in the article.

The five-provider panel (`run_validation.py`) is the one genuinely expensive
stage. Its annotations are shipped, and `recompute_validation.py` recomputes
every comparison from them without a single API call — use that rather than
re-annotating.

Moonshot (Kimi) was probed and deliberately excluded from the panel: its models
reject `temperature=0`, which the annotation protocol requires.

## 4. Reproduction order

Scripts live in `examples/` and are analysis drivers, not library code.

```
fetch_scopus.py        -> data/*.csv                (already done; do not re-run)
run_case.py            -> results/<corpus>/         both modes, tables, diagnostics
run_sensitivity.py     -> results/sensitivity/      corpus-size and seed stability
run_validation.py      -> panel study               COSTS API CALLS
run_matched.py         -> filter-eligible arm
recompute_validation.py-> disjoint-control results  no API calls, reads annotations on disk
run_consensus.py       -> consensus over 5 seeds x 4 sizes
make_figures.py        -> figures/fig1..fig6
build_supplementary.py -> the supplementary workbook
build_docx.py          -> the article in the journal template
build_docx_supp.py     -> the supplementary document
```

Every number quoted in the article is collected in
`paper/results/manuscript_numbers_disjoint.csv`. Each
`results/<corpus>/<mode>_run_config.json` fully describes the run that produced
the tables beside it.

## 5. Decisions that bind — do not silently revert these

These were argued through and several of them correct real defects. Changing
any of them changes reported numbers, so a change needs a decision, not a
refactor.

1. **The validation control set is disjoint from the candidate set.**
   `selected_and_control_keywords` builds it; `compare_keyword_sets` *refuses*
   to report parametric p-values when the two arms share a keyword. The
   earlier version compared candidates against the whole keyword population,
   which contains them — a two-sample statistic on nested samples has no
   defined null distribution. `selected_and_all_keywords` is kept as a
   deprecated alias so the superseded study still reproduces.
2. **`Keywords Plus` / `ID` is not an implicit alias for author keywords.**
   Those terms are generated by the database from cited-reference titles, and
   both procedures are defined on author-assigned keywords. Opt in explicitly
   with `column_map={"author_keywords": "Keywords Plus"}`.
3. **Co-author names that reach the candidate list are reported, not
   filtered.** One management-corpus article carries two co-author names in its
   Author Keywords field; they surface as Mode A candidates and the panel
   labels them `NOISE` with Novelty Score 1. The article uses this as evidence
   that the validation layer works. A proper-noun filter would hide a metadata
   defect the paper argues should surface.
4. **Keyword ordering is insertion-ordered, never set-ordered.**
   `tests/test_determinism.py` runs both pipelines in subprocesses under
   different `PYTHONHASHSEED` values and requires identical candidate tables,
   and asserts that three repeats at one fixed seed agree. The support
   statistic of `consensus_gaps` depends on this.
5. **The published version is 1.0.0.** Three higher tags existed earlier in the
   day and were deleted with the author's approval, because this is the first
   public upload and a version cannot move backwards. The article's metadata
   row C1, its permanent link C2 and reference [22] all point at `v1.0.0`.
6. **The article is produced by filling the journal template, not by
   recreating it.** `build_docx.py` opens
   `softwarex-osp-template.docx` as the base document and attaches the five
   mandatory sections to the template's own multilevel list (`numId 1`), so
   Word generates `1.`, `2.`, `2.1.` itself. Never type section numbers into
   heading text — the list would duplicate them.
7. **The honest headline result stays honest.** The embedding-based ranking
   gave *no* reliable novelty enrichment over a disjoint control beyond what
   the lexical and frequency filters already provide (Holm-adjusted minimum
   p = 0.19). The article reports this, recommends consensus pooling with
   per-candidate support, and frames output as hypotheses for expert
   screening. Do not let a later edit inflate this into a positive claim.

## 6. Open items

- **The corresponding-author email is unverified.**
  `sebastian.matysik@usz.edu.pl` appears in metadata row C8, in the
  corresponding-author line and in `CITATION.cff`. It was inferred from
  context, not retrieved from a trusted source — `host.get_user_email()`
  raised `ContactEmailUnavailable`. Confirm it before submission.
  Earlier commits in the repository history also carry it as the commit author;
  removing it from history would need a local `git filter-repo`, which was not
  done.
- **The six-page limit for sections 1–5 was never measured exactly.** The
  sandbox has no DOCX→PDF converter, so only a character-based estimate exists
  (~5.5 pages). Open the document in Word and read the page count. The word
  limit *is* verified: 3988 of 4000, counted as the journal counts it
  (abstract 107 + running text + captions + headings + declarations).
- **No expert study was run for this paper.** The expert arm of the evidence
  base is the AMCIS antecedent study (two experts, 16 candidates, Spearman
  ρ = 0.61, p = 0.012), reported in section 3.3 and supplementary S8b with an
  explicit statement that this paper did not repeat it. The instrument ships:
  `build_expert_sheet` writes blinded, row-shuffled workbooks (one per corpus,
  in `paper/results/`) and `expert_report` reads them back into a Spearman
  concordance against the panel. Running it with real experts is the obvious
  next study.
- **Submission itself has not happened.** `paper/submission/` holds the files
  in the form the journal asks for: the article in the template, the
  supplementary document, `figures/Figure_1..6.png` at 300 dpi,
  `highlights.txt` and `graphical_abstract.png`.

## 7. What lives where in the repository

| Path | Contents |
|---|---|
| `src/embedresearchgaps/` | the library: twelve modules plus a `validation` subpackage of five |
| `tests/` | 316 tests, offline; hosted backends tested against a recorded HTTP transport |
| `examples/` | the reproduction drivers listed in section 4 |
| `data/` | the three Scopus corpora with the queries and retrieval dates that produced them |
| `paper/manuscript.md`, `paper/supplementary_material.md` | the article sources |
| `paper/results/` | candidate tables, keyword tables, diagnostics, run configs, panel annotations, blinded expert sheets, sensitivity and consensus tables |
| `paper/figures/` | the six manuscript figures |
| `paper/submission/` | the files as submitted |
| `paper/refs_crossref.json` | Crossref metadata for every reference, fetched by DOI and verified twice per entry |
| `CHANGELOG.md` | one initial-release entry, with a note on how this release departs from the prototype notebooks |

## 8. Starting the next session

Paste this into the new Claude Science session, with the bundle or the
repository URL attached:

> This continues work on EmbedResearchGaps, a Python package implementing two
> published research-gap identification procedures, submitted to SoftwareX as
> an Original Software Publication. The repository is
> https://github.com/s-matysik/EmbedResearchGaps at tag v1.0.0. Read
> `HANDOFF.md` first: section 5 lists decisions that must not be reverted and
> section 6 the open items. The immediate next steps are confirming the
> corresponding-author email, checking the six-page limit in Word, and
> deciding whether to run the expert study the package ships the instrument
> for.
