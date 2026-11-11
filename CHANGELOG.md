# Changelog

## 1.1.0 - 2026-11-11

The revision submitted in response to peer review of the accompanying article
(SOFTX-D-26-01416). No reported number from 1.0.0 changes; the additions are
new analyses of the already deposited panel judgements, and corrections to how
existing numbers were described.

### Added

- `examples/revision/`: the analyses requested in review, all computed from the
  deposited annotations without further model calls. `analysis_baselines.py`
  scores random, rarity-matched random and frequency-ordered shortlists of the
  same size against the same panel judgements; `analysis_rarity_matched.py`
  draws the rarity-matched comparator from the stratum the filters admit;
  `analysis_validity.py` reports minimal detectable effects, two one-sided
  tests against a pre-declared 10 pp margin, and ordinal Krippendorff's alpha
  with bootstrap intervals; `analysis_cluster_bootstrap.py` resamples article
  clusters rather than keywords; `analysis_rank_stability.py` measures
  rank-biased overlap, top-k overlap and Kendall's tau across seeds on the
  offline backend; `external_audit.py` demonstrates the `load_external_ranking`
  path end to end.
- `MAINTENANCE.md`: the reproduction path, the policy for hosted models that
  providers deprecate, and the release policy.
- `environment-lock.txt`: pinned versions of every dependency, including the
  numerical stack, for the runs reported in the article.
- `paper/figures/figS16_other_corpora.png` and the deposited tables under
  `paper/results/revision/`.

### Changed

- Continuous integration now runs the suite on Linux, macOS and Windows across
  Python 3.10-3.13, so the cross-process determinism guarantee is tested where
  BLAS and library builds differ rather than asserted.
- `examples/make_figures.py`: Figure 1 redrawn as three numbered stages naming
  the module behind each component; Figure 4's annotation colour now follows
  cell luminance instead of a fixed value threshold, which had put white text
  on pale cells; Figure 6 is built from keyword-level statistics with bootstrap
  intervals, the unit of analysis the article reports; figure titles are
  descriptive rather than interpretive.
- `examples/build_docx.py` renders a second body table using the template's own
  table style.
- `paper/`: manuscript and supplementary material revised; the reference list
  renumbered by first appearance and extended from 26 to 39 entries, each new
  DOI verified against Crossref.

### Fixed

- The supplementary GAP-rate test was never reported: the permutation column of
  S14 is the Novelty Score test. Both are now given, with Holm adjustment
  across the six comparisons (S14b).
- "No silhouette reaches 0.07" contradicted a run reaching exactly 0.070; the
  dangling cross-reference to a nonexistent article table now points to Table 1;
  the two seed-overlap statistics measured on different designs are now
  distinguished where they meet.
- Supplementary S7 no longer names the two individuals whose names a metadata
  error placed in an author-keyword field, nor the affected record.

## Unreleased

### Added

- **External rankings.** `load_external_ranking` reads a candidate ranking
  produced outside this package — VOSviewer, BERTopic, a competing
  implementation, a human reviewer — from a CSV, TSV or Excel export, a
  `DataFrame`, or a sequence of keywords, and returns the same `PipelineResult`
  the built-in modes return (`mode='external'`). The validation subpackage, the
  disjoint control-set split, the candidate tables, `result.save` and the
  overlap measures therefore apply to a foreign list unchanged, which is what
  makes it possible to assess someone else's ranking on a corpus, or to compare
  it against ours on the same one. Keyword, score, rank and type columns resolve
  from an alias table that also matches compound headers such as
  `Total link strength`. Passing a `Corpus` attaches document frequency, source
  records and citation counts, and reports candidates absent from the corpus in
  `diagnostics['external']['unmatched_in_corpus']` rather than silently
  recording zeros. No clustering of ours is attached — `cluster` is -1 and no
  partition diagnostic is computed — and foreign candidates are typed
  `External Candidate`, deliberately not one of `GAP_TYPES`, unless the file
  names a type. Exposed on the CLI as `embedresearchgaps external`.

- **Optional citation filter.** `GapConfig.min_citations` and
  `GapConfig.max_citations` bound the citation support of a candidate — the
  citation count of its best-cited source record — and may be combined into a
  window. Both are `None` by default, in which case the filter short-circuits
  and the published runs are reproduced exactly. The bounds are applied to
  candidate eligibility in all four detectors, before ranking and before the
  per-cluster cap. Candidates whose citation support is unknown are kept unless
  `keep_uncited_candidates=False`, so a bound cannot silently empty the
  candidate list on a corpus without citation data. Exposed on the CLI as
  `--min-citations`, `--max-citations` and `--drop-uncited/--keep-uncited`.
- **Citation velocity.** `GapConfig.min_citations_per_year` bounds the rate
  rather than the level: a record's citations divided by its age in years,
  `citation_year_reference - publication_year + 1`. Two records with the same
  total but different ages are no longer equivalent, which is what the level
  form cannot express. The reference year defaults to the most recent
  publication year in the corpus rather than to the system clock, so a
  deposited export reproduces the same candidates on any day. Fractional
  thresholds are allowed. Exposed as `--min-citations-per-year` and
  `--citation-year`. There is deliberately no upper rate bound.

### Fixed

- The README described a citation filter as part of `keywords_first`. No such
  filter existed: citation counts were read, reported and used as a ranking
  tie-breaker only. The sentence now matches the code, and the filter it
  described is the optional feature added above.

## 1.0.0 - 2026-09-13

First public release: the version described in the accompanying article.

### Added

- **Two operating modes.** `keywords_first` clusters the author keywords of the
  analysis corpus and flags those whose distance from their own cluster
  centroid exceeds a cluster-specific percentile. `articles_first` clusters the
  articles, labels each cluster, builds a keyword sub-space inside every
  cluster and applies the three detectors of the gap typology — emerging
  concept, conceptual combination, cross-cluster concept — each with its own
  composite score. Both share the frequency ceiling, token-count rule,
  methodological blocklist and core-domain filter.
- **Seven embedding backends** behind one contract: `tfidf-svd` (offline,
  deterministic, no credentials), `sbert`, `openai`, `nomic`, `jina`, `cohere`
  and `precomputed`. The contract validates the returned row count and rejects
  non-finite vectors, so a keyword can never be silently paired with the wrong
  embedding. API keys are read from named environment variables and never enter
  a configuration object, a result file or a log.
- **Database-agnostic input.** `load_csv` resolves canonical fields through
  `COLUMN_ALIASES`, covering the Scopus CSV export, the Scopus Search API and
  both Web of Science export flavours — the tab-delimited file with two-letter
  field tags and the full-record file with spelled-out headers — without a
  column map. `Keywords Plus` is deliberately *not* an implicit alias for
  author keywords: those terms are generated by the database from the titles of
  cited references rather than assigned by the authors, and both procedures are
  defined on author-assigned keywords. A run that wants them must ask
  explicitly with `column_map={"author_keywords": "Keywords Plus"}`.
- **Stability-aware consensus.** `consensus_gaps` repeats a mode across k-means
  seeds and, optionally, across analysis-corpus sizes, reporting for every
  candidate the fraction of runs that produced it, and decomposing the pairwise
  overlap of the candidate sets into the part attributable to the seed and the
  part attributable to the corpus size. A partition whose silhouette falls
  below 0.10 raises a `LOW_CLUSTER_SEPARATION` warning naming the measured
  value.
- **Validation subpackage.** The annotation protocol and 1–10 Novelty Score of
  the antecedent studies; six chat providers called at temperature 0;
  `CallableAnnotator` for local models; label rates, Fleiss' kappa,
  Krippendorff's alpha and pairwise agreement; and the candidate-versus-control
  comparison. `selected_and_control_keywords` builds a control set **disjoint**
  from the candidates, and `compare_keyword_sets` refuses to report parametric
  p-values when the two arms share a keyword, because a two-sample statistic on
  nested samples has no defined null distribution. Reported tests are Welch's
  t-test, Mann-Whitney U, Cohen's d and a permutation test on the difference of
  means. `build_expert_sheet` and `expert_report` produce blinded, row-shuffled
  expert workbooks and read them back into a Spearman concordance against the
  panel.
- **Reproducibility.** One `RunConfig` fixes every threshold, weight and seed,
  and `PipelineResult.save()` writes it next to the tables, so a published
  result can be regenerated from the artefacts alone. Keyword ordering is
  insertion-ordered throughout rather than set-ordered, and
  `tests/test_determinism.py` runs both pipelines in subprocesses under
  different `PYTHONHASHSEED` values and requires identical candidate tables.
- **Test suite and CI.** 316 tests, 93 % statement coverage, no network access
  required: detectors are exercised on keyword spaces with embeddings placed at
  designed angles, and every hosted backend is tested against a recorded HTTP
  transport that checks the request payload and the reply parsing. GitHub
  Actions runs the suite on Python 3.10 through 3.13.
- **Reference results.** `paper/` holds the manuscript and supplementary
  sources, the six figures, the candidate and validation tables for all three
  case-study corpora, the sensitivity and consensus tables, and the files as
  submitted to the journal. `data/` holds the corpora with the queries and
  retrieval dates that produced them.

### Notes

The two procedures implemented here were introduced as notebooks in the
antecedent conference papers (references [16] and [17] of the article). This
release departs from those prototypes in three ways that change reported
numbers: the control set used for validation is disjoint from the candidate
set, k-means initialisation and analysis-corpus size are explicit parameters
that consensus pools over rather than fixed defaults, and runs are reproducible
across processes. Results obtained with this package are therefore comparable
with each other but not term-for-term with the prototype outputs.
