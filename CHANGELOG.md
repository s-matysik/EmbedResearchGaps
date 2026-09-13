# Changelog

## 1.1.0

Changed in response to peer review of the accompanying article. The statistical
fix changes reported p-values, so results produced with 1.0.0 are not
comparable with results produced here.

### Fixed

- **Control set independence.** `selected_and_all_keywords` returned the whole
  author-keyword population as the control set, which contains the candidates:
  every candidate contributed to both arms, and the Welch and Mann-Whitney
  statistics computed on them had no defined null distribution. The new
  `selected_and_control_keywords` returns a control set disjoint from the
  candidates; the old function is kept as a deprecated alias so a 1.0.0 study
  can still be reproduced exactly. `compare_keyword_sets` now detects
  overlapping arms and reports `overlapping_samples = 1.0` instead of
  parametric p-values.
- Consensus aggregates are coerced to numeric, so a mode that returns no
  candidate for one run no longer leaves the pooled score columns unsortable.

### Added

- `consensus_gaps(..., top_n=(...))` pools candidates over analysis-corpus
  sizes as well as k-means seeds, and `stability` now separates the overlap
  between runs sharing a corpus size (the seed effect) from the overlap across
  sizes (the size effect). On the case-study corpora the corpus size moves the
  candidate list more than the seed does.
- `permutation_mean_difference`, a two-sided permutation test on the difference
  of means, reported alongside Welch and Mann-Whitney. It assumes neither
  normality nor equal variance, which matters when one arm holds a few dozen
  judgements.
- `LOW_CLUSTER_SEPARATION`: a run whose partition has a silhouette below 0.10
  records a warning in `diagnostics["warnings"]` and raises it as a
  `UserWarning`, because a candidate list read off a partition that weak should
  not be interpreted as a single-run ranking.
- Continuous integration running the suite on Python 3.10 to 3.13.

### Changed

- The eligible-pool comparison is described as *filter-eligible* rather than
  *frequency-matched*: the arms are defined by which filters a keyword passes,
  not by matched frequency distributions. Its second contrast is now eligible
  against not-eligible keywords, which is disjoint, replacing eligible against
  the whole population, which was not.

## 1.0.0

First public release: two operating modes, seven embedding backends,
stability-aware consensus across seeds, a six-provider LLM annotation panel,
blinded expert sheets, seven figures and a CLI.
