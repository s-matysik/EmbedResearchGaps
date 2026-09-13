# Supplementary material

**EmbedResearchGaps: embedding-driven identification of research gaps in
author-keyword space**

Everything below is produced by the scripts in the repository from the corpora
in `corpora/`. Numbers in the article are taken from these tables, not
transcribed. The workbook `supplementary_material.xlsx` holds one sheet per
section below.

## S0. How to regenerate every number

```bash
pip install -e ".[all]"
export SCOPUSAPI=...            # Scopus search, corpus retrieval
export SCOPUSINSTTOKEN=...
export OPENAI=...               # text-embedding-3-large and GPT-4o
export ANTROPIC=... DEEPSEEK=... XAI=... GOOGLE=...   # annotation panel

python fetch_scopus.py                  # corpora/*.csv
python run_case.py                      # results/<corpus>/, both modes
python run_sensitivity.py               # results/sensitivity/
python run_validation.py                # LLM panel annotation
python run_matched.py                   # filter-eligible arm
python recompute_validation.py          # comparisons against a disjoint control
python run_consensus.py                 # consensus over seeds and corpus sizes
python make_figures.py                  # figures/fig1..fig6
python build_supplementary.py           # this workbook
```

Embeddings are cached in `cache/<corpus>_embeddings.npz`, so every stage after
the first run costs no API calls and reproduces bit-for-bit. Each
`results/<corpus>/<mode>_run_config.json` records the full configuration of the
run that produced the tables next to it.

## S1. Corpora (sheet `S1_corpora`)

Record counts, publication-year span, citation distribution, abstract coverage
and mean author keywords per record for the three corpora. Scopus queries, all
restricted to `DOCTYPE(ar) AND LANGUAGE(english)`:

| Corpus | Query | Subject areas | Records |
|--------|-------|---------------|---------|
| Management | `TITLE-ABS-KEY("dynamic capabilit*" AND "digital transformation")` | BUSI, ECON, DECI | 493 |
| Economics & finance | `TITLE-ABS-KEY(("fintech" OR "financial technology") AND ("financial inclusion" OR "credit risk" OR "bank lending"))` | ECON, BUSI | 487 |
| Gamification (conformance) | `TITLE-ABS-KEY(gamification AND marketing)` | — | 290 |

Records without author keywords are dropped at load time; the counts above are
after that filter.

## S2. Run diagnostics (sheet `S2_diagnostics`)

Per corpus and mode: analysis-corpus size, keyword-space size, selected *k* and
the rule that selected it, silhouette, Davies-Bouldin and Calinski-Harabasz
indices, cluster sizes, candidate counts by type, the frequency threshold and
peripheral share, the detected core-domain terms, candidate counts before and
after cross-cluster deduplication, the TF-IDF match-type breakdown and the
encoder description.

The silhouette of the chosen partition lies between 0.017 and 0.069 across all
six runs (0.017–0.055 for the article partitions of Mode B). The elbow rule
therefore selected *k* in every run, because the best silhouette over the
candidate range never reached the 0.25 floor. This weak cluster structure is
the direct cause of the seed sensitivity reported in S6.

## S3. LLM panel validation (sheet `S3_validation`)

Five providers, all at `temperature=0`:

| Provider | Model | Key variable |
|----------|-------|--------------|
| OpenAI | `gpt-4o` | `OPENAI` |
| Anthropic | `claude-sonnet-4-6` | `ANTROPIC` |
| DeepSeek | `deepseek-v4-pro` | `DEEPSEEK` |
| xAI | `grok-4.5` | `XAI` |
| Google | `gemini-2.5-flash` | `GOOGLE` |

Moonshot (Kimi) was probed and excluded: its current models reject
`temperature=0`, which would break the determinism the protocol depends on.
Claude Sonnet 5 was likewise excluded because `temperature` is deprecated for
it; `claude-sonnet-4-6` accepts the parameter and was used instead.

Each keyword is labelled `GAP`, `EXPLORED`, `LOW IMPORTANCE`, `METHOD` or
`NOISE` and scored 1–10 for novelty against the rubric reproduced in S8.

Every unique author keyword of the analysis corpus was annotated (146, 168 and
172 keywords), once per corpus and shared by both modes so that the two are
compared against an identical baseline. The **control arm excludes the
candidates**: comparing the candidate set against the whole population would
nest one sample inside the other, every candidate would contribute to both
arms, and a Welch or Mann-Whitney statistic computed on them would have no
defined null distribution. Restricting the control to the non-selected
keywords removes 20 to 130 judgements per corpus and mode from the control arm
and leaves 120 to 166 control keywords. `compare_keyword_sets` now refuses to
report parametric p-values when the two arms it is handed share a keyword, and
`selected_and_all_keywords` — which returned the nested control in v1.0.0 — is
deprecated in favour of `selected_and_control_keywords`.

The sheet reports label rates, Novelty means, the Welch and Mann-Whitney tests,
Cohen's *d*, a permutation test on the difference of means (20 000 resamples,
seed 42, which assumes neither normality nor equal variance), Fleiss' κ,
Krippendorff's α, the leave-out concordance and any annotator failures (none in
the final run).

Six comparisons were run (three corpora × two modes). One is nominally
significant — economics and finance Mode B, +12.7 pp GAP rate and +12.5 % mean
novelty, Welch *p* = 0.032, Mann-Whitney *p* = 0.014, permutation *p* = 0.046,
*d* = 0.21 — and none survives Holm correction across the six (smallest
adjusted *p* = 0.19). The `welch_p_holm` column of
`results/manuscript_numbers_disjoint.csv` carries the adjusted values.

## S4. Filter-eligible comparison (sheet `S4_matched_arm`)

This is an eligibility-matched comparison, not a frequency-matched one: the
distributions of document frequency are not matched between arms, the arms are
defined by which filters a keyword passes. The *eligible pool* is every keyword
of the analysis corpus that passes the frequency ceiling, the token-count rule,
the methodological blocklist and the core-domain filter, minus the candidates
the ranking selected: 45 keywords for management (all annotated) and 64 for
economics and finance (60 sampled with seed 2026). Two fully disjoint contrasts
are reported:

- `eligible vs not_eligible` — the eligible pool against the keywords that
  fail at least one filter: what the lexical and frequency filters contribute.
  Management +7.8 pp GAP and +8.3 % novelty (permutation *p* = 0.149);
  economics and finance +10.6 pp and +9.2 % (*p* = 0.069).
- `selected vs eligible` — the candidates against the eligible pool they were
  selected from: what the embedding-based ranking contributes on top of the
  filters. Management −6.1 pp and −8.4 % (*p* = 0.192); economics and finance
  +6.8 pp and +8.6 % (*p* = 0.150).

The filters move both corpora in the same direction by a similar amount; the
ranking moves them in opposite directions. The earlier `eligible vs all`
contrast was dropped because the eligible pool is a subset of the population it
was compared against.

## S5. Sensitivity to corpus size (sheet `S5_size_sensitivity`)

Both modes at *N* = 30, 50, 100 and 150 on all three corpora: selected *k*,
keyword-space size, candidate counts by type, silhouette, and the Jaccard
overlap of the candidate set with the *N* = 50 run. The `candidates` column of
the CSV holds the full candidate list of every cell; it is omitted from the
workbook for width.

## S6. Seed stability (sheet `S6_seed_stability`)

Five k-means seeds (42, 7, 123, 2026, 31337) with corpus, embeddings and every
threshold held fixed. Reported: mean, minimum and maximum pairwise Jaccard
overlap of the candidate sets and the mean candidate count. This is the
measurement `consensus_gaps` exposes as a per-candidate support value.

## S6b. Consensus across seeds and corpus sizes (sheet `S6b_consensus`)

`consensus_gaps(..., seeds=(42, 7, 123, 2026, 31337), top_n=(30, 50, 100, 150))`
executes twenty runs per corpus and mode and pools the candidates by support,
the fraction of runs that produced them. Reported per corpus and mode: the
number of pooled candidates, how many reach majority and unanimous support, the
mean pairwise Jaccard overlap, and its decomposition into run pairs that share
a corpus size (the seed effect) and pairs that do not (the size effect).

| Corpus | Mode | Pooled | Support ≥ 0.5 | Unanimous | Mean J | Same size | Across sizes |
|--------|------|--------|----------------|-----------|--------|-----------|--------------|
| Management | B | 161 | 12 | 2 | 0.234 | 0.451 | 0.176 |
| Management | A | 50 | 2 | 0 | 0.158 | 0.288 | 0.123 |
| Economics & finance | B | 166 | 11 | 2 | 0.194 | 0.385 | 0.143 |
| Economics & finance | A | 37 | 1 | 0 | 0.140 | 0.262 | 0.108 |

Full tables: `results/<corpus>/<mode>_consensus_seeds_sizes_consensus_gaps.csv`
(support over both factors) and `..._consensus_seeds_consensus_gaps.csv` (seeds
only), with `consensus_summary.json` beside them.

High support is not validity. The two personal names that a metadata error left
in the management keyword field reach support 0.85 and 0.75, because a data
defect reproduces perfectly across initialisations.

## S7. Full candidate tables (sheets `S7_*`)

Every candidate of every corpus and mode, with its type, cluster label,
composite score, document frequency, number of source articles, citation
count, the one-line rationale, the source article titles and DOIs, and all
underlying metric components (`metric_*` columns).

Two entries of the management Mode A table — `Ryad Titah` and `Pär Ågerfalk` —
are personal names that a metadata error placed in the author-keyword field of
one Scopus record (DOI 10.1080/0960085X.2020.1857666, single-authored by
J. Soluk, so neither name is the article's author). They are retained
deliberately rather than blocklisted: the annotation panel labelled both
`NOISE` with a Novelty Score of 1, which is what the validation layer is for.
Filtering personal names by pattern would be guesswork on a keyword field that
legitimately contains proper nouns (*Mittelstand*, *South Asia*), so the
package leaves the decision to the validation layer and the reviewer.

## S8. Annotation protocol

The prompt supplies the research problem, the domain, the numbered keyword
batch and this rubric verbatim:

- `GAP` — a topic plausibly relevant to the research problem that the corpus
  has not addressed substantively.
- `EXPLORED` — already well covered in the corpus or the wider literature.
- `LOW IMPORTANCE` — peripheral to the research problem.
- `METHOD` — a method, instrument, statistical technique or study design.
- `NOISE` — not a research topic (a place name alone, an artefact, an
  unintelligible string).

Novelty Score bands: 1–2 established, 3–4 partially explored, 5–7
under-explored with clear potential, 8–10 genuinely novel and promising.

Replies are requested as JSON only, one entry per keyword. Entries naming a
keyword that was not in the batch, or carrying a label outside the five, are
discarded rather than repaired. A reply that is almost-valid JSON — an
unescaped quotation mark inside a rationale, a trailing comma, a truncated
tail — is salvaged entry by entry so that one malformed entry does not discard
the rest of the batch; a batch yielding nothing usable is requested once more
before the error propagates.

## S8b. Prior expert validation of the method

The expert arm of the evidence base for this method is the AMCIS antecedent
study, reference [17] of the manuscript, not this paper. There, two independent
experts — an associate professor and a full professor, each with more than
twenty years in the field of the corpus — scored the sixteen candidates that
the articles-first procedure produced on a marketing corpus, using the same
1–10 Novelty Score and the same label set reproduced in S8. Three candidates
(*social commerce*, *time poverty*, *psychological distance*) were labelled
`GAP` unanimously by all five models of that study's panel and corroborated as
gaps by at least one expert. Expert mean Novelty Scores correlated with the
panel means at Spearman ρ = 0.61 (*p* = 0.012, *n* = 16). That study reported
the spread of expert opinion as wider than the spread across models and
described its expert arm as a pilot.

This paper did not repeat that arm. A sixteen-keyword pilot on one marketing
corpus and the five-provider panel study of S3 on three corpora in two other
domains are different measurements, and the disagreement between them — a
positive expert-panel concordance there, no reliable enrichment over a disjoint
control here — is what a larger calibrated expert study is needed to resolve.
What this paper contributes to that study is the instrument: the blinded sheets
of S9, shipped for all three corpora, and the concordance function that reads
them back.

## S9. Expert validation protocol

`build_expert_sheet` produces a blinded assessment sheet: rows shuffled with a
fixed seed, gap types and pipeline scores withheld, one tab per reviewer plus a
rubric tab identical to S8. Reviewers fill a Novelty Score and a label per
keyword. `load_expert_sheet` reads the completed workbook and
`expert_llm_concordance` reports Spearman ρ between the expert mean and the
panel mean, the share of keywords on which the majority labels agree, and the
GAP share of each side.

Sheets for both case studies are shipped as
`results/<corpus>/expert_sheet.xlsx`. No human expert study was run for this
article; the concordance code path was exercised by holding out two of the five
models and correlating them against the remaining three, which is reported as
`leaveout_*` in S3 and is a model-versus-model check, not a substitute for
human judgement.

## S9b. Input formats

`load_csv` resolves canonical fields (`title`, `abstract`, `author_keywords`,
`year`, `citations`, `doi`, `authors`, `source`, `eid`) through
`COLUMN_ALIASES`, which covers three export formats without a column map:

| Format | Title | Author keywords | Citations |
|--------|-------|-----------------|-----------|
| Scopus CSV export | `Title` | `Author Keywords` | `Cited by` |
| Scopus Search API (`view=COMPLETE`) | `dc:title` | `authkeywords` | `citedby-count` |
| Web of Science, tab-delimited (`savedrecs.txt`) | `TI` | `DE` | `TC` |
| Web of Science, full record | `Article Title` | `Author Keywords` | `Times Cited, All Databases` |

The tab-delimited Web of Science file needs the separator passed through to
pandas: `load_csv("savedrecs.txt", sep="\t")`. Any other export is handled by
`column_map={"author_keywords": "<column>", ...}`.

Web of Science `Keywords Plus` (`ID`) is **not** an alias for author keywords.
Those terms are generated by the database from the titles of cited references,
not assigned by the authors, and both published procedures are defined on
author-assigned keywords; mixing them in would change what the periphery
criterion is measuring. A user who wants them must pass
`column_map={"author_keywords": "Keywords Plus"}` and report that choice.
`tests/test_corpus.py::TestWebOfScienceExports` loads all three flavours of the
same records, asserts they produce an identical corpus, and asserts that
Keywords Plus is not picked up implicitly.

## S10. Parameters and formulae

Defaults, all overridable through `RunConfig`:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `top_n_articles` | 50 | analysis-corpus size after semantic ranking |
| `periphery_percentile` | 95 | within-cluster distance percentile (Mode A) |
| `max_keyword_frequency` | `round(N/50)` | document-frequency ceiling for a candidate |
| `gap_token_count` | 2 | required token count of a candidate keyword |
| `core_domain_share` | 0.30 | a keyword in this share of records defines the field |
| `combination_similarity_range` | (0.25, 0.65) | admissible cosine window for a combination |
| `combination_similarity_peak` | 0.45 | cosine at which the similarity term peaks |
| `min_tfidf` | 0.0 | TF-IDF salience floor |
| `max_gaps_per_cluster` | 10 | candidates kept per cluster per detector |
| `k_min`, `k_max` | 2, 10 | candidate range for k-selection |
| `silhouette_floor` | 0.25 | below this the elbow rule decides *k* |
| `n_init` | 10 | k-means restarts per fit |
| `random_state` | 42 | seed for k-means and sampling |

Scoring, with every component min-max normalised within the candidate set:

- Emerging Concept: `0.25·distance + 0.15·centrality + 0.30·tfidf + 0.30·rarity`
- Conceptual Combination: `0.40·similarity + 0.30·frequency + 0.30·tfidf`
- Cross-Cluster Concept: `0.25·frequency + 0.35·specificity + 0.40·tfidf`
- Peripheral Keyword (Mode A): distance from the keyword's own cluster centroid

Article ranking and all similarity measures use cosine distance,
`d(u, v) = 1 − cos(u, v)`. Inter-cluster thematic overlap is the pairwise
cosine similarity of cluster centroids.

## S11. Test suite

316 tests, 93 % statement coverage, no network access required. Coverage
includes: the three scoring formulae against hand-computed values; the four
detectors on keyword spaces with embeddings placed at exact angles, so the
similarity window is tested against the published thresholds rather than
against whatever a model produces; both modes end-to-end on a synthetic corpus
with planted gaps; cross-process determinism under differing `PYTHONHASHSEED`
values; every hosted backend's request payload and reply parsing through a
recorded HTTP transport, including rate-limit retries and the malformed-reply
salvage path; the disjointness of the control set and the refusal to report
parametric p-values on overlapping arms; the permutation test; consensus
pooling over seeds and corpus sizes including the seed/size decomposition; and
the full validation and figure stacks; and both Web of Science export flavours
against the Scopus export of the same records.
