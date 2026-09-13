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
python run_validation.py                # LLM panel, SELECTED vs control
python run_matched.py                   # frequency-matched arm
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
`NOISE` and scored 1–10 for novelty against the rubric reproduced in S8. The
control set is every unique author keyword of the same analysis corpus (146,
168 and 172 keywords), annotated once per corpus and shared by both modes so
that the two are compared against an identical baseline. The sheet reports
label rates, Novelty means, the Welch and Mann-Whitney tests, Cohen's *d*,
Fleiss' κ, Krippendorff's α, the leave-out concordance and any annotator
failures (none in the final run).

## S4. Frequency-matched arm (sheet `S4_matched_arm`)

The eligible pool is every keyword of the analysis corpus that passes the
frequency ceiling, the token-count rule, the methodological blocklist and the
core-domain filter, minus the candidates the ranking selected: 45 keywords for
management (all annotated) and 64 for economics and finance (60 sampled with
seed 2026). Two contrasts are reported:

- `eligible vs all` — what the lexical and frequency filters contribute.
- `selected vs eligible` — what the embedding-based ranking contributes on top
  of them.

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

289 tests, 93 % statement coverage, no network access required. Coverage
includes: the three scoring formulae against hand-computed values; the four
detectors on keyword spaces with embeddings placed at exact angles, so the
similarity window is tested against the published thresholds rather than
against whatever a model produces; both modes end-to-end on a synthetic corpus
with planted gaps; cross-process determinism under differing `PYTHONHASHSEED`
values; every hosted backend's request payload and reply parsing through a
recorded HTTP transport, including rate-limit retries and the malformed-reply
salvage path; and the full validation and figure stacks.
