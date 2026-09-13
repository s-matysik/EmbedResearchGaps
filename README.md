# EmbedResearchGaps

[![tests](https://github.com/s-matysik/EmbedResearchGaps/actions/workflows/tests.yml/badge.svg)](https://github.com/s-matysik/EmbedResearchGaps/actions/workflows/tests.yml)

Embedding-driven identification of research gaps in author-keyword space.

EmbedResearchGaps turns a bibliographic export (Scopus, Web of Science, or any
CSV with titles, abstracts and author keywords) into a ranked, auditable list of
candidate research gaps. It is the gap-detection companion to
[EmbedSLR](https://doi.org/10.1016/j.softx.2025.102416), which performs the
embedding-based screening step of a systematic literature review: EmbedSLR
decides *which papers belong in the review*, EmbedResearchGaps asks *what the
accepted papers do not talk about*.

The package implements two published procedures as two operating modes over the
same semantic space, and adds a reproducible validation layer (multi-model LLM
annotation plus human-expert concordance).

## Two operating modes

| | **Mode A — `keywords_first`** | **Mode B — `articles_first`** |
|---|---|---|
| Clustering order | author keywords directly | articles, then keywords inside each article cluster |
| Gap criterion | semantic periphery of a keyword cluster | three-type gap typology |
| Output granularity | one flat ranked list | candidates attributed to a named theme |
| Best for | fast periphery scan of a well-defined field | structured map of a heterogeneous field |
| Source | PACIS 2026 procedure | AMCIS 2026 procedure |

**Mode A** embeds the author keywords of the analysis corpus, clusters them with
k-means, and flags keywords whose distance from their own cluster centroid
exceeds a cluster-specific percentile (default: p95). Frequency, citation and
methodological-term filters then remove the terms that are peripheral for
uninteresting reasons.

**Mode B** clusters the articles first, labels each cluster from its dominant
keywords, then builds a keyword sub-space inside every cluster and applies three
detectors:

1. **Emerging Concepts** — rare, peripheral, lexically salient keywords.
2. **Conceptual Combinations** — pairs of established keywords that are
   semantically close enough to be combinable but never co-occur in one record.
3. **Cross-Cluster Concepts** — keywords well established elsewhere in the
   corpus but absent from this cluster.

## Installation

```bash
pip install embedresearchgaps                     # core
pip install "embedresearchgaps[all]"              # + sentence-transformers, krippendorff, openpyxl
```

From source:

```bash
git clone https://github.com/s-matysik/EmbedResearchGaps
cd EmbedResearchGaps
pip install -e ".[dev]"
pytest
```

Python 3.10 or newer. The core install needs no network access and no API key:
the default `tfidf-svd` encoder runs entirely offline.

## Quick start

```python
from embedresearchgaps import load_csv, RunConfig, EncoderConfig, articles_first

corpus = load_csv("scopus_export.csv")

config = RunConfig(
    top_n_articles=50,
    problem_description=(
        "How does the use of gamification in marketing influence "
        "consumer engagement and brand loyalty?"
    ),
    article_encoder=EncoderConfig(name="openai", model="text-embedding-3-large",
                                  api_key_env="OPENAI_API_KEY"),
    keyword_encoder=EncoderConfig(name="openai", model="text-embedding-3-large",
                                  api_key_env="OPENAI_API_KEY"),
)

result = articles_first(corpus, config)

print(result.n_clusters, result.cluster_labels)
print(result.top_gaps(10)[["gap_type", "keyword", "cluster_label", "score"]])
result.save("results/", prefix="gamification")
```

Mode A is a one-word change:

```python
from embedresearchgaps import keywords_first

result = keywords_first(corpus, config)
```

### Figures

```python
from embedresearchgaps.viz import generate_figures

generate_figures(result, "figures/", prefix="gamification")
```

Produces the keyword t-SNE map with candidates outlined, the article map, the
co-occurrence network, the inter-cluster similarity heatmap, the gap summary,
the per-cluster year distribution and the k-selection trace. Figures that do not
apply to a run are skipped and listed under the `skipped` key.

### Command line

```bash
embedresearchgaps run scopus_export.csv \
    --mode articles_first \
    --top-n 50 \
    --problem-file problem.txt \
    --encoder openai --model text-embedding-3-large --api-key-env OPENAI_API_KEY \
    --outdir results/
```

`embedresearchgaps encoders` lists the available backends.

## Input formats

Scopus and Web of Science exports both load without a column map. Canonical
fields are resolved through `COLUMN_ALIASES`:

| Source | Title | Author keywords | Citations |
|--------|-------|-----------------|-----------|
| Scopus CSV export | `Title` | `Author Keywords` | `Cited by` |
| Scopus Search API (`view=COMPLETE`) | `dc:title` | `authkeywords` | `citedby-count` |
| Web of Science, tab-delimited `savedrecs.txt` | `TI` | `DE` | `TC` |
| Web of Science, full record | `Article Title` | `Author Keywords` | `Times Cited, All Databases` |

```python
from embedresearchgaps import load_csv

scopus = load_csv("scopus.csv")
wos    = load_csv("savedrecs.txt", sep="\t")        # tab-delimited export
wos_xl = load_csv("wos_full_record.csv")             # spelled-out headers
other  = load_csv("export.csv", column_map={"author_keywords": "Index Terms"})
```

Web of Science **Keywords Plus** (`ID`) is deliberately *not* treated as author
keywords: those terms are generated by the database from the titles of cited
references rather than assigned by the authors, and both procedures are defined
on author-assigned keywords. To use them anyway, ask explicitly and say so when
reporting the run:

```python
corpus = load_csv("savedrecs.txt", sep="\t",
                  column_map={"author_keywords": "ID"})
```

`corpus.column_map` shows which source column each canonical field resolved to,
and `corpus.summary()` reports how many records were dropped for missing text
or missing keywords.

## Embedding backends

| Name | Notes |
|---|---|
| `tfidf-svd` | Offline, deterministic, no credentials. Captures lexical overlap only — adequate for Mode A periphery scans and for reproducing a run without keys, but it cannot support the similarity-window criterion of Type 2. |
| `sbert` | Any `sentence-transformers` checkpoint, e.g. `all-mpnet-base-v2`. Local, free, genuinely semantic. |
| `openai` | `text-embedding-3-large` / `-small`. |
| `nomic` | `nomic-embed-text-v1.5` — the model used in the two source publications. |
| `jina`, `cohere` | Hosted alternatives. |
| `precomputed` | Supply your own vectors via `EncoderConfig.extra["vectors"]`. |

API keys are read from the environment variable named in
`EncoderConfig.api_key_env` and are never written to a configuration object, a
result file or a log. Register a custom backend with
`register_encoder(name, factory)`.

## Validation

The validation subpackage reproduces the evaluation design of the two source
papers: a panel of LLM annotators labels the keywords the pipeline selected
and, as a control, the author keywords of the same corpus that it did *not*
select, at `temperature=0`. The control set is disjoint from the candidate set
by construction — comparing a candidate set against a population that contains
it nests the two samples, and no two-sample statistic on them has a defined
null distribution.

```python
from embedresearchgaps.validation import (
    get_annotator, selected_and_control_keywords, validate_gaps,
)

# Example configuration. The exact model identifiers used for the published
# study are listed under "Reproducing the published study" below; model names
# change, so pin them explicitly rather than relying on a provider default.
panel = [
    get_annotator("openai",   model="gpt-4o",            key_env="OPENAI_API_KEY"),
    get_annotator("anthropic",model="claude-sonnet-4-6", key_env="ANTHROPIC_API_KEY"),
    get_annotator("google",   model="gemini-2.5-flash",  key_env="GOOGLE_API_KEY"),
    get_annotator("deepseek", model="deepseek-v4-pro",   key_env="DEEPSEEK_API_KEY"),
    get_annotator("xai",      model="grok-4.5",          key_env="XAI_API_KEY"),
]

selected, control = selected_and_control_keywords(result)
report = validate_gaps(result, panel, config.problem_description,
                       domain="management", selected=selected, all_keywords=control)

print(report.comparison.gap_rate_delta_pp)     # percentage-point gain in GAP rate
print(report.comparison.novelty_increase_pct)  # relative gain in mean Novelty Score
print(report.comparison.novelty_test)          # Welch, Mann-Whitney, permutation, d
print(report.reliability)                      # Fleiss' kappa, Krippendorff's alpha
report.save("validation/")
```

Each keyword receives one of five labels — `GAP`, `EXPLORED`, `LOW IMPORTANCE`,
`METHOD`, `NOISE` — and a Novelty Score from 1 to 10. An annotator that fails
(missing credential, transport error, malformed reply) is recorded in
`report.failures` and skipped; the study continues with the rest and the report
states which models contributed. `CallableAnnotator` wraps any `str -> str`
function, for locally hosted models and for offline tests.

### Reproducing the published study

The article used `text-embedding-3-large` for every embedding and this panel at
`temperature=0`: `openai:gpt-4o`, `anthropic:claude-sonnet-4-6`,
`deepseek:deepseek-v4-pro`, `xai:grok-4.5`, `google:gemini-2.5-flash`.
Moonshot (Kimi) was probed and excluded because its current models reject
`temperature=0`, which would break the determinism the protocol depends on.
The scripts in `examples/` regenerate every reported table and figure; run them
in the order given in `examples/README.md`.

### Expert validation

```python
from embedresearchgaps.validation import build_expert_sheet, write_expert_sheet, \
    load_expert_sheet, expert_report

sheet = build_expert_sheet(result, config.problem_description)   # blinded, shuffled
write_expert_sheet(sheet, "experts.xlsx", config.problem_description, ("expert_1", "expert_2"))

# ... after the reviewers have filled it in ...
experts = load_expert_sheet("experts_completed.xlsx")
print(expert_report(report.annotations, experts))   # Spearman rho, label agreement
```

The sheet withholds gap types, cluster labels and pipeline scores by default and
shuffles the rows, so that the human judgement is independent of the model
output.

## Configuration

Every tunable parameter lives in `RunConfig`, so one serialisable object fully
describes a run; `result.save()` writes it next to the tables.

```python
from embedresearchgaps import RunConfig, GapConfig, ClusteringConfig, EmergingWeights

config = RunConfig(
    top_n_articles=50,
    clustering=ClusteringConfig(k_min=2, k_max=10, silhouette_floor=0.25, random_state=42),
    gaps=GapConfig(
        periphery_percentile=95.0,
        max_keyword_frequency=1,          # None applies the proportional rule n/50
        gap_token_count=2,                # None accepts any keyword length
        combination_similarity_range=(0.25, 0.65),
        combination_similarity_optimum=0.45,
        min_tfidf=0.005,
        max_gaps_per_cluster=3,
        emerging_weights=EmergingWeights(distance=0.25, centrality=0.15,
                                         tfidf=0.30, rarity=0.30),
    ),
    random_state=42,
)
```

Defaults reproduce the settings of the source publications. With `k_selection="auto"`
the number of clusters is taken from the WCSS elbow whenever the best silhouette
stays below `silhouette_floor`, and from the silhouette otherwise.

Runs are deterministic: the same corpus, configuration and seed give identical
tables.

## Stability-aware consensus

The clustering both modes rest on is k-means, and on real bibliographic corpora
its silhouette is small (0.02–0.07 in the case studies). The partition — and
with it the candidate list — therefore depends on the initialisation. A single
seed gives a list whose reproducibility is unknown, so measure it:

```python
from embedresearchgaps import consensus_gaps

consensus = consensus_gaps(corpus, "articles_first", config,
                           seeds=(42, 7, 123, 2026, 31337),
                           top_n=(30, 50, 100, 150),   # pool over corpus size too
                           min_support=0.5)

print(consensus.stability["mean_pairwise_jaccard"])        # overlap across runs
print(consensus.stability["mean_jaccard_same_size"])       # seed effect alone
print(consensus.stability["mean_jaccard_across_sizes"])    # corpus-size effect
print(consensus.gaps[["keyword_display", "gap_type", "support", "mean_score"]])
consensus.save("out/consensus")
```

`support` is the fraction of runs that produced the candidate, and
`at_support(1.0)` keeps only the invariant ones. Reporting a candidate list
without its support is reporting one draw from a distribution. The analysis
corpus size `N` is as arbitrary a choice as the seed and moves the candidate
list more, so pass `top_n=` as well unless you have a reason to fix `N`.

A run whose partition has a silhouette below 0.10 also returns a
`LOW_CLUSTER_SEPARATION` entry in `result.diagnostics["warnings"]` and raises
it as a Python warning: candidates read off a partition that weak should not be
interpreted as a single-run ranking.

## Reproducibility and audit trail

`result.diagnostics` carries everything a methods section needs: cluster sizes
and quality indices (silhouette, Davies-Bouldin, Calinski-Harabasz), the full
k-selection trace, the TF-IDF match-type breakdown, the detected core-domain
terms, the counts before and after cross-cluster deduplication, which clusters
were skipped and why, and the encoder descriptions. Every candidate gap carries
a `rationale` string and the titles and DOIs of the articles it came from.

Determinism is enforced across processes, not only within one. Keyword ordering
follows first appearance in the corpus rather than set iteration, because set
iteration order depends on the interpreter's hash seed and would otherwise
change the row order of the embedding matrix, the k-means++ initialisation and
hence the candidate list between runs. The test suite checks this by running
both modes in subprocesses started with different `PYTHONHASHSEED` values and
requiring identical candidate tables.

Author keywords are matched in lower case but reported in the spelling the
authors used: `corpus.spelling` maps each normalised keyword to its most
frequent source spelling, and every candidate table carries a
`keyword_display` column, so acronyms (`ESG`, `SME`, `FinTech`) survive into
cluster labels and into the reported candidates.

## Method summary

Article ranking uses cosine distance to the embedded problem statement,
`d(u, v) = 1 - cos(u, v)`. Keyword periphery is measured as distance from the
keyword's own cluster centroid, thresholded per cluster so that clusters of
different semantic density are treated on their own scale. The three Mode B
detectors combine min-max normalised components with the published weights:

- Emerging Concept: `0.25·distance + 0.15·centrality + 0.30·tfidf + 0.30·rarity`
- Conceptual Combination: `0.40·similarity + 0.30·frequency + 0.30·tfidf`,
  where the similarity term peaks at cosine 0.45 and vanishes outside
  `(0.25, 0.65)`
- Cross-Cluster Concept: `0.25·frequency + 0.35·specificity + 0.40·tfidf`

Inter-cluster thematic overlap is the pairwise cosine similarity of cluster
centroids.

## Testing

```bash
pip install -e ".[dev]"
pytest --cov=embedresearchgaps
```

The suite runs offline: 315 tests, 93% statement coverage. Detector behaviour is
tested on keyword spaces with designed embeddings placed at exact angles, so the
similarity criteria are checked against the published thresholds rather than
against whatever an embedding model happens to produce. Hosted backends are
tested with a recorded HTTP transport that verifies the request payload,
reply parsing and rate-limit retries of every provider without contacting any
API; the malformed-reply salvage path is tested separately against the
annotation parser. Cross-process determinism is tested with subprocesses under
differing hash seeds.

## Citation

If you use EmbedResearchGaps, please cite the software article and the two
conference papers that established and validated the method:

```bibtex
@article{embedresearchgaps2026,
  title   = {EmbedResearchGaps: embedding-driven identification of research
             gaps in author-keyword space},
  author  = {Matysik, Sebastian and Wisniewska, Joanna and
             Frankowski, Pawel Karol},
  journal = {SoftwareX},
  year    = {2026}
}
```

```bibtex
@inproceedings{frankowski2026pacis,
  title     = {Automated Identification of Research Gaps Using Keyword
               Clustering and an Embedding Model},
  author    = {Frankowski, Pawel Karol and Wi{\'s}niewska, Joanna and
               Matysik, Sebastian},
  booktitle = {PACIS 2026 Proceedings},
  year      = {2026},
  url       = {https://aisel.aisnet.org/pacis2026/adv_theory/adv_theory/7}
}

@inproceedings{frankowski2026amcis,
  title     = {Semantic Periphery Detection in Academic Keyword Space:
               An Embedding-Driven Framework for Automated Research Gap
               Identification},
  author    = {Frankowski, Pawel Karol and Wi{\'s}niewska, Joanna and
               Matysik, Sebastian},
  booktitle = {AMCIS 2026 Proceedings},
  year      = {2026},
  url       = {https://aisel.aisnet.org/amcis2026/ai_aiaa/ai_aiaa/7}
}
```

and, for the screening stage of the review pipeline:

```bibtex
@article{matysik2025embedslr,
  title   = {EmbedSLR: an open-source python framework for efficient
             embedding-based screening and bibliometric validation in
             systematic literature review},
  author  = {Matysik, Sebastian and Wi{\'s}niewska, Joanna and
             Frankowski, Pawel Karol},
  journal = {SoftwareX},
  volume  = {32},
  pages   = {102416},
  year    = {2025},
  doi     = {10.1016/j.softx.2025.102416}
}

@article{matysik2026embedslr2,
  title   = {Version 2.0 -- EmbedSLR: An open-source python framework for
             efficient embedding-based screening and bibliometric validation
             in systematic literature review},
  author  = {Matysik, Sebastian and Wi{\'s}niewska, Joanna and
             Frankowski, Pawel Karol},
  journal = {SoftwareX},
  volume  = {34},
  pages   = {102563},
  year    = {2026},
  doi     = {10.1016/j.softx.2026.102563}
}
```

## License

MIT — see [LICENSE](LICENSE).
