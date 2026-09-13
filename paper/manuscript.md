# EmbedResearchGaps: embedding-driven identification of research gaps in author-keyword space

**Sebastian Matysik** ^a,\*^, **Joanna Wiśniewska** ^b^, **Paweł Karol Frankowski** ^c,d^

^a^ University of Szczecin, Institute of Management, Doctoral School, Szczecin, Poland
^b^ University of Szczecin, Institute of Management, Szczecin, Poland
^c^ Maritime University of Szczecin, Faculty of Computer Science and Telecommunications, Szczecin, Poland
^d^ West Pomeranian University of Technology in Szczecin, Faculty of Electrical Engineering, Szczecin, Poland

\* Corresponding author: sebastian.matysik@usz.edu.pl

## Abstract

EmbedResearchGaps identifies candidate research gaps in the author-keyword
space of a screened literature corpus. It implements two published procedures
as two operating modes: clustering author keywords directly and flagging the
semantic periphery of each cluster, or clustering articles first and then
mining a keyword sub-space inside every article cluster for emerging concepts,
unrealised conceptual combinations and cross-cluster concepts. The package adds
seven interchangeable embedding backends, a five-provider LLM annotation panel,
blinded expert sheets, and a consensus mode that measures how far a candidate
list survives re-initialisation. Two Scopus case studies, in management and in
economics and finance, quantify both what the method delivers and how stable it
is.

**Keywords:** systematic literature review; research gaps; semantic
embeddings; keyword clustering; bibliometrics; research reproducibility

## Metadata

| Nr | Code metadata description | Metadata |
|----|---------------------------|----------|
| C1 | Current code version | v1.0.0 |
| C2 | Permanent link to code/repository used for this code version | https://github.com/s-matysik/EmbedResearchGaps |
| C3 | Legal code license | MIT License |
| C4 | Code versioning system used | git |
| C5 | Software code languages, tools and services used | Python (>= 3.10); numpy, pandas, scipy, scikit-learn, matplotlib, networkx, kneed, typer; optional sentence-transformers, krippendorff, openpyxl; optional HTTP access to the OpenAI, Nomic, Jina and Cohere embedding endpoints and to the OpenAI, Anthropic, Google, DeepSeek and xAI chat endpoints |
| C6 | Compilation requirements, operating environments and dependencies | Pure Python, no compilation: `pip install embedresearchgaps`. Runs fully offline with the built-in `tfidf-svd` backend; remote backends read API keys from named environment variables |
| C7 | If available, link to developer documentation/manual | https://github.com/s-matysik/EmbedResearchGaps#readme |
| C8 | Support email for questions | sebastian.matysik@usz.edu.pl |

## 1. Motivation and significance

Formulating a research gap is the step of a systematic literature review (SLR)
that has resisted automation most stubbornly. Screening has not: EmbedSLR ranks
candidate articles by the cosine distance between an embedded research question
and embedded titles and abstracts, and validates the resulting set
bibliometrically [1], with a multi-model consensus procedure added in version
2.0 [2]. Once a reviewer holds a screened corpus, however, the question changes
from *which papers belong here* to *what do these papers not talk about*, and
the usual answer is an unaided reading of discussion sections.

Author keywords are an attractive substrate for that second question. They are
assigned by the authors themselves, they are short enough to embed without
truncation, and their co-occurrence structure underpins a large bibliometric
literature. Two procedures for turning that structure into candidate gaps were
introduced recently. The first clusters the author keywords of a screened
corpus and treats the semantic periphery of each keyword cluster as the
interesting region, on the argument that a term far from its own cluster
centroid is attached to the field only loosely [3]. The second clusters the
articles first, labels each cluster, and then mines a keyword sub-space inside
every cluster with three detectors: rare and lexically salient terms (emerging
concepts), pairs of established terms that never co-occur (conceptual
combinations), and terms established elsewhere in the corpus but absent from
this cluster (cross-cluster concepts) [4]. Both were validated with a panel of
large language models annotating keywords against a calibrated rubric, and the
second additionally against two human experts.

Neither procedure was released as software. Both existed as notebooks in which
model choice, thresholds and API credentials were interleaved with the
analysis, which makes a published run impossible to repeat and a new corpus
expensive to analyse. EmbedResearchGaps turns them into one library with a
single configuration object, interchangeable embedding backends, and a
validation layer that is part of the package rather than an appendix to it. It
also adds what the prototypes lacked: a measurement of how far a candidate
list survives re-initialisation of the clustering it rests on.

The intended user is a reviewer who has already screened a corpus — with
EmbedSLR, with a database query, or by hand — and who wants a ranked, auditable
list of directions the corpus leaves open, together with the evidence behind
each one. The workflow is: export from Scopus or Web of Science, load, state
the research problem in one or two sentences, choose an embedding backend, run
one or both modes, and read the candidate table. Every candidate carries the
articles it came from, the metric values that produced its score, and a
one-line rationale.

## 2. Software description

### 2.1 Software architecture

Figure 1 shows the architecture. A shared front end loads a bibliographic
export, resolves the column names of the Scopus and Web of Science
conventions, splits and normalises author keywords, embeds the texts with the
selected backend and — when a research problem is supplied — ranks the records
by cosine distance to it and truncates the corpus to the *N* nearest. Two modes
then operate on that analysis corpus, and a shared back end handles consensus,
validation and output.

The package holds twelve top-level modules and a `validation` subpackage of
five. `config` defines the dataclasses that carry every threshold, weight and
seed; `corpus` loads and normalises records; `text` holds keyword
normalisation, the methodological-term filter and corpus-level TF-IDF
salience; `encoders` implements the embedding contract; `clustering` performs
semantic ranking, k-selection and k-means with per-cluster centroid distances;
`keywords` builds the keyword space; `gaps` contains the four detectors and the
three scoring formulae; `results` is the result container and its writers;
`pipelines` assembles the two modes; `consensus` repeats a mode across
initialisations; `viz` renders figures; `cli` exposes the command line. The
subpackage contains the annotation protocol, the annotator backends, the
agreement metrics, the study runner and the expert sheets.

Runs are deterministic: one `RunConfig` object fixes every threshold, weight
and seed, and `PipelineResult.save()` writes that object next to the tables, so
a published result can be regenerated from the artefacts alone.

### 2.2 Software functionalities

**Two operating modes.** `keywords_first` embeds the unique author keywords of
the analysis corpus, clusters them, and flags keywords whose distance from
their own cluster centroid exceeds a cluster-specific percentile (95 by
default). Because the threshold is computed inside each cluster, clusters of
different semantic density are judged on their own scale. `articles_first`
clusters the articles, labels each cluster from its most frequent non-core
keywords, builds a keyword sub-space inside every cluster, and applies the
three detectors of the typology, each with its own composite score. Both modes
share the filters that remove terms peripheral for uninteresting reasons: a
frequency ceiling, a token-count rule, a methodological blocklist of some fifty
phrases and acronyms, and a core-domain filter that drops the terms defining
the field itself.

**Seven embedding backends.** `tfidf-svd` is offline, deterministic and needs
no credentials; `sbert` runs any sentence-transformers checkpoint locally;
`openai`, `nomic`, `jina` and `cohere` call hosted endpoints; `precomputed`
accepts user-supplied vectors, which is how a published run is repeated
without re-embedding. All conform to one contract that validates the returned
row count and rejects non-finite vectors, so a silent misalignment between a
keyword and its embedding cannot occur. API keys are read from named
environment variables and never enter a configuration object, a result file or
a log.

**Stability-aware consensus.** `consensus_gaps` repeats a mode across seeds and
reports, for every candidate, the fraction of seeds that produced it, together
with the pairwise overlap of the candidate sets. Filtering on that support
turns a single-seed list into a ranking whose reproducibility is measured
rather than assumed; Section 3 shows why this is necessary.

**Validation.** The `validation` subpackage implements the annotation protocol
of the source papers: a keyword is labelled `GAP`, `EXPLORED`, `LOW
IMPORTANCE`, `METHOD` or `NOISE` and given a Novelty Score from 1 to 10 against
a rubric supplied verbatim in the prompt. Six chat providers are registered,
all called at temperature 0 where the model accepts it; `CallableAnnotator`
wraps any local function, and a failing annotator is recorded rather than
allowed to abort the study. The module computes label rates, Novelty
distributions, Fleiss' kappa, Krippendorff's alpha, pairwise annotator
agreement, and the comparison between the selected keywords and a control set
with a Welch t-test, a Mann-Whitney U test and Cohen's d. Blinded expert
sheets — rows shuffled, gap types and scores withheld — are generated as an
Excel workbook and read back into a Spearman concordance against the model
panel.

**Figures and diagnostics.** Seven figures are produced head-lessly: keyword
and article t-SNE maps with candidates outlined, the co-occurrence network, the
inter-cluster centroid-similarity heatmap, the candidate summary, the
per-cluster year distribution and the k-selection trace. Every run also returns
diagnostics: cluster sizes and quality indices, the full k-selection trace, the
TF-IDF match-type breakdown, the detected core-domain terms, deduplication
counts, skipped clusters and the encoder descriptions.

![**Fig. 1.** Architecture. A shared front end prepares the analysis corpus; two modes mine it; a shared back end measures stability, validates candidates and writes outputs.](figures/fig1_architecture.png)

## 3. Illustrative examples

Two corpora were retrieved from Scopus, restricted to English-language journal
articles in the business, economics and decision-science subject areas, and
analysed with `text-embedding-3-large`, *N* = 50 and otherwise default
settings. The management corpus holds 493 records matching *dynamic
capabilit\** and *digital transformation*, analysed against the problem
statement "How do dynamic capabilities enable the digital transformation of
established firms, and which organisational mechanisms mediate the
relationship between the adoption of digital technologies and firm
performance?". The economics and finance corpus holds 487 records matching
*fintech* or *financial technology* together with *financial inclusion*,
*credit risk* or *bank lending*, analysed against "How does fintech credit
affect financial inclusion and credit risk in banking systems, and which
mechanisms link digital lending technologies to borrower and bank outcomes?".
A third corpus of 290 records on gamification in marketing was analysed as a
conformance check against the topic of the source publications. Every table,
figure and diagnostic reported here is regenerated by the scripts in the
repository.

### 3.1 What the modes return

On the management corpus, k-selection chose *k* = 6 for both modes: the elbow
rule applied in both cases, because the best silhouette over the candidate
range (0.041 for the article space, 0.065 for the keyword space) fell below the
0.25 floor. From the 146 unique author keywords of the analysis corpus, Mode B
returned 27 candidates — 18 emerging concepts, 7 cross-cluster concepts and 2
conceptual combinations — and Mode A returned 6 peripheral keywords, 4.1 % of
its keyword space. Figure 2 shows the
keyword space with candidates outlined. The highest-ranked candidates are
single-study terms such as *Agrifood businesses*, *Automotive industry* and *DT
vision*, and the sole high-ranked combination pairs *Dynamic capabilities* with
*Industry 4.0*: two established terms of the same cluster that never co-occur
in one record.

On the economics and finance corpus (168 unique author keywords, *k* = 6 for
both modes), Mode B returned 27 candidates (16 emerging, 7 cross-cluster and 4
combinations) and Mode A returned 4, 2.4 % of its keyword space. Figure 3 shows
the type distribution and the
twelve highest-ranked candidates, headed by *South Asia*, *Soft information*
and *Regulatory resources*, with *Credit risk × Financial inclusion* as the
strongest combination. Figure 6 shows that the six article clusters of each
corpus overlap thematically without collapsing into one another.

The type distribution is dominated by emerging concepts in every corpus, which
reproduces the pattern reported for the articles-first procedure [4]. The
candidates themselves are a mixture of genuine research directions, contextual
scope terms (*South Asia*, *Sub-Saharan Africa*) and — in one management
record — two personal names that a metadata error had placed in the
author-keyword field, neither of them the article's author. The annotation
panel labelled those names
`NOISE` with a Novelty Score of 1, which is the behaviour a validation layer
exists to produce: the pipeline surfaces what the metadata contains, and the
layer above it catches what should not have been there.

![**Fig. 2.** Author-keyword space of the management corpus (t-SNE of the embeddings, 146 keywords). Marker colour is the article cluster a keyword is most often assigned to, marker area is document frequency, and black outlines mark the 26 Mode B candidates; the eight highest-scoring are labelled.](figures/fig2_keyword_space.png)

![**Fig. 3.** Economics and finance corpus, Mode B. Left: candidates per type. Right: the twelve highest-ranked candidates, coloured by type; scores are comparable within a type, not across types.](figures/fig3_finance_gaps.png)

### 3.2 How stable the candidate lists are

Figure 5 quantifies stability. Re-running a mode with five k-means seeds and
nothing else changed, the candidate sets overlap by a mean pairwise Jaccard
index of 0.44 (Mode B) and 0.14 (Mode A) on the management corpus, and 0.38 and
0.15 on the economics and finance corpus; the worst seed pair of Mode A shares
no candidate at all. Changing the analysis-corpus size from *N* = 50 to
*N* = 30, 100 or 150 leaves overlaps of 0.09 to 0.31 against the *N* = 50
reference for Mode B, and 0.00 to 0.38 for Mode A — the zero being the
economics and finance corpus at *N* = 150, whose eight candidates have nothing
in common with the four found at *N* = 50. A single-seed, single-*N* candidate
list is therefore
one draw from a broad distribution, not a property of the corpus. This is the
reason `consensus_gaps` exists and the reason we report support alongside every
candidate.

![**Fig. 5.** Stability of the candidate list. Left: Jaccard overlap with the N = 50 candidate list when the analysis corpus is resized, averaged over the two case studies. Right: mean pairwise Jaccard overlap across five k-means seeds with everything else held fixed; whiskers span the minimum and maximum seed pair.](figures/fig5_stability.png)

### 3.3 What the LLM panel says

Five providers (GPT-4o, Claude Sonnet 4.6, DeepSeek V4 Pro, Grok 4.5 and
Gemini 2.5 Flash) annotated every candidate and every author keyword of the
same analysis corpus at temperature 0 — between 758 and 990 judgements per
corpus and mode. Inter-annotator reliability on the candidate sets was moderate
(Fleiss' κ 0.45–0.70; Krippendorff's α on the Novelty Score 0.38–0.85). A
leave-out check, in which two models are held out and their mean Novelty Score
correlated against that of the remaining three, gave Spearman ρ between 0.64
and 0.83 on the Mode B candidate sets; on Mode A the coefficient rests on four
to six keywords and ranges from 0.20 to 1.00, which is too little data to
interpret. The panel is therefore internally consistent enough to be
informative on Mode B and inconclusive on Mode A.

Against that control, the result is a null one (Figure 4, left). The GAP rate
of Mode B candidates differs from the GAP rate of all author keywords by
−1.0 pp on management, +10.4 pp on economics and finance and −8.3 pp on
gamification; Mode A differs by −7.1, −9.3 and −3.7 pp. Mean Novelty Scores
move by −7.5 % to +8.6 % (Mode B) and −12.2 % to +3.5 % (Mode A). No
comparison reaches significance (Welch *p* = 0.12–0.80, |*d*| ≤ 0.20), and the
signs are not consistent across corpora.

A frequency-matched arm explains why. Comparing candidates not against all
keywords but against the *eligible pool* — the keywords that pass every
lexical and frequency filter and could have been selected — separates the two
things the pipeline does at once. The filters alone raise the GAP rate by
+5.1 pp (management) and +3.6 pp (finance) and mean novelty by +5.5 % and
0.0 %; the embedding-based ranking applied on top of them shifts the GAP rate
by −6.1 pp and +6.8 pp and novelty by −8.4 % and +8.6 % (Figure 4, right; all
*p* ≥ 0.15). On these corpora the lexical filters carry the effect, and the
semantic periphery criterion adds nothing the panel can detect. We report this
as the measurement it is: an independent replication on new corpora, in a
domain different from the source publications, recovers the machinery and the
type distribution but not an enrichment effect.

![**Fig. 4.** LLM panel validation, five providers at temperature 0. Left: GAP rate of the candidate set minus the GAP rate of all author keywords of the same analysis corpus, in percentage points, with the Welch p-value of the accompanying Novelty Score test (Mode B n = 24-26 candidates, Mode A n = 4-6). Right: mean Novelty Score across the three arms of the frequency-matched comparison; whiskers are standard errors of the mean.](figures/fig4_validation.png)

![**Fig. 6.** Pairwise cosine similarity of the six article-cluster centroids in each case study, the inter-cluster overlap measure of the source procedure.](figures/fig6_centroids.png)

## 4. Impact

The immediate impact is that two published procedures become executable. A
reviewer who wants to apply either one now installs a package, writes ten
lines, and obtains a candidate table with a rationale and source articles for
every row, instead of transcribing a notebook and re-entering credentials. The
`precomputed` backend plus the saved `RunConfig` make a published run
repeatable at zero embedding cost, which is what turns a gap list into
evidence a reviewer can be asked about.

The methodological impact is the measurement the prototypes did not report.
Because EmbedResearchGaps makes the seed an explicit parameter and pools across
it, the instability documented in Section 3.2 becomes visible and reportable
rather than hidden behind a default. Any future work that ranks candidates off
a k-means partition of an author-keyword space now has both a baseline to beat
and an instrument that measures whether it beat it. The frequency-matched
validation arm is the same kind of instrument for effect attribution: it is the
comparison that distinguishes "our filters are sensible" from "our ranking
works", and it is available to anyone extending either procedure.

For the wider SLR toolchain, the package closes the loop opened by EmbedSLR
[1,2]: embedding-based screening decides which papers enter a review, and
EmbedResearchGaps asks what the accepted papers leave unsaid, with the same
determinism guarantees and the same audit trail. The validation subpackage is
reusable on its own — any keyword list, whatever produced it, can be put
through the five-model panel and the blinded expert sheets.

## 5. Conclusions

EmbedResearchGaps implements, tests and documents two embedding-based
procedures for identifying candidate research gaps in author-keyword space,
with interchangeable embedding backends, a five-provider validation panel,
expert sheets, and cross-process determinism enforced by the test suite (289
tests, 93 % statement coverage).

Applied to two new Scopus corpora, the software reproduces the mechanics and
the candidate-type distribution of the published procedures, and shows two
limits that users should plan around: candidate lists are sensitive to the
k-means initialisation and to the analysis-corpus size, and on these corpora
the semantic ranking adds no novelty enrichment beyond what the lexical and
frequency filters already provide. We therefore recommend running the
articles-first mode through `consensus_gaps` across at least five seeds,
reporting support alongside every candidate, and treating the output as a
generator of hypotheses for expert screening rather than as a ranking to be
read off. Work in progress extends the consensus idea across embedding models
as well as seeds, and replaces k-means with a density-based partition whose
stability does not depend on an initialisation.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or
personal relationships that could have appeared to influence the work reported
in this paper.

## References

[1] S. Matysik, J. Wiśniewska, P.K. Frankowski, EmbedSLR: an open-source
python framework for efficient embedding-based screening and bibliometric
validation in systematic literature review, SoftwareX 32 (2025) 102416.
https://doi.org/10.1016/j.softx.2025.102416

[2] S. Matysik, J. Wiśniewska, P.K. Frankowski, Version 2.0 — EmbedSLR: an
open-source python framework for efficient embedding-based screening and
bibliometric validation in systematic literature review, SoftwareX 34 (2026)
102563. https://doi.org/10.1016/j.softx.2026.102563

[3] P.K. Frankowski, J. Wiśniewska, S. Matysik, Automated identification of
research gaps using keyword clustering and an embedding model, PACIS 2026
Proceedings 7. https://aisel.aisnet.org/pacis2026/adv_theory/adv_theory/7

[4] P.K. Frankowski, J. Wiśniewska, S. Matysik, Semantic periphery detection in
academic keyword space: an embedding-driven framework for automated research
gap identification, AMCIS 2026 Proceedings 7.
https://aisel.aisnet.org/amcis2026/ai_aiaa/ai_aiaa/7

[5] S. Matysik, J. Wiśniewska, P.K. Frankowski, EmbedResearchGaps v1.0.0
(software), 2026. https://github.com/s-matysik/EmbedResearchGaps
