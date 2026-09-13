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
blinded expert sheets, and a consensus mode that pools candidates across
k-means seeds and analysis-corpus sizes. In two Scopus case studies candidate
sets proved sensitive to both, and the embedding-based ranking gave no reliable
novelty enrichment beyond lexical and frequency filtering. The software
therefore reports candidate support explicitly and treats its output as
hypotheses for expert assessment.

**Keywords:** systematic literature review; research gaps; semantic
embeddings; keyword clustering; bibliometrics; research reproducibility

## Metadata

| Nr | Code metadata description | Metadata |
|----|---------------------------|----------|
| C1 | Current code version | v1.1.1 |
| C2 | Permanent link to code/repository used for this code version | https://github.com/s-matysik/EmbedResearchGaps/releases/tag/v1.1.1 |
| C3 | Legal code license | MIT License |
| C4 | Code versioning system used | git |
| C5 | Software code languages, tools and services used | Python (>= 3.10); numpy, pandas, scipy, scikit-learn, matplotlib, networkx, kneed, typer; optional sentence-transformers, krippendorff, openpyxl; optional HTTP access to the OpenAI, Nomic, Jina and Cohere embedding endpoints and to the OpenAI, Anthropic, Google, DeepSeek and xAI chat endpoints |
| C6 | Compilation requirements, operating environments and dependencies | Pure Python 3.10 or newer, no compilation: `pip install embedresearchgaps` (or `pip install -e .` from the repository). Runs fully offline with the built-in `tfidf-svd` backend; remote backends read API keys from named environment variables |
| C7 | If available, link to developer documentation/manual | https://github.com/s-matysik/EmbedResearchGaps#readme |
| C8 | Support email for questions | sebastian.matysik@usz.edu.pl |

## 1. Motivation and significance

Formulating a research gap is the step of a systematic literature review (SLR)
that has resisted automation most stubbornly. It is normally done by expert
reading [6], and the categories a gap can fall into have been formalised for
systematic reviews [7] without the identification itself becoming mechanical;
the organisational-research literature goes further and argues that
gap-spotting is a weak way of constructing a research question when it is not
disciplined by evidence [8]. Reporting standards specify how a review must be
documented [9] but say nothing about how its research agenda is arrived at.
Screening, by contrast, has been automated repeatedly, by active-learning
frameworks [10] and by embedding-based ranking: EmbedSLR ranks
candidate articles by the cosine distance between an embedded research question
and embedded titles and abstracts, and validates the resulting set
bibliometrically [1], with a multi-model consensus procedure added in version
2.0 [2]. Once a reviewer holds a screened corpus, however, the question changes
from *which papers belong here* to *what do these papers not talk about*, and
the usual answer is an unaided reading of discussion sections.

Author keywords are an attractive substrate for that second question. They are
assigned by the authors themselves, they are short enough to embed without
truncation, and their co-occurrence structure underpins a large bibliometric
literature with established tooling — VOSviewer [11], bibliometrix [12] and
CiteSpace [13] — and established reporting conventions [14]. That tooling
visualises structure and leaves the reading to the analyst; work that goes
further and detects emerging topics does so from citation and term-burst
signals rather than from a semantic space [15]. Substituting contextual
sentence embeddings [16,17] for term co-occurrence is what makes the semantic
neighbourhood of a keyword computable, and embedding-based clustering has been
shown to separate short texts more coherently than bag-of-words
representations [18]. Two procedures for turning that structure into candidate
gaps were introduced recently. The first clusters the author keywords of a screened
corpus and treats the semantic periphery of each keyword cluster as the
interesting region, on the argument that a term far from its own cluster
centroid is attached to the field only loosely [3]. The second clusters the
articles first, labels each cluster, and then mines a keyword sub-space inside
every cluster with three detectors: rare and lexically salient terms (emerging
concepts), pairs of established terms that never co-occur (conceptual
combinations), and terms established elsewhere in the corpus but absent from
this cluster (cross-cluster concepts) [4]. Both were validated with a panel of
large language models annotating keywords against a calibrated rubric — a
design supported by evidence that language models match or exceed human coders
on well-defined annotation tasks [19–21] — and the second additionally against
two human experts [4].

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

**Stability-aware consensus.** `consensus_gaps` repeats a mode across k-means
seeds and, optionally, across analysis-corpus sizes, and reports for every
candidate the fraction of runs that produced it, together with the pairwise
overlap of the candidate sets — decomposed into the overlap attributable to
the seed alone and the overlap across corpus sizes. Filtering on that support
turns a single-run list into a ranking whose reproducibility is measured
rather than assumed; Section 3 shows why this is necessary. Relatedly, a run
whose partition has a silhouette [22] below 0.10 returns a
`LOW_CLUSTER_SEPARATION` warning naming the measured value, both in the
diagnostics and as a Python warning, because a candidate list read off a
partition that weak should not be interpreted as a single-run ranking.

**Validation.** The `validation` subpackage implements the annotation protocol
of the source papers: a keyword is labelled `GAP`, `EXPLORED`, `LOW
IMPORTANCE`, `METHOD` or `NOISE` and given a Novelty Score from 1 to 10 against
a rubric supplied verbatim in the prompt. Six chat providers are registered and
called at temperature 0 where the model accepts it; the study of Section 3.3
uses five of them. `CallableAnnotator` wraps any local function, and a failing
annotator is recorded rather than allowed to abort the study. The module
computes label rates, Novelty distributions, Fleiss' kappa, Krippendorff's
alpha, pairwise annotator agreement, and the comparison between the candidate
set and a control set with a Welch t-test, a Mann-Whitney U test, Cohen's d
and a permutation test on the difference of means. `selected_and_control_keywords`
builds a control set *disjoint* from the candidates, and `compare_keyword_sets`
refuses to report parametric p-values when the two arms it is handed share a
keyword, because a two-sample statistic on nested samples has no defined null
distribution. Blinded expert
sheets — rows shuffled, gap types and scores withheld — are generated as an
Excel workbook and read back into a Spearman concordance against the model
panel.

**Database-agnostic input.** `load_csv` resolves canonical fields through an
alias table covering the Scopus CSV export, the Scopus Search API and both Web
of Science export flavours — the tab-delimited file with two-letter field tags
(`TI`, `AB`, `DE`, `PY`, `TC`) and the full-record file with spelled-out
headers (`Article Title`, `Author Keywords`, `Times Cited, All Databases`) —
so neither needs a column map; any other export is accommodated by passing
one. Web of Science `Keywords Plus` is deliberately not treated as author
keywords, because those terms are assigned by the database rather than by the
authors and both procedures are defined on author-assigned keywords; a user who
wants them must ask for them explicitly. The test suite loads all three export
flavours and asserts that they yield an identical corpus.

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
strongest combination. Figure 4 shows that the six article clusters of each
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

![**Fig. 4.** Pairwise cosine similarity of the six article-cluster centroids in each case study, the inter-cluster overlap measure of the source procedure.](figures/fig4_centroids.png)

### 3.2 How stable the candidate lists are

Figure 5 quantifies stability. Re-running a mode with five k-means seeds and
nothing else changed, the candidate sets overlap by a mean pairwise Jaccard
index of 0.44 (Mode B) and 0.14 (Mode A) on the management corpus, and 0.38 and
0.15 on the economics and finance corpus; the worst seed pair of Mode A shares
no candidate at all. Changing the analysis-corpus size from *N* = 50 to
*N* = 30, 100 or 150 leaves overlaps of 0.09 to 0.31 against the *N* = 50
reference for Mode B, and 0.00 to 0.38 for Mode A — the zero being the
economics and finance corpus at *N* = 150, whose eight candidates have nothing
in common with the four found at *N* = 50. Because *N* is as arbitrary a choice
as the seed, `consensus_gaps` pools over both: over five seeds and four corpus
sizes, twenty runs per mode, the mean pairwise overlap is 0.23 (management,
Mode B) and 0.19 (economics and finance, Mode B), and the decomposition
separates the two sources — 0.45 and 0.39 between runs that share a corpus
size, against 0.18 and 0.14 between runs that do not. The corpus size moves
the candidate list more than the seed does.

Pooling is what makes a list reportable. Of 161 distinct candidates produced
across the twenty management Mode B runs, 12 reach majority support and 2 are
unanimous; for economics and finance the figures are 166, 11 and 2. Mode A
pools to 50 and 37 candidates with 2 and 1 at majority support and none
unanimous, which is a quantitative statement that its single-run output should
not be read as a ranking at all. High support is not validity, though: the two
personal names that a metadata error left in the management keyword field
reach support of 0.85 and 0.75, because a data defect is perfectly
reproducible. Stability and correctness are separate questions, which is why
the package ships the validation layer as well as the consensus mode.

![**Fig. 5.** Stability of the candidate list. Left: Jaccard overlap with the N = 50 candidate list when the analysis corpus is resized, shown separately for the two modes and averaged over the two case studies. Right: mean pairwise Jaccard overlap across five k-means seeds with everything else held fixed; whiskers span the minimum and maximum seed pair.](figures/fig5_stability.png)

### 3.3 What the LLM panel says

Five providers (GPT-4o, Claude Sonnet 4.6, DeepSeek V4 Pro, Grok 4.5 and
Gemini 2.5 Flash) annotated every candidate and every author keyword of the
same analysis corpus at temperature 0 — between 758 and 990 judgements per
corpus and mode. The control arm is the keyword population *minus* the
candidates: the two arms must be disjoint, or the candidates contribute to
both and no two-sample statistic on them has a defined null distribution.
Inter-annotator reliability on the candidate sets was moderate (Fleiss' κ
0.45–0.70; Krippendorff's α on the Novelty Score 0.38–0.85). A leave-out
check, in which two models are held out and their mean Novelty Score
correlated against that of the remaining three, gave Spearman ρ between 0.64
and 0.83 on the Mode B candidate sets; on Mode A the coefficient rests on four
to six keywords and ranges from 0.20 to 1.00, which is too little data to
interpret. This is a model-against-model check that exercises the same
concordance code path as an expert study, not a substitute for one.

Against that control (Figure 6, left) the six comparisons point in different
directions. Mode B candidates are labelled `GAP` more often than control
keywords on the economics and finance corpus (+12.7 pp, mean Novelty Score
+12.5 %, Welch *p* = 0.032, Mann-Whitney *p* = 0.014, permutation *p* = 0.046,
*d* = 0.21) and less often on management (−1.2 pp) and gamification
(−9.1 pp); Mode A is negative on all three (−8.0, −9.3, −3.5 pp). One
nominally significant result out of six pre-planned comparisons is what chance
produces: under Holm correction across the six, nothing survives (smallest
adjusted *p* = 0.19). We therefore read the panel as showing no reliable
novelty enrichment, with a single corpus-and-mode combination worth
re-testing on an independent corpus rather than a demonstrated effect.

The model panel is not the only evidence on this method, and the earlier
evidence is more favourable. The AMCIS study validated its candidates against
two human experts as well as the panel: an associate professor and a full
professor, each with more than twenty years in the field of the corpus,
independently scored the sixteen candidates the pipeline produced. Three of
them — *social commerce*, *time poverty* and *psychological distance* — were
labelled `GAP` unanimously by all five models and corroborated as gaps by at
least one expert, and expert mean Novelty Scores correlated with the panel's at
Spearman ρ = 0.61 (*p* = 0.012, *n* = 16) [4]. That study also found the spread
of expert opinion wider than the spread across models, and described its expert
arm as a pilot to be replaced by a larger calibrated study. A sixteen-keyword
pilot on one marketing corpus and a panel study on three corpora in two other
domains are different measurements, and the disagreement between them is what
that larger expert study is needed to resolve. We did not repeat it — an expert
study is a study of people, not of software — so the package ships the
instrument instead: `build_expert_sheet` produces blinded, row-shuffled
workbooks and `expert_report` computes the same concordance against the panel.
The sheets for all three corpora are in the supplementary material.

A filter-eligible comparison locates where what little signal there is comes
from. The *eligible pool* is every keyword that passes the frequency ceiling,
the token-count rule, the methodological blocklist and the core-domain filter
but that the ranking did not select; the remaining keywords fail at least one
filter. Comparing the eligible pool against those failing keywords isolates
the filters, and comparing the candidates against the eligible pool isolates
the embedding-based ranking applied on top of them (Figure 6, right). The
filters raise the GAP rate by 7.8 pp on management and 10.6 pp on economics and
finance, and mean novelty by 8.3 % and 9.2 % (permutation *p* = 0.149 and
0.069) — consistent in sign and of similar size on both corpora. The ranking
then moves the GAP rate by −6.1 pp and +6.8 pp and novelty by −8.4 % and
+8.6 % (permutation *p* = 0.192 and 0.150) — inconsistent in sign. On these
corpora the lexical and frequency filters carry whatever enrichment exists,
and the semantic periphery criterion adds nothing the panel can detect. That
is the measurement an independent replication produced, in a domain different
from the source publications, with the machinery and the candidate-type
distribution recovered but the effect not.

![**Fig. 6.** LLM panel validation, five providers at temperature 0. Left: GAP rate of the candidate set minus that of the control set — the author keywords of the same analysis corpus that the pipeline did not select — in percentage points, annotated with the permutation p-value of the accompanying Novelty Score test (Mode B n = 24-26 candidates against 120-146 control keywords, Mode A n = 4-6 against 140-166). Right: mean Novelty Score across the three disjoint arms that separate the contribution of the filters from that of the ranking; whiskers are standard errors of the mean.](figures/fig6_validation.png)

## 4. Impact

The immediate impact is that two published procedures become executable. A
reviewer who wants to apply either one now installs a package, writes ten
lines, and obtains a candidate table with a rationale and source articles for
every row, instead of transcribing a notebook and re-entering credentials. The
`precomputed` backend plus the saved `RunConfig` make a published run
repeatable at zero embedding cost, which is what turns a gap list into
evidence a reviewer can be asked about. The release described here is archived
as v1.1.1 [5].

The methodological impact is the measurement the prototypes did not report.
Because EmbedResearchGaps makes both the seed and the corpus size explicit
parameters and pools across them, the instability documented in Section 3.2
becomes visible and reportable rather than hidden behind a default, and the
`LOW_CLUSTER_SEPARATION` warning tells a user when their partition is too weak
to read a single-run ranking off. Any future work that ranks candidates from a
k-means partition of an author-keyword space now has both a baseline to beat
and an instrument that measures whether it beat it. The filter-eligible
comparison of Section 3.3 is the same kind of instrument for effect
attribution: it is what distinguishes "our filters are sensible" from "our
ranking works", and it is available to anyone extending either procedure.

This positioning matters for how the output should be used. Because the
ranking adds no detectable enrichment over the filters and the lists move with
the initialisation, a candidate table from this package is a generator of
hypotheses for expert screening, not an inventory of established omissions.
The software is built for that use: every candidate carries a rationale, its
source articles and its support across runs, and the validation subpackage and
the blinded expert sheets exist so that the screening step has instruments
too.

EmbedResearchGaps is distinct from EmbedSLR [1,2] in input, algorithm and
output, and the two are complementary rather than incremental: EmbedSLR takes
a query and a candidate pool and decides which articles enter a review by
ranking full-text-level embeddings against inclusion criteria; EmbedResearchGaps
takes an already-screened corpus and mines the *author-keyword* space of the
accepted articles for what they do not discuss, returning typed candidates with
stability support rather than an inclusion decision. They share only the idea
that embeddings encode topical proximity. The package is likewise not a
bibliometric mapping tool: VOSviewer and bibliometrix visualise
co-occurrence and co-citation structure and leave interpretation to the reader,
whereas EmbedResearchGaps applies explicit, parameterised criteria to produce a
ranked, auditable candidate list with a measured reproducibility — and reports
when that reproducibility is poor.

For the wider toolchain, the validation subpackage is reusable on its own: any
keyword list, whatever produced it, can be put through the five-model panel,
the disjoint-control comparison and the blinded expert sheets.

## 5. Conclusions

EmbedResearchGaps implements, tests and documents two embedding-based
procedures for identifying candidate research gaps in author-keyword space,
with interchangeable embedding backends, a five-provider validation panel,
expert sheets, and cross-process determinism enforced by the test suite (306
tests, 93 % statement coverage).

Applied to two new Scopus corpora, the software reproduces the mechanics and
the candidate-type distribution of the published procedures, and shows two
limits that users should plan around: candidate lists are sensitive both to the
k-means initialisation and to the analysis-corpus size, and the semantic
ranking adds no reliable novelty enrichment beyond what the lexical and
frequency filters already provide. We therefore recommend running the
articles-first mode through `consensus_gaps` across at least five seeds and
several corpus sizes, reporting support alongside every candidate, and treating
the output as a generator of hypotheses for expert screening rather than as a
ranking to be read off. The package is built to be used that way, and reports
the diagnostics needed to tell when it should not be trusted further. Work in
progress extends the consensus idea across embedding models as well as seeds
and sizes, and replaces k-means with a density-based partition whose stability
does not depend on an initialisation.

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

[5] S. Matysik, J. Wiśniewska, P.K. Frankowski, EmbedResearchGaps v1.1.1
(software), 2026. https://github.com/s-matysik/EmbedResearchGaps

[6] M.J. Westgate, P.S. Barton, J.C. Pierson, D.B. Lindenmayer, Text analysis
tools for identification of emerging topics and research gaps in conservation
science, Conservation Biology 29 (2015) 1606–1614.
https://doi.org/10.1111/cobi.12605

[7] K.A. Robinson, I.J. Saldanha, N.A. McKoy, Development of a framework to
identify research gaps from systematic reviews, Journal of Clinical
Epidemiology 64 (2011) 1325–1330. https://doi.org/10.1016/j.jclinepi.2011.06.009

[8] J. Sandberg, M. Alvesson, Ways of constructing research questions:
gap-spotting or problematization?, Organization 18 (2011) 23–44.
https://doi.org/10.1177/1350508410372151

[9] M.J. Page, J.E. McKenzie, P.M. Bossuyt, I. Boutron, T.C. Hoffmann, C.D.
Mulrow, et al., The PRISMA 2020 statement: an updated guideline for reporting
systematic reviews, BMJ 372 (2021) n71. https://doi.org/10.1136/bmj.n71

[10] R. van de Schoot, J. de Bruin, R. Schram, P. Zahedi, J. de Boer, F.
Weijdema, et al., An open source machine learning framework for efficient and
transparent systematic reviews, Nature Machine Intelligence 3 (2021) 125–133.
https://doi.org/10.1038/s42256-020-00287-7

[11] N.J. van Eck, L. Waltman, Software survey: VOSviewer, a computer program
for bibliometric mapping, Scientometrics 84 (2010) 523–538.
https://doi.org/10.1007/s11192-009-0146-3

[12] M. Aria, C. Cuccurullo, bibliometrix: an R-tool for comprehensive science
mapping analysis, Journal of Informetrics 11 (2017) 959–975.
https://doi.org/10.1016/j.joi.2017.08.007

[13] C. Chen, CiteSpace II: detecting and visualizing emerging trends and
transient patterns in scientific literature, Journal of the American Society
for Information Science and Technology 57 (2006) 359–377.
https://doi.org/10.1002/asi.20317

[14] N. Donthu, S. Kumar, D. Mukherjee, N. Pandey, W.M. Lim, How to conduct a
bibliometric analysis: an overview and guidelines, Journal of Business Research
133 (2021) 285–296. https://doi.org/10.1016/j.jbusres.2021.04.070

[15] H. Small, K.W. Boyack, R. Klavans, Identifying emerging topics in science
and technology, Research Policy 43 (2014) 1450–1467.
https://doi.org/10.1016/j.respol.2014.02.005

[16] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, BERT: pre-training of deep
bidirectional transformers for language understanding, in: Proceedings of
NAACL-HLT 2019, pp. 4171–4186. https://doi.org/10.18653/v1/N19-1423

[17] N. Reimers, I. Gurevych, Sentence-BERT: sentence embeddings using Siamese
BERT-networks, in: Proceedings of EMNLP-IJCNLP 2019, pp. 3980–3990.
https://doi.org/10.18653/v1/D19-1410

[18] A. Petukhova, J.P. Matos-Carvalho, N. Fachada, Text clustering with large
language model embeddings, International Journal of Cognitive Computing in
Engineering 6 (2025) 100–108. https://doi.org/10.1016/j.ijcce.2024.11.004

[19] F. Gilardi, M. Alizadeh, M. Kubli, ChatGPT outperforms crowd workers for
text-annotation tasks, Proceedings of the National Academy of Sciences 120
(2023) e2305016120. https://doi.org/10.1073/pnas.2305016120

[20] P. Törnberg, Large language models outperform expert coders and supervised
classifiers at annotating political social media messages, Social Science
Computer Review 43 (2024) 1181–1195.
https://doi.org/10.1177/08944393241286471

[21] M. Heseltine, B. Clemm von Hohenberg, Large language models as a substitute
for human experts in annotating political text, Research & Politics 11 (2024).
https://doi.org/10.1177/20531680241236239

[22] P.J. Rousseeuw, Silhouettes: a graphical aid to the interpretation and
validation of cluster analysis, Journal of Computational and Applied
Mathematics 20 (1987) 53–65. https://doi.org/10.1016/0377-0427(87)90125-7
