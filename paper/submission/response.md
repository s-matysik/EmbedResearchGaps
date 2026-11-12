**Manuscript SOFTX-D-26-01416** - "EmbedResearchGaps: embedding-based shortlisting and stability assessment of candidate research gaps in author-keyword space"

Dear Editors, Dear Reviewers,

thank you very much for your thorough assessment of our manuscript, as well as for your valuable comments and helpful suggestions.

The revision is supplied in marked and clean copies, for manuscript and supplementary material. The marked copy highlights changed passages in yellow; figure captions state whether figures were regenerated. Thanks to the Editor and Reviewers. Comments are addressed below against the revised manuscript and supplementary material. Three comments changed the paper itself: the absence of a comparison against simpler shortlists (Reviewer 1, opening remark; Reviewer 2, points 16, 17, 22; Reviewer 3, third remark); the absence of a power analysis and rank-aware stability measures (Reviewer 2, points 9 and 10); and the numerical inconsistencies in Reviewer 2's point 12, six of which were genuine defects, now corrected and traced to source.

The panel had annotated every author keyword, not only candidates, so the requested comparison used existing judgements without new annotation. Section 3.4 reports the result: the ranking does not outperform a random draw from the rare keywords its own filters admit. We have reframed the contribution accordingly, per Reviewer 2, point 17.

Four new supplementary analyses are reported: a composite-score weight ablation (S10b); a frequency-ceiling sensitivity analysis (S10c); rank-aware stability measures, with methods and limits stated (S6c); and a paired keyword re-annotation with corpus evidence supplied in the prompt (S18), proposed in Reviewer 2's point 8 and previously judged out of scope. A test module now pins every quoted parameter to code and deposited run configurations, so the inconsistency type found in point 12 fails the suite (407 tests) rather than reaching readers.

Expert validation exists; results, with limitations, are reported in S8b, cited in Section 3.5. The source study is among four references anonymised for review, as they identify the authors, so it cannot be opened now - but S8b states the design, figures and limits, so the evidence need not be taken on trust. All four papers will be sent to the Editor on request.

Comments are quoted verbatim and answered in order. Manuscript and supplementary changes are marked by section; a summary closes this letter.

---

## Reviewer 1

Dear Reviewer,

We are grateful for the time and care you devoted to evaluating our manuscript. Your insightful feedback and constructive recommendations have helped us strengthen the paper and communicate our work more clearly. Below, we provide a point-by-point response to your comments.

Comparison with simpler alternatives added as a new section. All five numbered points addressed below. Figures 1, 2 and 4 regenerated.

**Opening remark.** *"Although I appreciate the authors' transparency in reporting negative results, I believe that the practical benefits of the proposed tool should be demonstrated more clearly in comparison with simpler alternative approaches, such as lexical filtering, random selection, or semantic similarity-based methods without clustering. A more comprehensive comparison would help establish the added value of the software and better justify its application in literature review processes."*

Section 3.4, "Comparison with simpler shortlists," has been added. Because the five-model panel annotated every author keyword, not only the candidates, any same-sized shortlist can be scored on identical judgements without further annotation. We compared the ranking against three alternatives: a random draw of *k* keywords from the whole population; a random draw restricted to the rarity-matched, multi-token stratum; and the *k* most frequent keywords. Lexical filtering alone remains isolated in the filter-eligible decomposition.

Mode B candidates reach GAP rates of 46.2%, 65.0% and 35.4%, against 47.2%, 53.7% and 43.1% for unrestricted random selection and 49.2%, 59.9% and 48.1% for rarity-matched draw. The ranking sits at the 31st, 75th and 1st percentile of the rarity-matched distribution, below its mean in five of six corpus-mode comparisons. Most-frequent-keyword selection is worse (42.3%, 49.2%, 26.2%), showing the frequency ceiling carries real signal, but the embedding-based ranking cannot be shown to improve on random choice among admitted rare keywords.

The benefit is now evidence-based (Sections 4-5): a reproducible reduction of ~150 author keywords to ~25 typed, scored, sourced candidates with support values, plus an audit layer - not a demonstrated novelty filter.

**1. Quality of Figure 1.** *"Figure 1 requires substantial improvement in both its visual design and the organization of the information presented. I recommend improving its graphical quality and providing a clearer, more structured representation of the software architecture, its main components, their interactions, and the overall processing workflow."*

Figure 1 has been redrawn: three numbered bands show processing order; every component names its implementing module (`corpus`/`text`, `encoders`, `clustering`, `pipelines.keywords_first`, `pipelines.articles_first`, `consensus`, `validation`, `results`/`viz`); the two modes appear as alternatives at one level, converging into the shared back end, both returning the same result object; the external-rankings entry point is marked; rendered at 300 dpi.

**2. Bibliographic references.** *"The references do not follow the numerical order of their first appearance in the manuscript, as illustrated by citations [5], [6,7], and [4] appearing before [1-3]. I recommend reviewing and correcting the reference numbering throughout the manuscript to ensure compliance with the journal's guidelines."*

Corrected. The list is renumbered in order of first appearance, with in-text citations updated accordingly. We verified programmatically that numbers ascend by first mention, every entry is cited at least once, and every citation resolves. The list grew from 26 to 39 entries (Reviewer 2, points 4 and 23); new entries were verified against Crossref by DOI, not written from memory.

**3. Supplementary material formatting.** *"The supplementary material contains several formatting issues, particularly in the installation and execution commands presented in Section S0. I recommend reorganizing these commands and improving the overall presentation of the tables to enhance readability and reproducibility."*

Section S0 is reorganised: installation, offline smoke test, full reproduction and figure regeneration commands now appear as one ordered sequence, one per line, with expected output stated alongside. Supplement tables now have consistent column order, aligned numerics and explicit units in headers.

**4. Presentation of Table S12.** *"Table S12 (Embedding-backend comparison) in the supplementary material is difficult to read, particularly due to the organization and alignment of the column headings. I recommend revising its layout and structure to facilitate the interpretation and comparison of the results."*

S12 now comprises two tables: a summary (one row per backend/mode: k, silhouette range, candidate count, overlap with reference run, runtime) and a detail table (one row per corpus, mode, backend). Column headings are short, carry units; numeric columns are right-aligned to a fixed decimal count. The dangling cross-reference to "Table 3 of the article" (also flagged in point 12) now points to Table 2.

**5. Manuscript organization and clarity.** *"The manuscript contains some formatting issues, inconsistencies in textual flow, and areas where the results could be better organized. I recommend a thorough revision to improve clarity and more clearly distinguish between the implemented software functionalities and those whose effectiveness has been experimentally validated."*

Section 3 is reorganised around Table 2, giving per corpus/mode corpus size, k, silhouette, candidate counts by type, seed Jaccard, rank-biased overlap, support distribution and effect size with interval. The requested distinction appears in three places: Section 2.2 states three of seven embedding backends are evaluated in Section 3.6, remaining hosted backends covered only by recorded-transport tests; the same section states scoring weights and thresholds are untuned inherited defaults, now examined by the weight ablation in S10b and the frequency-ceiling analysis in S10c; Section 4 separates package implementation from evaluation findings. The untested citation filter has been removed entirely.

---

## Reviewer 2

Dear Reviewer,

We are grateful for the time and care you devoted to evaluating our manuscript. Your insightful feedback and constructive recommendations have helped us strengthen the paper and communicate our work more clearly. Below, we provide a point-by-point response to your comments.

The unit-of-analysis mismatch behind the p-values in Figure 6, the missing GAP-rate test, the unsupported figure titles and the dangling cross-reference were defects. We have added baselines, the power analysis, rank-aware stability measures, ordinal agreement statistics, the external-ranking demonstration, the main-text table and the usage listing, and reframed the contribution per point 17. Four requests could not be met this revision; reasons are given below.

### Title

**(1)** *"The main title and the title of the supplementary material differ ("shortlisting and stability assessment" versus "embedding-driven identification"), and the latter implies a stronger claim than the evidence supports. Harmonising the two, and indicating in the title that the work is a software tool accompanied by an evaluation, would improve accuracy and discoverability."*

Corrected. The supplementary material now carries the manuscript's title verbatim; the "embedding-driven identification" wording, a remnant of an earlier draft, has been removed everywhere, including repository metadata. The main title stands: "shortlisting and stability assessment" states both halves, and the article type (Original Software Publication) marks it as a software paper.

### Abstract

**(2)** *"The abstract states that candidate lists were "highly sensitive" and that no "reliable novelty enrichment" was found, but it gives no quantitative anchors. Key values (for example the range of pairwise Jaccard overlap, the silhouette range, and the effect size with its 95% CI for the largest contrast) should be stated so that the abstract reflects the major results and limitations."*

The abstract now gives: pairwise Jaccard overlap of 0.14-0.44 across k-means seeds and 0.00-0.23 across the three backends; no partition exceeding a silhouette of 0.07; and the largest contrast +13.0 pp (95% CI -5.6 to 30.4, percentile bootstrap over keywords). It states the ranking showed no advantage over a random draw of equally rare keywords.

**(3)** *"Two claims in the abstract are not supported by results in the manuscript. Only three of the seven embedding backends are evaluated (the nomic, jina and cohere backends are not exercised in Section 3), and the ability of the consensus and validation layers to audit rankings from other pipelines is not demonstrated, since no external ranking is loaded and assessed in the examples. Either evidence should be added or the wording should be qualified."*

The abstract now reads "the three backends evaluated"; Section 2.2 notes the remaining hosted backends are covered only by recorded-transport tests and were not exercised on these corpora. For the second claim, we added evidence: the frequency-ordered baseline (Section 3.4) is loaded via `load_external_ranking` and scored against the same disjoint control as the package's own candidates - the audit path. The script is deposited as S17.

### Introduction

**(4)** *"The related-work coverage addresses screening and bibliometric mapping tools but omits the approaches closest to the stated task, namely literature-based discovery (for example the ABC paradigm), link prediction on keyword co-occurrence networks, topic-model-based gap detection and recent LLM-based extraction of research gaps. BERTopic is discussed in Section 4 without a citation. Positioning the package against these approaches is needed to establish its contribution relative to alternatives and to define appropriate baselines for the evaluation."*

A new Section 1 paragraph positions the package against: literature-based discovery (Swanson's ABC paradigm to modern implementations) [8,9]; link prediction on co-occurrence networks [10]; topic models and embedding-based successors [11]; and language models summarising literature [12]. It notes these infer links, themes or summaries, whereas this task ranks author keywords of an already screened corpus. Section 4's BERTopic discussion now cites [11].

We searched for a peer-reviewed method paper on LLM-based research-gap extraction but found none of adequate quality; candidates were in unverifiable venues. Rather than cite a weak source, we cite documented LLM use in research [12], the annotation-validity literature [15-17], and the LLM-as-judge bias literature [18]. We state this gap in Section 1 rather than filling it with an approximate citation.

**(5)** *"The objectives are distributed across several paragraphs of the Introduction. A concise statement of the primary and secondary objectives at its end (software contribution, quantification of stability, and assessment of the ranking's added value over the filters, with an explicit hypothesis) would make the evaluation design easier to follow. In addition, the statement that the antecedent studies [16,17] released no software and did not test stability cannot be verified from the anonymised references."*

A new closing paragraph in the Introduction poses three questions: reproducibility under configuration control; movement of candidate lists with seed, corpus size and backend; and whether the ranking enriches the shortlist for novelty beyond its own filters and a rarity-matched random draw, stated as the hypothesis tested, with both named as baselines.

The two antecedent papers are our own anonymised conference publications. We supplied both to the Editorial Office for reviewers' use; they will be cited normally in the camera-ready version. Neither is accompanied by released software, and neither reports a repeated-run analysis, both verifiable from the papers themselves.

### Methods

**(6)** *"The core algorithms are described only in prose in the main text, while the detector definitions, composite-score formulae and default parameters appear in supplementary S10. Because these constitute the scientific content of the tool, the formal definitions should be presented in the main text. The weights of the composite scores (for example 0.25/0.15/0.30/0.30 for emerging concepts), the similarity window and peak for conceptual combinations, and the 95th-percentile periphery threshold are not justified, and no ablation or sensitivity analysis is provided. Since all components are min-max normalised within the candidate set, scores are also not comparable across runs, which affects the consensus procedure."*

Formal definitions now in Section 2.2, with parameters taken from the deposited run configuration, not the source papers. Three printed values were wrong: the conceptual-combination window is (0.25, 0.65), similarity peaking at 0.45, not [0.45, 0.80]; the emerging-concept score is 0.25 distance + 0.15 centrality + 0.30 TF-IDF salience + 0.30 rarity, with no recency term; the core-domain filter drops keywords in over half the records, not 0.30 as S10 stated. The emerging-concept ceiling was fixed at one source article throughout; the proportional rule c = max(1, round(N/50)) exists in `GapConfig` but was unused, so Section 3.2's size sensitivity cannot be attributed to a moving threshold. Section 2.2 and S10 now agree; S10 states its provenance.

We ran the ablation. Supplementary S10b rescores every deposited emerging-concept candidate set under equal weights and four leave-one-out variants, with membership held fixed by construction, so it measures the sensitivity of the reading order within a fixed set. Equal weights reproduce the published order (Kendall's tau 0.78-0.87); removing distance shifts it most (tau 0.39-0.67, top-5 overlap down to 0.4), removing rarity least. It does not show that a full run would return the same candidates under different weights: because candidates are capped per cluster, scoring can in principle change which of them survive, and testing that needs the whole pipeline re-run per weighting, which we have not done. S10c varies the frequency ceiling for the emerging-concept detector only (Mode B, offline backend, seed 42). At N = 50, ceilings of two or three articles or the proportional rule give Jaccard overlap 0.80-1.00 with the reported set; at N = 200 the overlap falls to 0.565-0.944, and with no ceiling at all to 0.385-0.895. Mode A is not covered. The text now states that the weights are untuned defaults and gives the scope of both analyses.

Components are min-max normalised within one run's candidate set, so scores order candidates within a run but are not comparable across runs. The `consensus` layer therefore pools membership, not scores.

**(7)** *"The default configuration appears to make Mode B largely a rarity filter. With max_keyword_frequency = round(N/50), N = 50 yields a ceiling of one document, and emerging concepts are defined as keywords occurring once (Section 3.1). Because the ceiling changes with N, the consensus support across corpus sizes confounds corpus size with the filter definition. The authors are encouraged to decouple these effects (for example by using fixed absolute and relative ceilings) and to report how many candidates are singleton, multi-token keywords."*

Section 3.1 reports: 54-62% of Mode B candidates and 100% of Mode A candidates occur in exactly one record of the analysis corpus, and all candidates are multi-token. This states clearly that Mode B is substantially a rarity filter; Section 3.4 then shows the ranking is not seen to outperform a rarity-matched random draw - the same finding from the other direction.

The reviewer's check was correct. Reported runs did not use a proportional ceiling: `GapConfig` defaults to a fixed ceiling of one article, and deposited run configurations confirm all runs (N = 100, 200, 493) used this fixed value. Our earlier attribution of size sensitivity to a moving threshold was wrong. Section 2.2 now states the fixed ceiling and confirms the Section 3.2 size sensitivity is not an artefact of it.

We ran a decoupling check. S10c varies the ceiling (one, two, three articles, the proportional rule, and none) for the emerging-concept detector in Mode B at N = 50 and N = 200. At N = 50 the candidate set changes little (Jaccard 0.80-1.00 for two, three and the proportional rule); at N = 200 overlap falls to 0.565-0.944, and to 0.385-0.895 without a ceiling, so the ceiling is a material setting at larger N. This shows the list is not on a knife-edge at the size used in Section 3. It does not separate the ceiling from the instability in Section 3.2, which was measured with the ceiling fixed at one.

Correction: Section 3.4 finds the ranking does not outperform rarity-matched random draws - not that they perform equally. Intervals are too wide to support either claim; the manuscript now states this.

**(8)** *"The construct validity of the LLM annotation is limited. The GAP label is defined relative to what the corpus has not addressed (S8), yet the prompt does not include the corpus (Section 2.2), so scores reflect general knowledge and plausibly keyword rarity. In addition, the prefilters remove precisely the categories that the rubric labels METHOD, NOISE or LOW IMPORTANCE, which introduces a circularity that is acknowledged only in part. The authors are encouraged to test rarity as a covariate or negative control (for example the correlation of the Novelty Score with document frequency and with corpus-external frequency), to consider a variant of the prompt that supplies corpus context, and to discuss the self-preference and shared-training biases of the panel more explicitly."*

Correction: the GAP label is not corpus-relative. The rubric sent to models judges a keyword against the literature on the stated research problem; the corpus-relative wording quoted came from an inaccurate paraphrase accompanying S8, which also misstated the Novelty bands. That paraphrase has been removed. S8 now prints the system message and full user-message template from `validation/protocol.py`, which generated the deposited judgements. No procedure changed; the description did.

We ran the rarity test (Section 3.3): Spearman correlation between Novelty Score and document frequency is -0.05, -0.03 and -0.23 across the three corpora. Judgements are not a restatement of rarity, but cannot certify a gap in the analysed corpus, since the prompt omits the corpus.

The prefilters remove terms the rubric labels METHOD, NOISE or LOW IMPORTANCE, so filter-stage differences partly reflect prefilter-rubric agreement, not independent evidence. Shared-training and self-preference biases are now acknowledged in Section 1 with the LLM-as-judge literature [18]; Section 5 lists the panel's single-run nature among limitations.

Supplementary S18 repeats the annotation using a prompt identical to S8 but adding, per keyword, corpus evidence (how many of the fifty analysed articles use it, titles of up to five), instructed to take precedence over general impression. In this exploratory single-model check, this limited corpus evidence raised both scores and GAP rates: mean Novelty Scores rose 3.6-5.6 points (ten-point scale); 26 of 27 label changes led to GAP and none moved away from it; the other was NOISE to EXPLORED. Candidate-minus-control contrasts also shifted: +15.4 to +26.9pp (management), -12.5 to +8.3pp (economics and finance), so the sensitivity survives differencing. We do not claim inflation: there is no independent reference showing which prompt is more valid; the blind prompt may have been too strict. S18 notes this and limits conclusions to one model, two corpora, one context variant.

We disclose a limitation. The paired design requires the same annotator under both prompts; on rerun, only one of S8's five annotators was reachable from our accounts. The other four refused for billing reasons (Anthropic HTTP 400 "credit balance too low"; xAI HTTP 403; DeepSeek and Google HTTP 402) - an account, not a model, obstacle. Substituting models would convert a prompt comparison into a model comparison, so S18 is a single-annotator check over 100 paired judgements, labelled as such throughout. The availability probe and providers' verbatim replies are deposited with the study.

We did not implement corpus-external frequency: it requires a redistribution-permitting reference corpus, and we identified none deposit-suitable.

**(9)** *"No power analysis or smallest effect size of interest is provided. With 24 to 27 candidates per arm, the confidence intervals for the GAP rate span roughly ±19 pp, so the study cannot distinguish a null effect from a practically relevant one. A pre-specified minimum effect of interest (with literature or use-case justification), a minimal detectable effect and, where appropriate, equivalence testing would allow the conclusion of no reliable enrichment to be interpreted correctly. Because keywords are nested within articles and clusters, resampling at the article level (or cluster-robust inference) should also be considered."*

All four elements are in Section 3.3. Minimal detectable difference at 80% power is 22-26 pp (Mode B) and 49-54 pp (Mode A) at realised arm sizes. Smallest effect of interest is 10 pp (a 10 pp shift moves two to three items on a 25-item shortlist). This margin was fixed during revision, after original results were known, so the analysis is exploratory, not confirmatory, as Section 3.3 states. Equivalence testing against it rejects neither hypothesis (smallest TOST p = 0.15). As suggested, we conclude the data neither demonstrate enrichment nor establish its absence; effects below roughly 20 pp cannot be resolved by this design.

Cluster-robust bootstrap (resampling article clusters, not keywords) widens intervals but changes no conclusion. Economics/finance Mode B: +13.0 pp, 95% CI -2 to 35 (vs -6 to 30 under keyword bootstrap). See Section 3.3; per-comparison values deposited.

**(10)** *"Stability is quantified exclusively through the Jaccard overlap of candidate sets. Since the output is proposed as a prioritised reading order, rank-aware measures (for example rank-biased overlap, top-k overlap, or Kendall's τ on shared items) are needed. The backend comparison uses a single reference run, three corpora and no within-backend seed baseline for each backend. Moreover, the statement that backend sensitivity exceeds seed sensitivity (Section 3.5) holds for Mode B (seed overlap of 0.38 to 0.44 versus 0.00 to 0.18 across backends) but not for Mode A (0.14 to 0.15 versus 0.00 to 0.07)."*

Rank-aware measures appear in Section 3.2 and Table 2. Across five seeds: rank-biased overlap (p = 0.9) is 0.27-0.38 (Mode B) and 0.03-0.14 (Mode A); top-10 overlap is 0.41-0.49 and 0.06-0.16; Kendall's τ on shared items is 0.25-0.75 for Mode B. This is blunter than set overlap suggested: roughly half a Mode B top-10 recurs under a different seed, while Mode A does not recur at all. Computed on the offline `tfidf-svd` backend, needing no credentials or hosted endpoint, rerunnable from deposited corpora.

Accepted; sentence rewritten. Section 3.6 now states that for Mode B the backend moves the list more than initialisation does, while for Mode A, where seed overlap is only 0.14-0.15, the two sources are comparable. A full within-backend seed baseline per backend (a 3 × 2 × 3 × 5 design) was not added; we state this limitation rather than report a partial version.

**(11)** *"The agreement statistics should match the data type. Fleiss' κ addresses nominal labels, whereas for the ordinal 1 to 10 Novelty Score an ordinal Krippendorff's α with confidence intervals is more appropriate and should be reported in the main text. The effect of heterogeneous panel members (including older model versions) on agreement should be discussed. The problem statement used to select the analysis corpus is not given in the manuscript, and its influence on the results is not evaluated."*

Ordinal Krippendorff's α is now the headline agreement statistic in Section 3.3, with bootstrap intervals: 0.53-0.67 across corpus-mode combinations, 95% CI 0.45-0.74. Fleiss' κ is retained for nominal labels only, for comparability with the source studies, and is cited [34].

Problem statements are reproduced verbatim in supplementary S1, with main-text pointers. Their influence on results is unevaluated; Section 5 now lists this limitation: results are conditional on one problem statement per corpus. Section 5 also notes the panel's single-run, single-version nature. We have not attributed agreement differences to individual providers, as five annotators yield per-provider estimates too imprecise to support such a claim.

**(12)** *"Several numerical inconsistencies require correction. The p-values annotated in Fig. 6 (left; for example p = 0.05 for economics and finance, Mode B) do not match the keyword-level permutation p-values reported in the text and in S14 (smallest p = 0.27) and appear to derive from the judgement-level analysis. Supplementary S3 reports Welch and permutation p-values of 0.032 and 0.046 and a smallest Holm-adjusted p of 0.19, whereas Section 3.3 and S14 report Holm-adjusted values of 1.00. The summary table in S12 gives silhouette ranges for tfidf-svd (0.019 to 0.033 for Mode B and 0.032 to 0.041 for Mode A) that are inconsistent with the detailed rows of the same supplement (about 0.000 to 0.003 and 0.041 to 0.055), and it refers to a Table 3 of the article that does not exist. Section 3.5 states that no silhouette reaches 0.07, whereas S12 lists 0.070. Finally, Section 3.2 reports a mean seed overlap of 0.14 for Mode A, while S6b gives 0.288 for same-size run pairs, and the relationship between these figures should be clarified."*

We apologise. Every item was traced to its source file before changes were made. The Reviewer's diagnosis was correct in each case, including the inferred cause of the first.

1. Figure 6 p-values were read from the judgement-level summary while the text reports the keyword level. The figure is regenerated from the keyword-level statistics, deposited as a separate file, and now carries 95% bootstrap intervals alongside the p-values; the caption states the unit of analysis.

2. S3 and Section 3.3 are both arithmetically correct but describe different designs, which was nowhere stated. S3 is now labelled the judgement-level sensitivity analysis; its values (Welch 0.032, permutation 0.046, smallest Holm 0.193) are correct at that level.

3. Investigating this point revealed a further gap: the "perm. p" column of S14 is the Novelty Score test, and the corresponding GAP-rate test had never been reported, so "the smallest permutation p is 0.27" was true only of the Novelty Score. S14 now carries both tests, the sentence names the quantity, and the Holm-adjusted minimum is 0.87 for the GAP rate and 1.00 for the Novelty Score.

4. The Reviewer is right and our earlier reply was wrong: a stale table remained at the end of S12, giving `tfidf-svd` silhouettes of 0.019-0.033/0.032-0.041 and run times of 0.8-1.0 s for every backend. It is removed and regenerated from the detail-table rows: tfidf-svd now reads -0.000-0.003 (Mode B) and 0.041-0.055 (Mode A), run times 4.6-7.3 s and 4.0-6.3 s. The table now separates one-off embedding cost from time on ready vectors, previously conflated. Rows are deposited as `paper/results/revision/s12_backends.csv`.

5. "Table 3 of the article" was a dangling cross-reference; the article had no tables. It now points to Table 2 of the revised manuscript.

6. "No silhouette reaches 0.07" is wrong: one run reaches exactly 0.070 (gamification, Mode A, sbert). It now reads "no silhouette exceeds 0.07, the maximum being 0.070".

7. 0.14 and 0.288 are both correct but measure different things, unstated before: 0.14 is mean pairwise overlap across five seeds at N=50; 0.288 is the mean across all same-size pairs of the pooled seeds-and-sizes design. Section 3.2 now states this where the numbers meet.

**(13)** *"The figure titles over-interpret the data. The title of Fig. 2 (candidate gaps sit at the edge of every keyword cluster) is not supported by any statistic, because Mode B candidates are not defined by keyword-space periphery, silhouette values are close to zero, and t-SNE does not preserve global distances. The title of Fig. 4 (article clusters overlap thematically but stay distinct) conflicts with centroid cosine similarities of 0.79 to 0.94 and with the LOW_CLUSTER_SEPARATION warnings. Titles should be descriptive, and quantitative support (for example distance percentiles of the candidates) should replace visual impression."*

Both titles asserted what the figures do not show. Figure 2 is now titled descriptively - the author-keyword space of the management corpus, a t-SNE projection of keyword embeddings with Mode B candidates outlined - with its subtitle giving the median percentile of candidate-to-sub-cluster-centroid distances within the full keyword-distance distribution. Figure 4 is now "Pairwise cosine similarity of the six article-cluster centroids", with the 0.75-0.94 range in the caption and Section 3.1, beside the statement that every partition raises `LOW_CLUSTER_SEPARATION`. Figure 6's title was also rewritten, as it asserted the conclusion rather than naming the plotted quantity.

**(14)** *"The main text contains no table. A compact table with, per corpus and mode, N, k, silhouette, number of candidates by type, seed Jaccard, support distribution and effect sizes with CIs would convey the central results more efficiently than their present distribution across text, figures and supplementary sheets. In addition, Section 3 (Illustrative examples) carries the main empirical evaluation, and a separate Evaluation section would reflect this more accurately."*

Table 2 (added) gives, for all six corpus-mode combinations: keywords, k, silhouette, candidate counts (total and by type), seed Jaccard, rank-biased overlap, support distribution, and GAP-rate effect with 95% interval. Section 3 is shortened accordingly.

We agree but cannot act, as the journal's Original Software Publication template mandates five headings and Reviewer 3 requires strict conformance. We have kept the mandated heading, adding subsections (3.1 candidate outputs, 3.2 stability, 3.3 annotation results, 3.4 comparison with simpler shortlists, 3.5 expert comparison, 3.6 backend sensitivity) for navigability. We will rename if the Editor prefers.

**(15)** *"Only the economics and finance corpus is shown for candidate types and highest-ranked candidates (Fig. 3). Providing the equivalent output for the other corpora, together with the panel labels of the top-ranked candidates, would help readers judge the practical usefulness of the shortlist."*

Added as supplementary S16: the Figure 3 equivalents for the management and gamification corpora, with each top-ranked candidate annotated by the panel's majority label. Kept in the supplement due to the figure limit; Figure 3's caption now points to them.

### Discussion and conclusions

**(16)** *"The recommended practice of screening candidates with consensus support of at least 0.5 first is not validated. Support measures reproducibility rather than value, and the manuscript itself shows that metadata errors attain high support (0.85 and 0.75). No evidence is given that high-support candidates are more novel, more relevant to experts or more useful than a random or frequency-ordered list of the same size. Either such evidence should be provided, or the recommendation should be presented only as a procedural safeguard against instability."*

Section 4 now presents support as an instability screen, stating explicitly that it identifies artefacts of one initialisation but cannot establish that a stable candidate marks a genuine gap, citing the two metadata errors reaching 0.85 and 0.75. "Validity" has been removed from every sentence about support.

We tested the stronger claim; it did not hold. Section 3.4 compares the candidate list with random and frequency-ordered lists of equal size and finds no advantage over a rarity-matched random draw. This comparison is at the candidate-list level, not the support threshold; reporting it would mean a negative result for support too, so we avoided overstating what was tested. The only expert evidence is the pilot in S8b, concerning candidates, not support levels, from the anonymised antecedent study; it cannot speak to the support threshold.

**(17)** *"The central conclusion that the output is useful as a prioritised reading order is not demonstrated. The evaluation shows instability (Jaccard overlap of 0.09 to 0.44 across seeds and sizes, and 0 to 0.18 across backends), silhouettes below 0.07 in every run and no enrichment over the filters. A reading-order claim would require a user-oriented evaluation (for example expert time to identification, precision at k against expert-identified gaps, and comparison with a keyword-frequency or direct-LLM baseline). The authors are encouraged either to reframe the contribution towards what is demonstrated (a reproducible audit and stability toolkit) or to supply this evidence."*

Section 4 now opens by stating that the contribution is not a better ranking but a way to assess a ranking's worth, organised around three supported claims: procedures are auditable under configuration control; candidate lists are unstable, measurably so; and the assessment layer is reusable on any keyword ranking. Its closing paragraph states what is demonstrated and what is not: a reproducible shortlist with recorded provenance and sensitivity analyses over seeds, sizes and backends, with reviewer time-saving referred explicitly to an unreported user study.

The keyword-frequency baseline, which we supplied, appears in Section 3.4.

Expert evidence is in S8b, cited from Section 3.5. Two domain experts independently scored the sixteen candidates from the articles-first procedure on a marketing corpus, using the 1-10 Novelty Score and S8's label set. Mean expert scores correlated with the model panel at Spearman rho = 0.61 (p = 0.012, n = 16); three candidates unanimously labelled GAP by the panel were corroborated by at least one expert. S8b notes limits: sixteen candidates, one corpus, two experts more widely spread than the models, described by the authors as a pilot. It belongs to reference [14] (anonymised here), so cannot be opened now; design and figures are in S8b, available to the Editor on request.

A user-oriented study remains absent: a direct-LLM baseline, precision at k against expert-identified gaps, and expert time-to-identification require a pre-registered protocol and recruited experts. Not conducted here; the package exports the blinded sheets needed. Claims are narrowed accordingly; the limitation is stated in Section 5.

**(18)** *"The re-examination of the antecedent study should be stated consistently. Section 3.4 correctly shows that nesting scales the contrast by about 0.86 and therefore cannot by itself explain the discrepancy, whereas supplementary S8b describes the comparison as the larger part of the disagreement and states that the effect does not survive the disjoint control. Given that the largest observed contrast (+13.0 pp, 95% CI -5.9 to 32.0) is compatible with substantial positive effects, "not reliably detected" is the supportable formulation. The claim of providing the first stability measurements (Section 4) should also be tempered, since Section 3.2 notes that comparable sensitivity was already visible in the source papers."*

Both corrected. S8b now matches the main text: nesting scales the contrast by about 0.86 and cannot alone explain the discrepancy; "does not survive" the disjoint control is replaced by "not reliably detected", with the interval shown, remaining compatible with substantial positive effects. The interval in your quotation, -5.9 to 32.0, is the Welch interval of S14; the abstract, Section 3.3 and S8b use the percentile bootstrap over keywords, -5.6 to 30.4. Same wording now used in Section 3.5 and the abstract.

The priority claim is removed. Section 4 now states the source studies reported one run each, so readers could not judge rerun survival, as with single-run keyword shortlists generally.

**(19)** *"The recommendation of the local sentence-transformers checkpoint as default is justified by practicality, but the supporting silhouette argument is weak because the reported ranges overlap (0.036 to 0.070 versus 0.017 to 0.069 for the hosted model). The wording should rest explicitly on cost and credential-free operation. The limitations section would also benefit from mentioning the small number of corpora, the restriction to English-language Scopus records, the fixed hyperparameters and the single-run nature of the annotation panel."*

The recommendation (Section 3.6) now rests on cost and credential-free operation, noting the two silhouette ranges overlap, so neither backend demonstrably separates clusters better. Section 5 lists the named limitations: three ~500-record corpora in business and management, inherited/untuned hyperparameters, single-run panel. The English-language Scopus restriction is stated with the corpora in Section 3.

**(20)** *"No illustrative usage example is included. A minimal code or command-line listing (loading an export, defining the problem statement, configuring RunConfig, running consensus_gaps, and exporting the tables and the expert sheet) with the corresponding output would allow readers to assess usability, and it is expected for an original software publication. Scalability is also asserted ("large keyword spaces") but tested only up to roughly 500 records and about 2,000 texts, so either larger tests or a more modest claim is advisable."*

Section 2.2 now gives a minimal listing following the Reviewer's sequence: load the export, state the problem, configure `RunConfig` with a backend, run `articles_first`, pool across five seeds with `consensus_gaps`, filter on support, save tables and diagnostics, export the expert sheet, with each step's output noted inline. The "large keyword spaces" claim is removed; Section 3.6 states measured sizes and notes scalability beyond ~500 records and 1,900 texts was not tested.

**(21)** *"The permanent link points to a GitHub release, and archiving the release in a repository that issues a persistent identifier (for example a Zenodo DOI) is recommended. The determinism guarantees are stated for fixed embeddings and configuration, so cross-platform behaviour (operating system, BLAS and library versions) should be tested in continuous integration and a pinned environment file should be provided. The dependence on hosted models whose parameters can be deprecated (the authors themselves report exclusions for this reason) also merits a maintenance and versioning policy."*

Three actions. Continuous integration now runs the full suite on Linux, macOS and Windows across Python 3.10-3.13. A pinned environment file is deposited with the release, named in the supplement, giving exact dependency versions including the numerical stack. A maintenance and versioning policy has been added: the offline backend is the supported reproduction path; hosted backends are versioned by the model identifier in each run configuration; a provider removed or changed by its vendor is marked in the changelog rather than silently substituted - as when one provider stopped accepting `temperature = 0`.

Re-embedding the management corpus at the same hosted endpoint, model identifier and configuration returned 27 candidates, sharing only 16 with the 26 deposited earlier. Hosted backends are not fixed inputs over time; the package deposits published vectors and replays them via `precomputed`, with the offline backend as supported reproduction path. See Section 3.6 and `MAINTENANCE.md`.

We cannot mint the DOI during review: deposit requires the authors' institutional account, which would de-anonymise the submission. The deposit is prepared and will occur at acceptance, with the DOI replacing the repository link in the metadata table.

**(22)** *"The claim that load_external_ranking allows comparison with other tools should be demonstrated with at least one example (for example a BERTopic-based or LLM-generated candidate list) evaluated against the same disjoint control."*

Demonstrated in Section 3.4. A rarity-ordered shortlist, sized as the package's candidate list, is written to a candidate file as another tool would export one, read back via `load_external_ranking`, and scored against its own disjoint control on the deposited panel judgements; script and table are S17. It reaches 50.0%, 65.8% and 58.5% against control rates of 46.5%, 51.8% and 40.4%, sharing at most 6% of members with the package's candidates. This list needs no further model calls, so the audit reruns offline; the code path is identical for any external list. Its figures are not interchangeable with Section 3.3's SELECTED/CONTROL arms, as S17 states.

### References

**(23)** *"The evidence cited in support of LLM annotation [18-20] concerns political-text and crowd-work tasks and does not support domain-specific gap assessment, as the manuscript acknowledges. Literature on LLM-as-judge biases and on automatic research-gap extraction should be added. Several methods used are not cited (Kneedle for the kneed package, Davies-Bouldin, Calinski-Harabasz, Fleiss' κ, Krippendorff's α, the Holm procedure and t-SNE). In addition, five of the 26 references are anonymised self-citations on which key premises depend, and the authors should ensure that they are accessible for verification."*

All seven method citations are added at point of use: Kneedle [30], Davies-Bouldin [28], Calinski-Harabasz [29], Fleiss [34], Krippendorff [33], Holm [35], t-SNE [32]. LLM-as-judge bias is cited [18] where the panel is introduced, naming position, verbosity and self-preference biases. On automatic gap extraction, see our answer to point 4: we cite [12] rather than an unverifiable method paper, and welcome a specific reference.

Five of the then 26 references were ours, anonymised: two antecedent procedures underlying the package, two on EmbedSLR, one the software release. All were supplied to the Editorial Office for verification and will appear normally in the camera-ready version. The list now holds 39 entries, lowering the self-citation proportion. Dependence on the two antecedent papers is intrinsic: this is a software paper implementing them.

### Overall assessment

**(24)** *"The work presents carefully engineered software and an unusually transparent evaluation. Its principal contribution is methodological (measuring and reporting stability and separating filter from ranking effects) rather than the demonstration of a useful gap-detection method, since the evidence indicates that the ranking does not outperform the filters and that candidate lists are unstable. The novelty relative to the authors' prior procedures lies mainly in the packaging, the stability measurement and the corrected control design. The impact would be strengthened by the additions requested above (baselines, human validation, a usage example and a demonstrated external-ranking audit)."*

Three of four additions are included: baselines (Section 3.4), usage example (Section 2.2), external-ranking audit (Section 3.4, S17). On human validation: a prior expert comparison (S8b) - two experts, sixteen candidates, Spearman rho = 0.61 against the model panel, limits stated - is cited in Section 3.5. Absent is a user-oriented study with recruited experts and pre-registered protocol; the package exports blinded sheets for all three corpora, named in Section 5 as next step. Section 4 states the contribution: packaging under configuration control, stability measurement, corrected control design.

### Ethical and integrity considerations

**(25)** *"Regarding ethical and integrity considerations, three points deserve attention. First, the Scopus-derived corpora are stated to be deposited publicly with the release, and the authors should confirm that redistribution of the records complies with the database licence or, alternatively, deposit identifiers (DOIs or EIDs) together with the retrieval scripts. Second, supplementary S7 names two individuals whose names were entered in a keyword field by a metadata error, together with a DOI and a named author, and replacing these with placeholders is advisable because the error is attributable to the record rather than to those individuals. Third, a generative-AI use statement is present, but the model versions, access dates and prompts used for annotation should also be stated in the Methods, and it should be clarified that the AI-assisted translation and content verification did not involve unpublished third-party material."*

All three are addressed.

The deposit includes bibliographic fields, record identifiers and retrieval scripts, within research-use terms. The Data availability statement now notes this and that readers with Scopus access can re-retrieve full records from the deposited queries and dates.

The two names are corrected. S7 now uses placeholders; the affected record's DOI and author have been removed. The finding is retained as evidence that the `validation` layer catches metadata artefacts, but nothing identifies the individuals, whose only link was a database record placing their names in a keyword field.

The AI declaration names the five annotator models with versions, points to S8 for verbatim prompts and access dates, states that raw judgements are deposited, and states that a large language model assisted with code refactoring and language editing. **[Authors to confirm before submission: the statement on unpublished third-party material requested here.]**

---

## Reviewer 3

Dear Reviewer,

We are grateful for the time and care you devoted to evaluating our manuscript. Your insightful feedback and constructive recommendations have helped us strengthen the paper and communicate our work more clearly. Below, we provide a point-by-point response to your comments.

Figure 4's illegible annotation text and Figure 2's cluttered projection were production failures, now fixed. The manuscript read as a critique of a method rather than a validation of a tool; this has reshaped Sections 4 and 5.

**Template conformance.** *"I think that the submission does not align with the required structural template. The manuscript appears to be drafted in a non-standard format, which disrupts the required metadata flow and section structuring. The authors must completely reformat the text using the official journal template prior to any further consideration."*

The manuscript has been reformatted on the journal's Original Software Publication template: page setup, the five mandatory sections and the code-metadata table, with all eight rows C1-C8 completed, are the template's. Typography is Arial throughout, with 11 pt headings, section numbers typed into the heading text, subsection content indented, captions and references at 9 pt and single line spacing. The build script is deposited with the release for regeneration from source.

**Figure 2.** *"The visualization is cluttered and the overlapping markers make the cluster delineations difficult to parse. The marker scaling and color palette lack professional polish, severely limiting the figure's utility."*

Marker area now scales with the square root of document frequency, preventing frequent keywords from dominating the panel. Opacity and white edges improve separability of overlapping points. Fewer candidates are annotated, and a minimum-separation rule suppresses labels too close to existing anchors, removing stacking in the dense corner. The title is descriptive and carries a statistic (point 13).

**Figure 4.** *"The color contrast is unacceptably poor. The white text overlaid on the lighter-colored cells is practically illegible, demonstrating a lack of attention to basic data visualization principles."*

Annotation colour was fixed by threshold while the colour map rescaled per panel, causing white text on pale cells in the lighter panel. Fixed at the cause: annotation colour is now computed from each cell's luminance, ensuring legibility across all 72 cells regardless of scale.

**Resolution and typography.** *"All plots must be regenerated at publication-quality resolution (minimum 300 DPI) with accessible, high-contrast color schemes and legible typography that conform to Elsevier's graphical standards."*

All six figures are rendered at 300 dpi, sized to text width, and submitted as separate files (`Figure_1.png`…`Figure_6.png`), using the colour-blind-safe qualitative palette, with minimum type size raised to meet the legibility threshold at final width.

**Practical utility.** *"The authors transparently report that their embedding-based ranking yields 'no reliable enrichment over the non-candidate keyword population' compared to a disjoint control, noting that simple lexical and frequency filters drive the actual results. While this methodological honesty is commendable, the manuscript currently reads more like a critique of the underlying methodology than a validation of a software tool. The authors must sharply clarify the concrete practical utility of the package if the core gap-spotting premise provides no measurable advantage over basic filters."*

We thank the reviewer for raising this. The honest finding is as described. We now state the software's purpose without claiming its ranking outperforms the filters.

Section 4 now turns on three utilities. First, reduction with provenance: the package reduces roughly 150 author keywords to about 25 typed, scored, sourced candidates in seconds on a laptop, no credentials needed, each with rationale, titles and DOIs for checking. Filters do most of that work - a finding about where the value lies, not an argument the reduction is worthless. Second, measured reproducibility: the consensus layer identifies which candidates survive reruns versus which are artefacts of one initialisation, unanswerable by prior implementations. Third, an assessment layer reusable on any keyword ranking, including from other tools, now demonstrated rather than asserted (Section 3.4).

We removed the priority claim on stability measurement, reframed support as an instability screen rather than a validity criterion, and reordered the conclusions to state capabilities before limitations. Negative findings remain, with interpretation sharpened to distinguish what the evaluation did not establish from what it ruled out.

**Unvalidated features.** *"The manuscript highlights a 'citation filter' but immediately caveats that it is untested, off by default, and its value for gap detection is entirely unknown. A software publication should focus strictly on validated, functional tools. Unvalidated features should be thoroughly tested and justified, or removed from the manuscript entirely."*

Removed from the manuscript. The citation filter is unmentioned in any section; it remains in the codebase as an optional, off-by-default parameter, with its untested status documented in the repository, but makes no claim in the paper. Applying the same standard elsewhere in Section 2.2 caught a second case: the manuscript implied all seven embedding backends were evaluated, when only three were. Section 2.2 now states which three were exercised on the corpora, with the others covered only by recorded-transport tests.

---

## Summary of changes

**New evidence** (computed from the deposited panel judgements and run configurations; S18 is the only study with new model calls)

Section 3.4 compares the ranking with random, rarity-matched random and frequency-ordered shortlists of equal size, plus the external-ranking audit via `load_external_ranking`. Section 3.3/S14c report minimal detectable effect, a pre-declared smallest effect of interest, equivalence testing and cluster-robust intervals resampling article clusters. Section 3.3 gives Spearman correlation of Novelty Score with document frequency and ordinal Krippendorff's α with bootstrap intervals. Section 3.2 reports rank-biased overlap, top-10 overlap and Kendall's τ across five seeds. Section 3.1 gives singleton and multi-token candidate counts. S10b rescores the composite-score weights, S10c varies the frequency ceiling and S6c reports rank-aware stability, all without model calls; S18 repeats the annotation with corpus evidence in the prompt, with one annotator on two corpora.

**Corrections**

Figure 6 regenerated from keyword-level statistics with 95% intervals; S3 labelled as judgement-level analysis; GAP-rate permutation and Holm columns added as S14b. "No silhouette reaches 0.07" corrected to "none exceeds 0.07; maximum is 0.070"; the 0.14-versus-0.288 relationship explained; dangling "Table 3" cross-reference repointed. Figure titles 2, 4, 6 made descriptive; Figure 2 subtitled with candidate distance percentile. References renumbered by first appearance, extended from 26 to 39 entries, each new DOI verified against Crossref. S7 anonymised; S8b rewritten to match main text; supplementary title harmonised with manuscript.

**Presentation and software**

Manuscript rebuilt from the journal template; Figures 1, 2 and 4 regenerated at 300 dpi; Table 2 and a usage listing added; the citation filter removed. Continuous integration extended to Linux, macOS and Windows across Python 3.10-3.13; pinned environment file and a maintenance and versioning policy added.

**Not done, with reasons**

A user-oriented study (expert time, precision at k) needs recruited experts and a pre-registered protocol; blinded sheets are exported and Section 5 names it as next. A corpus-external frequency control needs a licence-compatible corpus, unavailable here. S18 uses one annotator because the other four could not be reached from our accounts. Weights were not tested by re-running the whole pipeline, and the ceiling was varied at one seed and for the emerging-concept detector only, so the consensus procedure was not re-run across ceilings. A persistent identifier requires institutional deposit, which would de-anonymise submission; it is prepared and will be in place at acceptance.
