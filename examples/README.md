# Reproduction scripts

The scripts that produced every table and figure of the SoftwareX article, in
run order. They live outside the package because they are analysis drivers,
not library code; `pip install -e .` is enough to run them.

| Script | Produces |
|--------|----------|
| `fetch_scopus.py` | `corpora/*.csv` — the three Scopus corpora |
| `run_case.py` | `results/<corpus>/` — both modes, tables, figures, diagnostics |
| `run_sensitivity.py` | `results/sensitivity/` — corpus-size and seed stability |
| `run_validation.py` | LLM panel study, candidates vs full-corpus control |
| `run_matched.py` | filter-eligible arm (filters vs ranking) |
| `recompute_validation.py` | recomputes every comparison against a control set disjoint from the candidates, from the annotations already on disk (no API calls) |
| `run_consensus.py` | consensus over five seeds and four corpus sizes, with the seed/size overlap decomposition |
| `repair_validation.py` | re-runs one annotator and recomputes a study whose panel lost a model |
| `make_figures.py` | `figures/fig1..fig6` — the manuscript figures |
| `build_supplementary.py` | the supplementary workbook |

Credentials are read from environment variables and never written anywhere:
`SCOPUSAPI`, `SCOPUSINSTTOKEN` for retrieval, `OPENAI` for embeddings, and
`OPENAI`, `ANTROPIC`, `DEEPSEEK`, `XAI`, `GOOGLE` for the annotation panel.
Embeddings are cached under `cache/`, so every stage after the first costs no
API calls.
