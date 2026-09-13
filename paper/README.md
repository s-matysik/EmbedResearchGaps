# Paper artefacts

| Path | Contents |
|------|----------|
| `manuscript.md` | manuscript source (SoftwareX template) |
| `supplementary_material.md` | supplementary sections S0-S11 |
| `supplementary_material.xlsx` | supplementary tables S1-S7 |
| `figures/fig1..fig6` | the six manuscript figures |
| `results/<corpus>/` | candidate tables, keyword tables, diagnostics, run configs, validation annotations and reports, blinded expert sheets |
| `results/sensitivity/` | corpus-size and k-means-seed sensitivity tables |
| `results/manuscript_numbers.csv` | every number quoted in the article, in one table |
| `submission/` | the files as submitted: the article in the journal's Original Software Publication template, the supplementary document, `figures/Figure_1..6.png` at 300 dpi, `highlights.txt` and `graphical_abstract.png` |

Each `<mode>_run_config.json` fully describes the run that produced the tables
beside it. Regenerate everything with the scripts in `../examples/`.
