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

Each `<mode>_run_config.json` fully describes the run that produced the tables
beside it. Regenerate everything with the scripts in `../examples/`.
