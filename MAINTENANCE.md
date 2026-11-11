# Maintenance and versioning policy

## Reproduction path

The offline `tfidf-svd` backend is the supported reproduction path. It needs no
credentials and no network, it is deterministic across processes and operating
systems, and continuous integration exercises it on Linux, macOS and Windows
across Python 3.10 to 3.13. Any result in the paper that does not require a
hosted model can be regenerated from the deposited corpora with that backend
alone.

`environment-lock.txt` pins the exact version of every dependency, including
the numerical stack, used for the runs reported in the accompanying article.
Installing from it reproduces the measured numbers; installing from
`pyproject.toml` reproduces the behaviour under current releases.

## Hosted models

Hosted embedding and annotation backends are versioned by the model identifier
recorded in each run configuration (`*_run_config.json`) and in the deposited
annotation tables, together with the access date. The package never substitutes
one model for another silently.

When a provider deprecates a model, or changes an API in a way that breaks the
protocol, the affected backend is marked in `CHANGELOG.md` with the date and
the reason, and the deposited results produced with it are left untouched. Two
such cases are already recorded: one provider's models stopped accepting
`temperature=0`, and a second deprecated the temperature parameter; both were
excluded from the panel rather than replaced by a non-deterministic
substitute.

Hosted endpoints are not fixed inputs over time. Re-embedding the management
corpus at the same endpoint, with the same model identifier and the same run
configuration, months after the deposited run returned 27 candidates sharing 16
with the deposited 26. Treat any hosted backend as a dated observation, not as
a reproducible component.

Published embeddings are preserved through the `precomputed` backend, which
keys vectors by text rather than by row position, so a deposited run can be
regenerated after a hosted endpoint has changed or disappeared.

## Releases

Versions follow semantic versioning. A release that changes any reported number
states so in `CHANGELOG.md` under a `Changed` heading, with the reason. Release
archives are tagged, and the tag is the citable unit.

## Support

Issues and questions: the repository issue tracker. The support address is in
the article's code-metadata table.
