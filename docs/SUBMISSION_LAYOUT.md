# Project and submission layout

The working project keeps one core library at `src/labkit/`. Do not change import
paths or move its modules to prepare a submission. Data, notebooks and scripts
remain alongside it so the existing pipeline can run.

```text
project/
  src/labkit/                 core library
  notebooks/                 NB1-NB6 Python sources
  scripts/                   run, verification and packaging helpers
  data/                      frozen core corpus and splits
  results/                   measured core results and bonus summaries
  adapters/                  local model artifacts; not included in Option B ZIP
  bonus/
    B2_UIT/                  custom-domain data, configuration and results
    B4/                      original per-rank evidence and compact evidence ZIP
    B5_HUB/                  local upload staging; excluded from ZIP and Git
  submission/
    REPORT.md                source report
    REFLECTION.md            learner reflection
    README.md                which file to submit
  backups/
    runs/B4/                 full Colab run backups
    submissions/             old ZIPs and expanded packages
    colab-setup/              setup and repair ZIPs
    screenshots/             archived screenshots
```

Submit the GitHub repository directly, following Option B of `rubric.md`, with
the public adapter URL in LINKS.md and submission/REPORT.md. Source/data and bonus
evidence are included in Git; model weights, caches, unfinished downloads, tokens,
and backups are excluded. Historical expanded packages and ZIPs live in backups/.

Package checks: `python scripts/verify.py` and `python scripts/verify_b4.py`, run from
the repository root. B2 contains AI review, not a claim of independent human review.
B3 is incomplete and is not submitted as completed bonus evidence.
