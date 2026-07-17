# Data Availability and Governance

This repository does not contain individual-level data from CHARLS, HRS, or ELSA and does not contain participant-level predictions.

The source data must be requested or downloaded from the original cohort repositories under their respective registration, licensing, and data-use conditions. Users are responsible for confirming that their use complies with the current terms of each cohort.

The public materials are limited to:

- analysis and reporting code;
- aggregate tables and figures;
- frozen model pipelines without source participant records;
- software metadata, model metadata, and cryptographic hashes; and
- documentation needed to reconstruct the analysis after obtaining authorized data access.

The `.gitignore` file excludes common restricted-data formats and the local `data_link/` directory. Do not add source cohort files, row-level derived datasets, row-level predictions, direct identifiers, or linkage files to this repository.
