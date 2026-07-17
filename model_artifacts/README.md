# Frozen Model Package

Freeze date: 2026-07-16

This directory contains the final CHARLS-trained M1 and M2 pipelines for later confirmatory evaluation in ELSA. The artifacts include preprocessing, imputation, scaling, education encoding, and fitted logistic regression coefficients.

## Frozen Analysis

- Development cohort: CHARLS 2011 and 2013, with 2015 outcome.
- Development sample: n = 7,264; events = 2,886.
- Primary outcome: decline of at least 0.5 SD in the cohort-specific 20-point index memory score.
- M1 features: age, female, education_level, index_cognition.
- M2 features: age, female, education_level, index_cognition, previous_cognition.

## ELSA Confirmation Rule

Load the saved pipeline and apply it unchanged. ELSA must not be used to choose predictors, tune hyperparameters, alter preprocessing, or refit model coefficients. Original transport performance must be reported before any ELSA-specific recalibration.

The complete hash manifest is `outputs/analysis_freeze_manifest_2026-07-16.csv`.
