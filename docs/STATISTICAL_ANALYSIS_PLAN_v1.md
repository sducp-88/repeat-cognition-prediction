# Statistical Analysis Plan v1

Study: Incremental value of repeat cognitive assessment for subsequent cognitive decline  
Version/date: 1.0 / 2026-07-15  
Status: Frozen before formal model comparison  
Parent protocol: `STUDY_MASTER_PLAN.md`

## 1. Primary objective

Estimate the incremental predictive value of a prior cognitive assessment beyond demographics and a single current assessment for subsequent objective memory decline.

Primary contrast: M2 repeat-assessment model versus M1 single-assessment model.

## 2. Cohorts and time origin

| Cohort | First assessment | Index/second assessment | Outcome |
|---|---:|---:|---:|
| CHARLS development | 2011 | 2013 | 2015 |
| HRS external validation | 2012 | 2014 | 2016 |

The prediction time origin is the second assessment. All modifiable clinical predictors are taken from the second assessment. Sex and education are treated as stable attributes; age is measured at the second assessment.

## 3. Primary analysis population

- Age 50 years or older at the first assessment.
- Direct cognitive assessment with complete immediate and delayed recall at all three waves.
- HRS: proxy interviews and RAND-imputed immediate/delayed recall components are excluded at all three waves.
- CHARLS: explicit proxy and dementia diagnosis indicators are unavailable in Harmonized Version D; complete objective cognition is required.
- HRS exclusion of reported dementia or Alzheimer disease by the index wave is a prespecified sensitivity analysis, not a cross-cohort primary rule.
- Age 65 years or older is a prespecified subgroup/sensitivity population.

## 4. Cognitive scores and primary outcome

Primary score: immediate recall 0-10 plus delayed recall 0-10, total 0-20.

For each cohort, calculate the standard deviation of the index-wave 20-point score among age-eligible participants with a complete index score. The primary outcome is:

`outcome score - index score <= -0.5 * index-score SD`

The threshold is calculated without using outcome values. The CHARLS development threshold is not imposed on HRS because the test distribution and language differ; each cohort uses the same prespecified standardized definition.

## 5. Predictor sets

### M0 demographic reference

- Age at index assessment, continuous.
- Sex.
- Education level.

### M1 single-assessment model

- M0 plus index 20-point memory score.

### M2 repeat-assessment model

- M1 plus first 20-point memory score.

M2 may be reparameterized as index score plus previous change for interpretation; this is algebraically equivalent in a linear model. The primary prediction uses both observed scores and does not define the outcome using residualized change.

### M3 expanded clinical model

- M2 plus marital/partnered status, rural residence, self-rated health, depressive symptom fraction, ever smoking, current smoking, alcohol use, hypertension, diabetes, heart disease, stroke, 5-item ADL difficulty count, and current paid work.
- All updateable variables are measured at the index wave: CHARLS 2013 and HRS 2014.
- Laboratory CKM staging is not included because aligned index-wave biomarkers are unavailable in both cohorts. Hypertension, diabetes, heart disease and stroke form the prespecified clinical CKM-related domain.

### M4 algorithmic comparator

- Same candidate information as M3.
- Elastic-net logistic regression and histogram gradient boosting only.
- No post hoc model zoo or algorithm selection based on HRS results.

## 6. Model specification

- M0-M3: logistic regression with prespecified predictors; continuous variables standardized using CHARLS training data.
- Education is categorical and one-hot encoded.
- Binary predictors retain 0/1 coding.
- Missing predictor values are imputed inside each training fold. Numeric predictors use median imputation with missingness indicators; categorical predictors use the most frequent level.
- The fitted CHARLS preprocessing and model pipeline is applied unchanged to HRS.
- Elastic net tuning grid: `C` in 0.01, 0.1, 1, 10 and `l1_ratio` in 0, 0.25, 0.5, 0.75, 1.
- Gradient boosting tuning grid is limited to prespecified combinations of learning rate, leaf count and L2 regularization. Final grid will be stored in code before execution.

## 7. Internal and external validation

- CHARLS internal validation: repeated stratified 5-fold cross-validation for M0-M3.
- Tuned M4 models: nested cross-validation, with inner tuning performed only in the training portion of each outer fold.
- Final external validation: fit once to the full eligible CHARLS sample, then apply without refitting to HRS.
- HRS recalibration analyses: intercept-only recalibration and intercept-plus-slope recalibration reported separately from original transport performance.
- ELSA remains untouched until model, preprocessing, thresholds and reporting code are frozen.

## 8. Performance measures

Primary performance measure: difference in AUROC between M2 and M1.

Additional measures:

- AUPRC.
- Brier score.
- Calibration intercept and slope.
- Observed and mean predicted risk.
- Decision-curve net benefit.
- Sensitivity, specificity and positive predictive value when the highest-risk 10%, 20% and 30% are selected.

Uncertainty for paired model differences will use participant-level paired bootstrap with at least 1000 replicates in final analyses. Pilot runs may use fewer replicates and must be labelled exploratory.

## 9. Missing outcome, attrition and death

- Primary performance analysis uses observed three-wave cognition.
- Describe each attrition step and compare index characteristics of observed versus unobserved outcomes.
- Prespecified sensitivity: inverse probability of outcome observation weighting.
- Death is not coded as cognitive decline. A composite adverse outcome of death or cognitive decline is supplementary.
- Survey-weighted estimates are sensitivity analyses; unweighted individual-level prediction is primary.

## 10. Prespecified sensitivity analyses

1. Age 65 years or older.
2. Common 25-point score adding serial 7s.
3. Decline of at least 1 index-score SD.
4. Continuous outcome score/change.
5. Standardized regression/reliable change outcome.
6. HRS exclusion of reported dementia/Alzheimer disease by 2014.
7. HRS analysis allowing RAND-imputed cognition components.
8. Exclusion of baseline score below the cohort-specific fifth percentile.
9. Exclusion of prevalent stroke and stratification by incident stroke.
10. Long-term outcome ending in 2018 after CHARLS test-equating and HRS interview-mode handling.

## 11. Subgroups and fairness

Prespecified subgroups: sex, age 50-64 versus 65 or older, education level, and rural/urban residence. Report discrimination, calibration and fixed-capacity sensitivity. Subgroup results are descriptive and are not used to retune the main model.

## 12. Multiplicity and interpretation

- M2 versus M1 is the only primary model comparison.
- M3 and M4 comparisons are secondary.
- Sensitivity and subgroup analyses are interpreted by consistency, effect size and confidence intervals rather than isolated P values.
- No claim that screening improves patient outcomes will be made; the study evaluates risk stratification only.

## 13. Data governance and reproducibility

- No HRS or CHARLS person-level records are written to project outputs by the new analysis scripts.
- Outputs are aggregate tables, model objects without source records, figures and logs.
- Random seeds, software versions and exact variable mappings are saved.
- Changes to this SAP require a dated amendment in both this file and `STUDY_MASTER_PLAN.md`.

## 14. Amendments

None after freezing this version.
