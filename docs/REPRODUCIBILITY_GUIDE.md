# Reproducibility Guide

## Frozen Primary Question

Does a previous objective memory assessment improve prediction of subsequent memory decline beyond demographics and a single current assessment?

Development and internal evaluation use CHARLS 2011-2013-2015. External evaluation uses HRS 2012-2014-2016. Locked confirmation uses ELSA wave 6-7-8 after the CHARLS models were frozen.

## Environment

- Python 3.12.13
- pandas 3.0.1
- NumPy 2.5.1
- scikit-learn 1.9.0
- Install dependencies from `requirements.txt`.
- Place locally obtained cohort files under `data_link`, or pass a different directory with `--data-dir`.

## Execution Order

1. `21_repeat_cognition_feasibility_audit.py` audits 3-wave cognition, missingness, proxy status, and sample flow.
2. `22_repeat_cognition_primary_pilot.py` builds M0-M3 and the primary M2 vs M1 comparison.
3. `23_repeat_cognition_sensitivity_recalibration.py` runs prespecified outcome/population sensitivities and HRS recalibration.
4. `24_repeat_cognition_m4_ml_comparison.py` evaluates elastic net and gradient boosting with nested CHARLS tuning.
5. `25_repeat_cognition_ipcw_sensitivity.py` evaluates attrition using inverse-probability weighting.
6. `26_repeat_cognition_reporting_assets.py` regenerates prediction-based calibration and decision-curve data.
7. `27_current_results_and_manuscript_positioning.py` summarizes the final 1000-bootstrap results.
8. `28_formal_external_validation_and_tables.py` creates formal absolute-performance CIs, subgroup results, fixed-capacity metrics, and manuscript tables.
9. `29_journal_figures.py` renders 300-dpi PNG and TIFF figures from aggregate outputs.
10. `30_freeze_models_and_manifest.py` saves the final CHARLS M1/M2 pipelines and a SHA-256 manifest.
11. `31_elsa_data_inventory_and_validation.py` verifies raw versus harmonized ELSA core fields and constructs the locked wave 6-8 sample.
12. `32_elsa_frozen_model_validation.py` verifies artifact hashes and performs the untouched ELSA confirmation.
13. `33_build_supplement.py` assembles eMethods and aggregate eTables, including frozen coefficients and artifact verification.
14. The manuscript workspace uses a separate editorial audit to check agreement among aggregate CSVs, manuscript values, figures, and the supplement; that audit is not needed to rerun the public analysis.

## Final Commands

```powershell
python 23_repeat_cognition_sensitivity_recalibration.py --bootstrap 1000 --cv-repeats 5
python 24_repeat_cognition_m4_ml_comparison.py --bootstrap 1000
python 25_repeat_cognition_ipcw_sensitivity.py --bootstrap 1000
python 28_formal_external_validation_and_tables.py --bootstrap 1000 --cv-repeats 5
python 29_journal_figures.py
python 30_freeze_models_and_manifest.py
python 31_elsa_data_inventory_and_validation.py
python 32_elsa_frozen_model_validation.py --bootstrap 1000
python 33_build_supplement.py
```

Run the commands with Python 3.12 or later in an isolated environment.

## Data Governance

The formal reporting scripts save aggregate tables, figures, model pipelines, software metadata, and hashes. They do not save person-level CHARLS or HRS records. Access to the source cohort data remains governed by the respective data-use agreements.

## ELSA Confirmation

Completed on 2026-07-16 using ELSA wave 6 -> wave 7 -> wave 8:

1. Audit wave-specific direct immediate and delayed recall fields and exclusions.
2. Construct the same 20-point score at 3 waves.
3. Define decline using 0.5 times the ELSA index-score SD.
4. Load the frozen M1 and M2 pipelines from `model_artifacts`.
5. Report original transport performance before any recalibration.
6. Do not alter predictors, preprocessing, coefficients, thresholds, or model selection based on ELSA results.

The locked sample included 6,907 participants and 2,098 events. Frozen M2 improved AUROC over M1 by 0.0332 (95% CI, 0.0278 to 0.0388). Original transport overpredicted absolute risk, so fivefold cross-validated intercept-plus-slope recalibration is reported separately as a secondary implementation analysis.
