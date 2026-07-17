# Repeat Cognitive Assessment and Memory Decline Prediction

This repository contains the analysis code, frozen model artifacts, aggregate results, and reproducibility documentation for a cross-cohort study evaluating whether a previous objective memory assessment improves prediction of subsequent memory decline beyond demographics and a single current assessment.

## Study Design

- Development and internal evaluation: CHARLS 2011, 2013, and 2015.
- First external evaluation: HRS 2012, 2014, and 2016.
- Locked confirmatory evaluation: ELSA waves 6, 7, and 8.
- Primary outcome: a decline of at least 0.5 SD in the cohort-specific 20-point immediate-plus-delayed recall score from the index to outcome assessment.
- Primary comparison: repeat-assessment model M2 versus single-assessment model M1.

The machine-learning analyses are secondary robustness comparisons. The main scientific question concerns the incremental and transportable value of repeat cognitive assessment.

## Repository Contents

- Numbered Python files in the repository root: analysis and reporting scripts in execution order.
- `model_artifacts/`: frozen CHARLS M1 and M2 pipelines, metadata, and SHA-256 hashes.
- `outputs/`: aggregate tables, submission-ready vector figures, editable SVG artwork, and figure-source data only.
- `docs/`: statistical analysis plan, reproducibility guide, and data-governance statement.

No individual-level CHARLS, HRS, or ELSA data or participant-level predictions are included.

## Data Access

Researchers must obtain cohort data from the original repositories and comply with their registration and data-use conditions. The scripts expect the following locally obtained files under `data_link/`, unless another directory is supplied with `--data-dir`:

- CHARLS: `H_CHARLS_D_Data/H_CHARLS_D_Data.dta`
- HRS: `randhrs1992_2022v1.dta`
- Harmonized ELSA: `gh_elsa_h.dta`
- Raw ELSA: `wave_6_elsa_data_eul.dta`, `wave_7_elsa_data_eul.dta`, and `wave_8_elsa_data_eul_v2.dta`

`data_link/` is ignored by Git and must never be committed.

## Environment

Python 3.12 or later is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Reproduction

See [`docs/REPRODUCIBILITY_GUIDE.md`](docs/REPRODUCIBILITY_GUIDE.md) for the execution order and frozen analysis decisions. Scripts should be run from the repository root. For example:

```bash
python 23_repeat_cognition_sensitivity_recalibration.py --data-dir data_link --output-dir outputs --bootstrap 1000 --cv-repeats 5
```

Because later scripts import earlier numbered scripts, keep the numbered files together if relocating them.

Run `python 40_jama_vector_figures.py` after the aggregate analysis outputs are available to regenerate the vector PDF/SVG figures and their plot-specific aggregate CSV files. The workbook at `outputs/jama_vector_figures/Figure_Source_Data.xlsx` is supplied for independent editing and redrawing.

## Data Governance

Only code, aggregate results, and fitted model objects without source records are public. The data-use agreements for CHARLS, HRS, and ELSA do not permit redistribution of individual-level cohort data from this repository.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). A version-specific DOI will be added after the first GitHub release is archived with Zenodo.

## License

The analysis code is released under the MIT License. Cohort data remain subject to their original licenses and data-use conditions.
