from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_PACKAGES = PROJECT_DIR / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


DEFAULT_DATA_DIR = PROJECT_DIR / "data_link"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
DEFAULT_MODEL_DIR = PROJECT_DIR / "model_artifacts"
FREEZE_DATE = "2026-07-16"


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load_module("22_repeat_cognition_primary_pilot.py", "pilot_freeze")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze final CHARLS M1/M2 pipelines and create a reproducibility manifest."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        rows.append(
            {
                "relative_path": path.relative_to(PROJECT_DIR).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "modified_local": datetime.fromtimestamp(
                    path.stat().st_mtime
                ).astimezone().isoformat(timespec="seconds"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    charls_all, hrs_all = pilot.load_cohorts(args.data_dir)
    charls, charls_y, index_sd = pilot.analysis_sample(
        charls_all, minimum_age=50
    )
    hrs, hrs_y, _ = pilot.analysis_sample(hrs_all, minimum_age=50)
    model_files: list[Path] = []
    model_metadata: dict[str, object] = {
        "freeze_date": FREEZE_DATE,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "development_cohort": "CHARLS",
        "development_waves": [2011, 2013, 2015],
        "minimum_baseline_age": 50,
        "primary_score": "immediate recall 0-10 plus delayed recall 0-10",
        "primary_outcome": "outcome minus index score <= -0.5 * cohort index-score SD",
        "charls_index_score_sd": index_sd,
        "development_n": int(charls_y.size),
        "development_events": int(charls_y.sum()),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "models": {},
        "elsa_rule": (
            "Apply these saved preprocessing and model pipelines unchanged. "
            "Do not select predictors, tune hyperparameters, or refit coefficients in ELSA."
        ),
    }

    for model_name in ["M1_single_assessment", "M2_repeat_assessment"]:
        features = pilot.MODEL_FEATURES[model_name]
        pipeline = pilot.build_model(features)
        pipeline.fit(charls[features], charls_y)
        filename = f"{model_name.lower()}_charls_frozen_{FREEZE_DATE}.joblib"
        path = args.model_dir / filename
        joblib.dump(pipeline, path, compress=3)
        reloaded = joblib.load(path)
        original_probabilities = pipeline.predict_proba(hrs[features])[:, 1]
        reloaded_probabilities = reloaded.predict_proba(hrs[features])[:, 1]
        if not np.array_equal(original_probabilities, reloaded_probabilities):
            raise RuntimeError(f"Round-trip predictions changed for {model_name}.")
        model_files.append(path)
        model_metadata["models"][model_name] = {
            "features_in_order": features,
            "artifact": path.relative_to(PROJECT_DIR).as_posix(),
            "artifact_sha256": sha256(path),
            "round_trip_verification": {
                "hrs_n": int(hrs_y.size),
                "hrs_events": int(hrs_y.sum()),
                "auroc": float(roc_auc_score(hrs_y, reloaded_probabilities)),
                "auprc": float(
                    average_precision_score(hrs_y, reloaded_probabilities)
                ),
                "brier": float(
                    brier_score_loss(hrs_y, reloaded_probabilities)
                ),
                "predictions_bitwise_identical_after_reload": True,
            },
        }

    metadata_path = args.model_dir / f"frozen_model_metadata_{FREEZE_DATE}.json"
    metadata_path.write_text(
        json.dumps(model_metadata, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    frozen_project_files = [
        PROJECT_DIR / "README.md",
        PROJECT_DIR / "DATA_AVAILABILITY.md",
        PROJECT_DIR / "CITATION.cff",
        PROJECT_DIR / "docs" / "STATISTICAL_ANALYSIS_PLAN_v1.md",
    ]
    frozen_project_files.extend(
        PROJECT_DIR / f"{number}_{name}"
        for number, name in [
            (21, "repeat_cognition_feasibility_audit.py"),
            (22, "repeat_cognition_primary_pilot.py"),
            (23, "repeat_cognition_sensitivity_recalibration.py"),
            (24, "repeat_cognition_m4_ml_comparison.py"),
            (25, "repeat_cognition_ipcw_sensitivity.py"),
            (26, "repeat_cognition_reporting_assets.py"),
            (27, "current_results_and_manuscript_positioning.py"),
            (28, "formal_external_validation_and_tables.py"),
            (29, "journal_figures.py"),
            (30, "freeze_models_and_manifest.py"),
        ]
    )
    key_outputs = [
        args.output_dir / "repeat_cognition_formal_absolute_performance.csv",
        args.output_dir / "repeat_cognition_formal_paired_differences.csv",
        args.output_dir / "repeat_cognition_formal_hrs_subgroups.csv",
        args.output_dir / "repeat_cognition_formal_hrs_fixed_capacity.csv",
        args.output_dir / "repeat_cognition_formal_hrs_decision_curve.csv",
        args.output_dir / "repeat_cognition_table1_cohort_characteristics.csv",
        args.output_dir / "figure1_primary_cohort_flow.png",
        args.output_dir / "figure2_sensitivity_forest.png",
        args.output_dir / "figure3_external_validation_calibration_decision_curve.png",
    ]
    all_paths = frozen_project_files + model_files + [metadata_path] + key_outputs
    manifest = pd.DataFrame(manifest_rows(all_paths)).sort_values("relative_path")
    manifest_path = args.output_dir / f"analysis_freeze_manifest_{FREEZE_DATE}.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    readme = f"""# Frozen Model Package

Freeze date: {FREEZE_DATE}

This directory contains the final CHARLS-trained M1 and M2 pipelines for later confirmatory evaluation in ELSA. The artifacts include preprocessing, imputation, scaling, education encoding, and fitted logistic regression coefficients.

## Frozen Analysis

- Development cohort: CHARLS 2011 and 2013, with 2015 outcome.
- Development sample: n = {charls_y.size:,}; events = {int(charls_y.sum()):,}.
- Primary outcome: decline of at least 0.5 SD in the cohort-specific 20-point index memory score.
- M1 features: {', '.join(pilot.M1_FEATURES)}.
- M2 features: {', '.join(pilot.M2_FEATURES)}.

## ELSA Confirmation Rule

Load the saved pipeline and apply it unchanged. ELSA must not be used to choose predictors, tune hyperparameters, alter preprocessing, or refit model coefficients. Original transport performance must be reported before any ELSA-specific recalibration.

The complete hash manifest is `outputs/analysis_freeze_manifest_{FREEZE_DATE}.csv`.
"""
    (args.model_dir / "README.md").write_text(readme, encoding="utf-8")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
