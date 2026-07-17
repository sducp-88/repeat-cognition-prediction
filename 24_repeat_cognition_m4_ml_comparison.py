from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import warnings
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_PACKAGES = PROJECT_DIR / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


DEFAULT_DATA_DIR = PROJECT_DIR / "data_link"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
RANDOM_SEED = 20260716
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")


def load_pilot_module():
    spec = importlib.util.spec_from_file_location(
        "repeat_cognition_primary_pilot",
        PROJECT_DIR / "22_repeat_cognition_primary_pilot.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load primary pilot module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load_pilot_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested CHARLS tuning and HRS external validation for M4 ML comparators."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--outer-folds", type=int, default=5)
    return parser.parse_args()


def build_reference_m2() -> Pipeline:
    return pilot.build_model(pilot.MODEL_FEATURES["M2_repeat_assessment"])


def build_elastic_net() -> tuple[Pipeline, dict[str, list[object]]]:
    model = Pipeline(
        [
            ("preprocess", pilot.preprocessing(pilot.M3_FEATURES)),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=8000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    }
    return model, grid


def build_gradient_boosting() -> tuple[Pipeline, dict[str, list[object]]]:
    model = Pipeline(
        [
            ("preprocess", pilot.preprocessing(pilot.M3_FEATURES)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    random_state=RANDOM_SEED,
                    early_stopping=True,
                ),
            ),
        ]
    )
    grid = {
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__max_leaf_nodes": [7, 15, 31],
        "model__l2_regularization": [0.0, 1.0],
        "model__min_samples_leaf": [30],
    }
    return model, grid


def metric_values(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    intercept, slope = pilot.calibration_parameters(y, probabilities)
    return {
        "auroc": float(roc_auc_score(y, probabilities)),
        "auprc": float(average_precision_score(y, probabilities)),
        "brier": float(brier_score_loss(y, probabilities)),
        "observed_percent": 100.0 * float(y.mean()),
        "mean_predicted_percent": 100.0 * float(probabilities.mean()),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def nested_oof_and_external(
    estimator: Pipeline,
    grid: dict[str, list[object]] | None,
    features: list[str],
    charls: pd.DataFrame,
    charls_y: np.ndarray,
    hrs: pd.DataFrame,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Counter[str]]:
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    oof = np.full(charls_y.size, np.nan, dtype=float)
    selected: Counter[str] = Counter()
    for fold, (train_index, test_index) in enumerate(outer.split(charls[features], charls_y), 1):
        if grid is None:
            fold_model = clone(estimator)
        else:
            inner = StratifiedKFold(
                n_splits=inner_folds, shuffle=True, random_state=seed + fold
            )
            fold_model = GridSearchCV(
                clone(estimator),
                grid,
                scoring="roc_auc",
                cv=inner,
                n_jobs=1,
                refit=True,
            )
        fold_model.fit(charls.iloc[train_index][features], charls_y[train_index])
        if grid is not None:
            selected[str(fold_model.best_params_)] += 1
        oof[test_index] = fold_model.predict_proba(charls.iloc[test_index][features])[:, 1]

    if np.isnan(oof).any():
        raise RuntimeError("Outer cross-validation left missing predictions.")

    if grid is None:
        final_model = clone(estimator)
    else:
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + 999)
        final_model = GridSearchCV(
            clone(estimator),
            grid,
            scoring="roc_auc",
            cv=inner,
            n_jobs=1,
            refit=True,
        )
    final_model.fit(charls[features], charls_y)
    if grid is not None:
        selected["FULL_FINAL " + str(final_model.best_params_)] += 1
    external = final_model.predict_proba(hrs[features])[:, 1]
    return oof, external, selected


def performance_rows(
    model_name: str,
    charls_y: np.ndarray,
    charls_probabilities: np.ndarray,
    hrs_y: np.ndarray,
    hrs_probabilities: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for dataset, y, probabilities in (
        ("CHARLS nested 5-fold CV", charls_y, charls_probabilities),
        ("HRS external validation", hrs_y, hrs_probabilities),
    ):
        row = {
            "dataset": dataset,
            "model": model_name,
            "n": int(y.size),
            "events": int(y.sum()),
            "event_percent": 100.0 * float(y.mean()),
        }
        row.update(metric_values(y, probabilities))
        rows.append(row)
    return rows


def paired_difference_rows(
    dataset: str,
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    candidate_name: str,
    bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    rows = pilot.bootstrap_difference(y, reference, candidate, bootstrap, seed)
    for row in rows:
        row.update(
            {
                "dataset": dataset,
                "reference_model": "M2_repeat_assessment",
                "candidate_model": candidate_name,
            }
        )
    return rows


def counter_rows(model_name: str, selected: Counter[str]) -> list[dict[str, object]]:
    return [
        {"model": model_name, "parameter_set": parameter_set, "count": count}
        for parameter_set, count in selected.most_common()
    ]


def write_report(
    performance: pd.DataFrame,
    differences: pd.DataFrame,
    tuning: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = [
        "# Repeat Cognition Phase 3: M4 Machine-Learning Comparison",
        "",
        "Date: 2026-07-16",
        "",
        "Scope: M4 uses the same candidate variables as M3. Hyperparameters were tuned inside CHARLS only. HRS is external validation without refitting.",
        "",
        "## Performance",
        "",
        "| Dataset | Model | n | Events | AUROC | AUPRC | Brier | Observed | Predicted |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in performance.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['model']} | {int(row['n'])} | "
            f"{int(row['events'])} | {row['auroc']:.3f} | {row['auprc']:.3f} | "
            f"{row['brier']:.3f} | {row['observed_percent']:.1f}% | "
            f"{row['mean_predicted_percent']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## M4 vs M2 Paired Differences",
            "",
            "| Dataset | Candidate | Metric | Difference | 95% CI |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in differences.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['candidate_model']} | {row['metric']} | "
            f"{row['difference_candidate_minus_reference']:.4f} | "
            f"{row['ci_low']:.4f} to {row['ci_high']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If M4 does not materially improve HRS external performance over M2, the manuscript should emphasize repeat cognition as a transparent, transportable predictor rather than AI model complexity.",
            "- Machine-learning results can remain as a secondary analysis to show that the main finding is not an artifact of an underpowered linear model.",
        ]
    )
    (output_dir / "repeat_cognition_phase3_m4_ml_comparison_2026-07-16.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def update_master_plan() -> None:
    master = PROJECT_DIR / "STUDY_MASTER_PLAN.md"
    text = master.read_text(encoding="utf-8")
    addition = """

### 2026-07-16 Phase 3 M4 machine-learning update

- Added `24_repeat_cognition_m4_ml_comparison.py`.
- Completed M4 nested CHARLS tuning for elastic-net logistic regression and histogram gradient boosting.
- HRS was used only for external validation; no HRS-based predictor or hyperparameter selection was performed.
- Outputs saved under `outputs/repeat_cognition_phase3_*`.
- IPCW, decision-curve plots and final manuscript figures remain pending.
"""
    if "2026-07-16 Phase 3 M4 machine-learning update" not in text:
        marker = "## Progress Log"
        if marker in text:
            text = text.replace(marker, addition.strip() + "\n\n" + marker)
        else:
            text = text.rstrip() + "\n\n" + addition.strip() + "\n"
        master.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    charls_all, hrs_all = pilot.load_cohorts(args.data_dir)
    charls, charls_y, _ = pilot.analysis_sample(charls_all, minimum_age=50)
    hrs, hrs_y, _ = pilot.analysis_sample(hrs_all, minimum_age=50)

    model_specs = {
        "M2_repeat_assessment": (
            build_reference_m2(),
            None,
            pilot.MODEL_FEATURES["M2_repeat_assessment"],
        ),
        "M4_elastic_net": (*build_elastic_net(), pilot.M3_FEATURES),
        "M4_gradient_boosting": (*build_gradient_boosting(), pilot.M3_FEATURES),
    }

    predictions: dict[tuple[str, str], np.ndarray] = {}
    tuning_rows: list[dict[str, object]] = []
    performance_output: list[dict[str, object]] = []
    for index, (model_name, (estimator, grid, features)) in enumerate(model_specs.items()):
        charls_prob, hrs_prob, selected = nested_oof_and_external(
            estimator,
            grid,
            features,
            charls,
            charls_y,
            hrs,
            args.outer_folds,
            args.inner_folds,
            RANDOM_SEED + index * 100,
        )
        predictions[("CHARLS nested 5-fold CV", model_name)] = charls_prob
        predictions[("HRS external validation", model_name)] = hrs_prob
        performance_output.extend(
            performance_rows(model_name, charls_y, charls_prob, hrs_y, hrs_prob)
        )
        tuning_rows.extend(counter_rows(model_name, selected))

    difference_output: list[dict[str, object]] = []
    for dataset, y in (
        ("CHARLS nested 5-fold CV", charls_y),
        ("HRS external validation", hrs_y),
    ):
        reference = predictions[(dataset, "M2_repeat_assessment")]
        for offset, model_name in enumerate(["M4_elastic_net", "M4_gradient_boosting"]):
            difference_output.extend(
                paired_difference_rows(
                    dataset,
                    y,
                    reference,
                    predictions[(dataset, model_name)],
                    model_name,
                    args.bootstrap,
                    RANDOM_SEED + offset + len(dataset),
                )
            )

    performance = pd.DataFrame(performance_output)
    differences = pd.DataFrame(difference_output)
    tuning = pd.DataFrame(tuning_rows)

    performance.to_csv(
        args.output_dir / "repeat_cognition_phase3_m4_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    differences.to_csv(
        args.output_dir / "repeat_cognition_phase3_m4_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tuning.to_csv(
        args.output_dir / "repeat_cognition_phase3_m4_tuning.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_report(performance, differences, tuning, args.output_dir)
    update_master_plan()
    print(performance.to_string(index=False))
    print(differences.to_string(index=False))


if __name__ == "__main__":
    main()
