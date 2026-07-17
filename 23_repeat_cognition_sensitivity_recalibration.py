from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_PACKAGES = PROJECT_DIR / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


DEFAULT_DATA_DIR = PROJECT_DIR / "data_link"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
RANDOM_SEED = 20260716


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
        description="Run sensitivity analyses and HRS recalibration for repeat cognition models."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--cv-repeats", type=int, default=5)
    return parser.parse_args()


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def expit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


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


def analysis_sample(
    data: pd.DataFrame,
    minimum_age: int,
    decline_sd: float,
    exclude_dementia_or_ad: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    eligible = data["baseline_age"].ge(minimum_age) & data["index_cognition"].notna()
    if exclude_dementia_or_ad:
        eligible = eligible & data["index_dementia_or_alzheimer"].eq(0)
    index_sd = float(data.loc[eligible, "index_cognition"].std(ddof=1))
    complete = (
        eligible
        & data["previous_cognition"].notna()
        & data["outcome_cognition"].notna()
    )
    sample = data.loc[complete].copy()
    sample["previous_change"] = (
        sample["index_cognition"] - sample["previous_cognition"]
    )
    outcome = (
        sample["outcome_cognition"] - sample["index_cognition"]
        <= -decline_sd * index_sd
    ).astype(int)
    return sample, outcome.to_numpy(), index_sd


def fit_predict_pair(
    charls: pd.DataFrame,
    charls_y: np.ndarray,
    hrs: pd.DataFrame,
    repeats: int,
) -> dict[tuple[str, str], np.ndarray]:
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for model_index, model_name in enumerate(
        ["M1_single_assessment", "M2_repeat_assessment"]
    ):
        features = pilot.MODEL_FEATURES[model_name]
        charls_probabilities = pilot.repeated_oof_predictions(
            charls,
            charls_y,
            features,
            repeats,
            RANDOM_SEED + model_index,
        )
        model = pilot.build_model(features)
        model.fit(charls[features], charls_y)
        hrs_probabilities = model.predict_proba(hrs[features])[:, 1]
        predictions[("CHARLS repeated 5-fold CV", model_name)] = charls_probabilities
        predictions[("HRS external validation", model_name)] = hrs_probabilities
    return predictions


def paired_difference_rows(
    scenario: str,
    dataset: str,
    y: np.ndarray,
    predictions: dict[tuple[str, str], np.ndarray],
    bootstrap: int,
) -> list[dict[str, object]]:
    rows = pilot.bootstrap_difference(
        y,
        predictions[(dataset, "M1_single_assessment")],
        predictions[(dataset, "M2_repeat_assessment")],
        bootstrap,
        RANDOM_SEED + len(scenario) + len(dataset),
    )
    for row in rows:
        row.update(
            {
                "scenario": scenario,
                "dataset": dataset,
                "reference_model": "M1_single_assessment",
                "candidate_model": "M2_repeat_assessment",
            }
        )
    return rows


def sensitivity_scenario(
    scenario: str,
    charls_all: pd.DataFrame,
    hrs_all: pd.DataFrame,
    minimum_age: int,
    decline_sd: float,
    exclude_hrs_dementia_or_ad: bool,
    repeats: int,
    bootstrap: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    charls, charls_y, charls_index_sd = analysis_sample(
        charls_all, minimum_age, decline_sd
    )
    hrs, hrs_y, hrs_index_sd = analysis_sample(
        hrs_all, minimum_age, decline_sd, exclude_hrs_dementia_or_ad
    )
    predictions = fit_predict_pair(charls, charls_y, hrs, repeats)

    performance_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    for dataset, data, y, index_sd in (
        ("CHARLS repeated 5-fold CV", charls, charls_y, charls_index_sd),
        ("HRS external validation", hrs, hrs_y, hrs_index_sd),
    ):
        for model_name in ["M1_single_assessment", "M2_repeat_assessment"]:
            probabilities = predictions[(dataset, model_name)]
            row = {
                "scenario": scenario,
                "dataset": dataset,
                "model": model_name,
                "minimum_age": minimum_age,
                "decline_sd": decline_sd,
                "exclude_hrs_dementia_or_ad": exclude_hrs_dementia_or_ad
                if dataset.startswith("HRS")
                else False,
                "n": int(y.size),
                "events": int(y.sum()),
                "event_percent": 100.0 * float(y.mean()),
                "index_score_sd": index_sd,
            }
            row.update(metric_values(y, probabilities))
            performance_rows.append(row)
        difference_rows.extend(
            paired_difference_rows(scenario, dataset, y, predictions, bootstrap)
        )
    return performance_rows, difference_rows


def strict_hrs_ser7(raw: pd.DataFrame, wave: int) -> pd.Series:
    value = pilot.numeric(raw[f"r{wave}ser7"])
    flag = pilot.numeric(raw[f"r{wave}fser7"])
    proxy = pilot.numeric(raw[f"r{wave}proxy"])
    return value.where(flag.eq(0) & proxy.eq(0))


def load_score25_cohorts(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    charls_columns = pilot.CHARLS_COLUMNS + ["r1ser7", "r2ser7", "r3ser7"]
    hrs_main_columns = pilot.HRS_MAIN_COLUMNS + [
        "r11ser7",
        "r11fser7",
        "r12ser7",
        "r12fser7",
        "r13ser7",
        "r13fser7",
    ]
    charls_raw = pd.read_stata(
        data_dir / "H_CHARLS_D_Data" / "H_CHARLS_D_Data.dta",
        columns=charls_columns,
        convert_categoricals=False,
    )
    hrs_main = pd.read_stata(
        data_dir / "randhrs1992_2022v1.dta",
        columns=hrs_main_columns,
        convert_categoricals=False,
    )
    hrs_supplement = pd.read_stata(
        data_dir / "H_HRS_d.dta",
        columns=pilot.HRS_SUPPLEMENT_COLUMNS,
        convert_categoricals=False,
    )
    hrs_raw = hrs_main.merge(
        hrs_supplement, on="hhidpn", how="left", validate="one_to_one"
    )

    charls = pilot.derive_charls(charls_raw)
    charls["previous_cognition"] = pilot.complete_sum(
        charls_raw, ["r1imrc", "r1dlrc", "r1ser7"]
    )
    charls["index_cognition"] = pilot.complete_sum(
        charls_raw, ["r2imrc", "r2dlrc", "r2ser7"]
    )
    charls["outcome_cognition"] = pilot.complete_sum(
        charls_raw, ["r3imrc", "r3dlrc", "r3ser7"]
    )

    hrs = pilot.derive_hrs(hrs_raw)
    for wave, target in (
        (11, "previous_cognition"),
        (12, "index_cognition"),
        (13, "outcome_cognition"),
    ):
        components = pd.concat(
            [
                pilot.strict_hrs_component(
                    hrs_raw, f"r{wave}imrc", f"r{wave}fimrc", f"r{wave}proxy"
                ),
                pilot.strict_hrs_component(
                    hrs_raw, f"r{wave}dlrc", f"r{wave}fdlrc", f"r{wave}proxy"
                ),
                strict_hrs_ser7(hrs_raw, wave),
            ],
            axis=1,
        )
        hrs[target] = components.sum(axis=1).where(components.notna().all(axis=1))
    return charls, hrs


def intercept_only_shift(y: np.ndarray, probabilities: np.ndarray) -> float:
    logits = logit(probabilities)
    observed = float(y.mean())
    low, high = -20.0, 20.0
    for _ in range(100):
        mid = (low + high) / 2.0
        predicted = float(expit(logits + mid).mean())
        if predicted < observed:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def cross_validated_recalibration(
    y: np.ndarray,
    probabilities: np.ndarray,
    mode: str,
    splits: int = 5,
) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_SEED)
    logits = logit(probabilities)
    recalibrated = np.full(y.size, np.nan, dtype=float)
    for train_index, test_index in splitter.split(logits.reshape(-1, 1), y):
        train_logits = logits[train_index]
        test_logits = logits[test_index]
        if mode == "intercept_only":
            shift = intercept_only_shift(y[train_index], probabilities[train_index])
            recalibrated[test_index] = expit(test_logits + shift)
        elif mode == "intercept_slope":
            model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)
            model.fit(train_logits.reshape(-1, 1), y[train_index])
            recalibrated[test_index] = model.predict_proba(
                test_logits.reshape(-1, 1)
            )[:, 1]
        else:
            raise ValueError(f"Unknown recalibration mode: {mode}")
    if np.isnan(recalibrated).any():
        raise RuntimeError("Recalibration left missing predictions.")
    return recalibrated


def capacity_rows(
    scenario: str,
    model_name: str,
    calibration: str,
    y: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, object]]:
    rows = pilot.capacity_rows("HRS external validation", model_name, y, probabilities)
    for row in rows:
        row["scenario"] = scenario
        row["calibration"] = calibration
    return rows


def recalibration_primary_rows(
    charls_all: pd.DataFrame,
    hrs_all: pd.DataFrame,
    repeats: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    charls, charls_y, _ = analysis_sample(charls_all, 50, 0.5)
    hrs, hrs_y, _ = analysis_sample(hrs_all, 50, 0.5)
    predictions = fit_predict_pair(charls, charls_y, hrs, repeats)

    performance_rows: list[dict[str, object]] = []
    capacity_output: list[dict[str, object]] = []
    for model_name in ["M1_single_assessment", "M2_repeat_assessment"]:
        original = predictions[("HRS external validation", model_name)]
        prediction_sets = {
            "none_original_transport": original,
            "cv_intercept_only": cross_validated_recalibration(
                hrs_y, original, "intercept_only"
            ),
            "cv_intercept_slope": cross_validated_recalibration(
                hrs_y, original, "intercept_slope"
            ),
        }
        for calibration, probabilities in prediction_sets.items():
            row = {
                "scenario": "primary_20pt_age50_0.5sd",
                "dataset": "HRS external validation",
                "model": model_name,
                "calibration": calibration,
                "n": int(hrs_y.size),
                "events": int(hrs_y.sum()),
                "event_percent": 100.0 * float(hrs_y.mean()),
            }
            row.update(metric_values(hrs_y, probabilities))
            performance_rows.append(row)
            capacity_output.extend(
                capacity_rows(
                    "primary_20pt_age50_0.5sd",
                    model_name,
                    calibration,
                    hrs_y,
                    probabilities,
                )
            )
    return performance_rows, capacity_output


def format_ci(row: pd.Series) -> str:
    return f"{row['ci_low']:.4f} to {row['ci_high']:.4f}"


def write_report(
    output_dir: Path,
    sensitivity_performance: pd.DataFrame,
    sensitivity_differences: pd.DataFrame,
    recalibration: pd.DataFrame,
    capacity: pd.DataFrame,
) -> None:
    lines = [
        "# Repeat Cognition Phase 2: Sensitivity and HRS Recalibration",
        "",
        "Date: 2026-07-16",
        "",
        "Scope: prespecified M1 single-assessment vs M2 repeat-assessment analyses. "
        "This file does not include M4 machine-learning comparators or IPCW.",
        "",
        "## Sensitivity: M2 vs M1",
        "",
        "| Scenario | Dataset | Metric | Difference | 95% CI |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in sensitivity_differences.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['dataset']} | {row['metric']} | "
            f"{row['difference_candidate_minus_reference']:.4f} | {format_ci(row)} |"
        )

    lines.extend(
        [
            "",
            "## Scenario Performance",
            "",
            "| Scenario | Dataset | Model | n | Events | AUROC | AUPRC | Brier | Observed | Predicted |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in sensitivity_performance.iterrows():
        lines.append(
            f"| {row['scenario']} | {row['dataset']} | {row['model']} | "
            f"{int(row['n'])} | {int(row['events'])} | {row['auroc']:.3f} | "
            f"{row['auprc']:.3f} | {row['brier']:.3f} | "
            f"{row['observed_percent']:.1f}% | {row['mean_predicted_percent']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## HRS Recalibration",
            "",
            "| Model | Calibration | AUROC | AUPRC | Brier | Observed | Predicted | Intercept | Slope |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in recalibration.iterrows():
        lines.append(
            f"| {row['model']} | {row['calibration']} | {row['auroc']:.3f} | "
            f"{row['auprc']:.3f} | {row['brier']:.3f} | "
            f"{row['observed_percent']:.1f}% | {row['mean_predicted_percent']:.1f}% | "
            f"{row['calibration_intercept']:.3f} | {row['calibration_slope']:.3f} |"
        )

    top20 = capacity[
        (capacity["model"].eq("M2_repeat_assessment"))
        & (capacity["selected_fraction"].eq(0.2))
    ].copy()
    lines.extend(
        [
            "",
            "## HRS Capacity at Top 20% for M2",
            "",
            "| Calibration | True positives | Sensitivity | PPV |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in top20.iterrows():
        lines.append(
            f"| {row['calibration']} | {int(row['true_positive_n'])} | "
            f"{100 * row['sensitivity']:.1f}% | "
            f"{100 * row['positive_predictive_value']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The main evidential question remains whether M2 consistently improves AUROC, AUPRC and Brier score over M1.",
            "- HRS recalibration is reported as transport correction only; it is not used to choose predictors or refit the CHARLS model.",
            "- Recalibration can correct absolute risk overprediction but should not change ranking metrics materially.",
        ]
    )
    (output_dir / "repeat_cognition_phase2_sensitivity_recalibration_2026-07-16.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def update_master_plan(output_dir: Path) -> None:
    master = PROJECT_DIR / "STUDY_MASTER_PLAN.md"
    text = master.read_text(encoding="utf-8")
    marker = "## Progress Log"
    addition = """

### 2026-07-16 Phase 2 sensitivity/recalibration update

- Added `23_repeat_cognition_sensitivity_recalibration.py`.
- Completed prespecified M1 vs M2 robustness checks for: primary 20-point age >=50 0.5 SD decline, age >=65, 1 SD decline, HRS exclusion of index dementia/Alzheimer disease, and 25-point cognition score.
- Added HRS cross-validated recalibration analyses: no recalibration, intercept-only recalibration, and intercept-plus-slope recalibration.
- Saved aggregate outputs only; no person-level records were written.
- Machine-learning M4 comparison and IPCW remain pending after this phase.
"""
    if "2026-07-16 Phase 2 sensitivity/recalibration update" not in text:
        if marker in text:
            text = text.replace(marker, addition.strip() + "\n\n" + marker)
        else:
            text = text.rstrip() + "\n\n" + addition.strip() + "\n"
        master.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    charls20, hrs20 = pilot.load_cohorts(args.data_dir)
    charls25, hrs25 = load_score25_cohorts(args.data_dir)

    scenario_specs = [
        ("primary_20pt_age50_0.5sd", charls20, hrs20, 50, 0.5, False),
        ("age65_20pt_0.5sd", charls20, hrs20, 65, 0.5, False),
        ("decline_20pt_1sd_age50", charls20, hrs20, 50, 1.0, False),
        ("hrs_no_dementia_ad_20pt_age50_0.5sd", charls20, hrs20, 50, 0.5, True),
        ("score25_age50_0.5sd", charls25, hrs25, 50, 0.5, False),
    ]

    performance_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    for spec in scenario_specs:
        scenario_performance, scenario_differences = sensitivity_scenario(
            *spec,
            repeats=args.cv_repeats,
            bootstrap=args.bootstrap,
        )
        performance_rows.extend(scenario_performance)
        difference_rows.extend(scenario_differences)

    recalibration_rows, capacity_rows_output = recalibration_primary_rows(
        charls20, hrs20, args.cv_repeats
    )

    sensitivity_performance = pd.DataFrame(performance_rows)
    sensitivity_differences = pd.DataFrame(difference_rows)
    recalibration = pd.DataFrame(recalibration_rows)
    capacity = pd.DataFrame(capacity_rows_output)

    sensitivity_performance.to_csv(
        args.output_dir / "repeat_cognition_phase2_sensitivity_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensitivity_differences.to_csv(
        args.output_dir / "repeat_cognition_phase2_sensitivity_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    recalibration.to_csv(
        args.output_dir / "repeat_cognition_phase2_hrs_recalibration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    capacity.to_csv(
        args.output_dir / "repeat_cognition_phase2_hrs_capacity_recalibrated.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_report(args.output_dir, sensitivity_performance, sensitivity_differences, recalibration, capacity)
    update_master_plan(args.output_dir)

    print(sensitivity_differences.to_string(index=False))
    print(recalibration.to_string(index=False))


if __name__ == "__main__":
    main()
