from __future__ import annotations

import argparse
import importlib.util
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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


DEFAULT_DATA_DIR = PROJECT_DIR / "data_link"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
RANDOM_SEED = 20260716


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load_module("22_repeat_cognition_primary_pilot.py", "pilot_formal")
phase2 = load_module(
    "23_repeat_cognition_sensitivity_recalibration.py", "phase2_formal"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create formal aggregate external-validation and reporting tables."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild the narrative report from previously generated aggregate CSVs.",
    )
    return parser.parse_args()


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def calibration_parameters_fast(
    y: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    x = logit(probabilities)
    beta = np.array([0.0, 1.0], dtype=float)
    for _ in range(30):
        eta = np.clip(beta[0] + beta[1] * x, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        weight = np.maximum(mu * (1 - mu), 1e-10)
        score = np.array(
            [(y - mu).sum(), ((y - mu) * x).sum()], dtype=float
        )
        information = np.array(
            [
                [weight.sum(), (weight * x).sum()],
                [(weight * x).sum(), (weight * x * x).sum()],
            ],
            dtype=float,
        )
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def metric_values(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration_parameters_fast(y, probabilities)
    return {
        "auroc": float(roc_auc_score(y, probabilities)),
        "auprc": float(average_precision_score(y, probabilities)),
        "brier": float(brier_score_loss(y, probabilities)),
        "observed_percent": 100.0 * float(y.mean()),
        "mean_predicted_percent": 100.0 * float(probabilities.mean()),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def bootstrap_absolute_metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    names = [
        "auroc",
        "auprc",
        "brier",
        "observed_percent",
        "mean_predicted_percent",
        "calibration_intercept",
        "calibration_slope",
    ]
    draws = {name: [] for name in names}
    for _ in range(replicates):
        index = rng.integers(0, y.size, y.size)
        if np.unique(y[index]).size < 2:
            continue
        values = metric_values(y[index], probabilities[index])
        for name in names:
            if np.isfinite(values[name]):
                draws[name].append(values[name])
    return {
        name: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for name, values in draws.items()
    }


def absolute_performance_rows(
    prediction_sets: list[tuple[str, str, str, np.ndarray, np.ndarray]],
    bootstrap: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (dataset, model, calibration, y, probabilities) in enumerate(
        prediction_sets
    ):
        values = metric_values(y, probabilities)
        intervals = bootstrap_absolute_metrics(
            y, probabilities, bootstrap, RANDOM_SEED + 100 + index
        )
        row: dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "calibration": calibration,
            "n": int(y.size),
            "events": int(y.sum()),
            "bootstrap_replicates": bootstrap,
        }
        for name, value in values.items():
            row[name] = value
            row[f"{name}_ci_low"] = intervals[name][0]
            row[f"{name}_ci_high"] = intervals[name][1]
        rows.append(row)
    return rows


def paired_difference_rows(
    dataset: str,
    calibration: str,
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    rows = pilot.bootstrap_difference(
        y, reference, candidate, bootstrap, seed
    )
    for row in rows:
        row.update(
            {
                "dataset": dataset,
                "calibration": calibration,
                "reference_model": "M1_single_assessment",
                "candidate_model": "M2_repeat_assessment",
            }
        )
    return rows


def subgroup_rows(
    data: pd.DataFrame,
    y: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    m2_recalibrated: np.ndarray,
    bootstrap: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    row_index = 0
    for subgroup, labels in pilot.subgroup_labels(data).items():
        for level in sorted(labels.unique().tolist()):
            mask = labels.eq(level).to_numpy()
            subgroup_y = y[mask]
            if subgroup_y.size < 100 or np.unique(subgroup_y).size < 2:
                continue
            m1_values = metric_values(subgroup_y, m1[mask])
            m2_values = metric_values(subgroup_y, m2[mask])
            recalibrated_values = metric_values(
                subgroup_y, m2_recalibrated[mask]
            )
            differences = pilot.bootstrap_difference(
                subgroup_y,
                m1[mask],
                m2[mask],
                bootstrap,
                RANDOM_SEED + 300 + row_index,
            )
            difference_lookup = {row["metric"]: row for row in differences}
            row = {
                "subgroup": subgroup,
                "level": level,
                "n": int(subgroup_y.size),
                "events": int(subgroup_y.sum()),
                "event_percent": 100.0 * float(subgroup_y.mean()),
                "m1_auroc": m1_values["auroc"],
                "m2_auroc": m2_values["auroc"],
                "delta_auroc": difference_lookup["auroc"][
                    "difference_candidate_minus_reference"
                ],
                "delta_auroc_ci_low": difference_lookup["auroc"]["ci_low"],
                "delta_auroc_ci_high": difference_lookup["auroc"]["ci_high"],
                "m1_auprc": m1_values["auprc"],
                "m2_auprc": m2_values["auprc"],
                "delta_auprc": difference_lookup["auprc"][
                    "difference_candidate_minus_reference"
                ],
                "delta_auprc_ci_low": difference_lookup["auprc"]["ci_low"],
                "delta_auprc_ci_high": difference_lookup["auprc"]["ci_high"],
                "m1_brier": m1_values["brier"],
                "m2_brier": m2_values["brier"],
                "delta_brier": difference_lookup["brier"][
                    "difference_candidate_minus_reference"
                ],
                "delta_brier_ci_low": difference_lookup["brier"]["ci_low"],
                "delta_brier_ci_high": difference_lookup["brier"]["ci_high"],
                "m2_original_mean_predicted_percent": m2_values[
                    "mean_predicted_percent"
                ],
                "m2_recalibrated_mean_predicted_percent": recalibrated_values[
                    "mean_predicted_percent"
                ],
                "m2_recalibrated_calibration_intercept": recalibrated_values[
                    "calibration_intercept"
                ],
                "m2_recalibrated_calibration_slope": recalibrated_values[
                    "calibration_slope"
                ],
                "bootstrap_replicates": bootstrap,
            }
            rows.append(row)
            row_index += 1
    return rows


def top_fraction_selection(probabilities: np.ndarray, fraction: float) -> np.ndarray:
    selected_n = int(np.ceil(fraction * probabilities.size))
    order = np.argsort(-probabilities, kind="mergesort")
    selected = np.zeros(probabilities.size, dtype=bool)
    selected[order[:selected_n]] = True
    return selected


def capacity_metric_values(
    y: np.ndarray, labels: np.ndarray, selected: np.ndarray
) -> dict[str, float]:
    mask = labels.astype(bool)
    group_y = y[mask]
    group_selected = selected[mask]
    true_positive = int(((group_y == 1) & group_selected).sum())
    selected_n = int(group_selected.sum())
    events = int(group_y.sum())
    return {
        "selection_percent": 100.0 * selected_n / group_y.size,
        "sensitivity": true_positive / events if events else np.nan,
        "positive_predictive_value": (
            true_positive / selected_n if selected_n else np.nan
        ),
    }


def fixed_capacity_rows(
    data: pd.DataFrame,
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    bootstrap: int,
    fraction: float = 0.20,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    labels_by_subgroup = pilot.subgroup_labels(data)
    labels_by_subgroup["overall"] = pd.Series(
        "all", index=data.index, dtype="string"
    )
    group_masks: list[tuple[str, str, np.ndarray]] = []
    for subgroup, labels in labels_by_subgroup.items():
        for level in sorted(labels.unique().tolist()):
            mask = labels.eq(level).to_numpy()
            if mask.sum() >= 100 and y[mask].sum() > 0:
                group_masks.append((subgroup, level, mask))

    metric_names = [
        "selection_percent",
        "sensitivity",
        "positive_predictive_value",
    ]
    for model_name, probabilities in predictions.items():
        selected = top_fraction_selection(probabilities, fraction)
        draws = {
            (subgroup, level): {name: [] for name in metric_names}
            for subgroup, level, _ in group_masks
        }
        rng = np.random.default_rng(
            RANDOM_SEED + 500 + (0 if model_name.startswith("M1") else 1)
        )
        for _ in range(bootstrap):
            index = rng.integers(0, y.size, y.size)
            bootstrap_selected = top_fraction_selection(
                probabilities[index], fraction
            )
            for subgroup, level, mask in group_masks:
                bootstrap_mask = mask[index]
                if not bootstrap_mask.any() or y[index][bootstrap_mask].sum() == 0:
                    continue
                values = capacity_metric_values(
                    y[index], bootstrap_mask, bootstrap_selected
                )
                for name in metric_names:
                    if np.isfinite(values[name]):
                        draws[(subgroup, level)][name].append(values[name])

        for subgroup, level, mask in group_masks:
            values = capacity_metric_values(y, mask, selected)
            row: dict[str, object] = {
                "model": model_name,
                "subgroup": subgroup,
                "level": level,
                "n": int(mask.sum()),
                "events": int(y[mask].sum()),
                "global_selected_fraction": fraction,
                "bootstrap_replicates": bootstrap,
            }
            for name, value in values.items():
                metric_draws = draws[(subgroup, level)][name]
                row[name] = value
                row[f"{name}_ci_low"] = float(
                    np.quantile(metric_draws, 0.025)
                )
                row[f"{name}_ci_high"] = float(
                    np.quantile(metric_draws, 0.975)
                )
            rows.append(row)
    return rows


def decision_curve_rows(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prevalence = float(y.mean())
    n = y.size
    for threshold in np.arange(0.10, 0.51, 0.05):
        odds = threshold / (1 - threshold)
        treat_all = prevalence - (1 - prevalence) * odds
        for model_name, probabilities in predictions.items():
            selected = probabilities >= threshold
            tp = int(((y == 1) & selected).sum())
            fp = int(((y == 0) & selected).sum())
            rows.append(
                {
                    "dataset": "HRS external validation",
                    "calibration": "cv_intercept_slope",
                    "model": model_name,
                    "threshold": round(float(threshold), 2),
                    "selected_n": int(selected.sum()),
                    "true_positive_n": tp,
                    "false_positive_n": fp,
                    "net_benefit": tp / n - fp / n * odds,
                    "treat_all_net_benefit": treat_all,
                    "treat_none_net_benefit": 0.0,
                }
            )
    return rows


def mean_sd(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return f"{values.mean():.1f} ({values.std(ddof=1):.1f})"


def n_percent(series: pd.Series, value: float = 1.0) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    count = int(values.eq(value).sum())
    return f"{count} ({100.0 * count / values.size:.1f}%)"


def cohort_characteristic_rows(
    cohorts: dict[str, tuple[pd.DataFrame, np.ndarray]]
) -> list[dict[str, str]]:
    definitions: list[tuple[str, str, float | None]] = [
        ("Age at index assessment, mean (SD), y", "age", None),
        ("Female sex, No. (%)", "female", 1.0),
        ("Education: low, No. (%)", "education_level", 1.0),
        ("Education: middle, No. (%)", "education_level", 2.0),
        ("Education: high, No. (%)", "education_level", 3.0),
        ("Rural residence, No. (%)", "rural", 1.0),
        ("Previous memory score, mean (SD)", "previous_cognition", None),
        ("Index memory score, mean (SD)", "index_cognition", None),
        ("Previous-to-index change, mean (SD)", "previous_change", None),
        ("Outcome memory score, mean (SD)", "outcome_cognition", None),
        ("Hypertension, No. (%)", "hypertension", 1.0),
        ("Diabetes, No. (%)", "diabetes", 1.0),
        ("Heart disease, No. (%)", "heart_disease", 1.0),
        ("Stroke, No. (%)", "stroke", 1.0),
        ("ADL difficulty count, mean (SD)", "adl_difficulty_count", None),
    ]
    rows: list[dict[str, str]] = []
    sample_row: dict[str, str] = {"characteristic": "Participants, No."}
    event_row: dict[str, str] = {"characteristic": "Memory decline, No. (%)"}
    for cohort, (data, y) in cohorts.items():
        sample_row[cohort] = str(data.shape[0])
        event_row[cohort] = f"{int(y.sum())} ({100.0 * y.mean():.1f}%)"
    rows.extend([sample_row, event_row])
    for label, column, category in definitions:
        row = {"characteristic": label}
        for cohort, (data, _) in cohorts.items():
            row[cohort] = (
                mean_sd(data[column])
                if category is None
                else n_percent(data[column], category)
            )
        rows.append(row)
    return rows


def predictor_missingness_rows(
    cohorts: dict[str, tuple[pd.DataFrame, np.ndarray]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cohort, (data, _) in cohorts.items():
        for variable in pilot.M3_FEATURES:
            missing = int(data[variable].isna().sum())
            rows.append(
                {
                    "cohort": cohort,
                    "variable": variable,
                    "n": int(data.shape[0]),
                    "missing_n": missing,
                    "missing_percent": 100.0 * missing / data.shape[0],
                }
            )
    return rows


def format_ci(estimate: float, low: float, high: float, digits: int = 3) -> str:
    return f"{estimate:.{digits}f} ({low:.{digits}f} to {high:.{digits}f})"


def write_report(
    output_dir: Path,
    performance: pd.DataFrame,
    differences: pd.DataFrame,
    subgroups: pd.DataFrame,
    capacity: pd.DataFrame,
    decision: pd.DataFrame,
) -> None:
    hrs_m2 = performance[
        performance["dataset"].eq("HRS external validation")
        & performance["model"].eq("M2_repeat_assessment")
    ]
    original = hrs_m2[hrs_m2["calibration"].eq("none_original_transport")].iloc[0]
    recalibrated = hrs_m2[
        hrs_m2["calibration"].eq("cv_intercept_slope")
    ].iloc[0]
    hrs_delta = differences[
        differences["dataset"].eq("HRS external validation")
        & differences["calibration"].eq("none_original_transport")
    ].set_index("metric")
    charls_delta = differences[
        differences["dataset"].eq("CHARLS repeated 5-fold CV")
        & differences["calibration"].eq("none_internal_cv")
    ].set_index("metric")
    overall_capacity = capacity[capacity["subgroup"].eq("overall")].set_index(
        "model"
    )
    useful_thresholds = []
    for threshold, group in decision.groupby("threshold"):
        lookup = group.set_index("model")
        m2 = lookup.loc["M2_repeat_assessment", "net_benefit"]
        m1 = lookup.loc["M1_single_assessment", "net_benefit"]
        treat_all = lookup.loc[
            "M2_repeat_assessment", "treat_all_net_benefit"
        ]
        if m2 > m1 and m2 > max(treat_all, 0.0):
            useful_thresholds.append(float(threshold))
    subgroup_min = subgroups.loc[subgroups["delta_auroc"].idxmin()]
    subgroup_max = subgroups.loc[subgroups["delta_auroc"].idxmax()]
    lines = [
        "# Formal CHARLS-HRS External Validation Summary",
        "",
        "Date: 2026-07-16",
        "",
        "## Primary Finding",
        "",
        (
            "Adding the previous memory assessment improved discrimination and overall "
            "prediction error in both cohorts. The prespecified M2 vs M1 AUROC difference "
            f"was {format_ci(charls_delta.loc['auroc', 'difference_candidate_minus_reference'], charls_delta.loc['auroc', 'ci_low'], charls_delta.loc['auroc', 'ci_high'], 4)} in CHARLS and "
            f"{format_ci(hrs_delta.loc['auroc', 'difference_candidate_minus_reference'], hrs_delta.loc['auroc', 'ci_low'], hrs_delta.loc['auroc', 'ci_high'], 4)} in HRS."
        ),
        "",
        "## HRS Transport and Recalibration",
        "",
        (
            f"Original M2 transport: observed risk {original['observed_percent']:.1f}%, "
            f"mean predicted risk {original['mean_predicted_percent']:.1f}%, "
            f"AUROC {original['auroc']:.3f}, Brier {original['brier']:.3f}, "
            f"calibration intercept {original['calibration_intercept']:.3f}, and slope "
            f"{original['calibration_slope']:.3f}."
        ),
        (
            f"Cross-validated intercept-plus-slope recalibration: mean predicted risk "
            f"{recalibrated['mean_predicted_percent']:.1f}%, Brier "
            f"{recalibrated['brier']:.3f}, calibration intercept "
            f"{recalibrated['calibration_intercept']:.3f}, and slope "
            f"{recalibrated['calibration_slope']:.3f}."
        ),
        "",
        "## Fixed Screening Capacity",
        "",
        (
            "When the highest-risk 20% of HRS participants were selected, M1 identified "
            f"{100.0 * overall_capacity.loc['M1_single_assessment', 'sensitivity']:.1f}% "
            "of decline events and M2 identified "
            f"{100.0 * overall_capacity.loc['M2_repeat_assessment', 'sensitivity']:.1f}%. "
            f"Positive predictive value increased from {100.0 * overall_capacity.loc['M1_single_assessment', 'positive_predictive_value']:.1f}% to "
            f"{100.0 * overall_capacity.loc['M2_repeat_assessment', 'positive_predictive_value']:.1f}%."
        ),
        "",
        "## Subgroup Consistency",
        "",
        (
            f"Across prespecified HRS subgroup levels, the point estimate for AUROC gain "
            f"ranged from {subgroup_min['delta_auroc']:.3f} in "
            f"{subgroup_min['subgroup']}={subgroup_min['level']} to "
            f"{subgroup_max['delta_auroc']:.3f} in "
            f"{subgroup_max['subgroup']}={subgroup_max['level']}."
        ),
        "Subgroup estimates are descriptive; the study is not powered for interaction claims.",
        "",
        "## Decision Curve",
        "",
        (
            "After cross-validated local recalibration, M2 had higher net benefit than "
            "M1, treat-all, and treat-none at the following evaluated thresholds: "
            + (", ".join(f"{value:.0%}" for value in useful_thresholds) or "none")
            + "."
        ),
        "",
        "## Reporting Position",
        "",
        "- The primary manuscript claim should remain incremental value of repeat objective cognition.",
        "- Original HRS transport and locally recalibrated HRS performance must be reported separately.",
        "- Fixed-capacity results support a clinically interpretable screening argument without claiming treatment benefit.",
        "- ELSA remains a later confirmatory validation and was not used in these analyses.",
    ]
    (output_dir / "repeat_cognition_formal_external_validation_2026-07-16.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def update_master_plan() -> None:
    master = PROJECT_DIR / "STUDY_MASTER_PLAN.md"
    text = master.read_text(encoding="utf-8")
    addition = """
### 2026-07-16 Phase 7 formal external-validation reporting update

- ELSA access was deferred because the UK Data Service download workflow was not yet complete.
- Added `28_formal_external_validation_and_tables.py`.
- Added bootstrap confidence intervals for absolute performance, formal HRS subgroup comparisons, fixed-capacity screening metrics, and decision-curve summaries using cross-validated local recalibration.
- Generated manuscript-oriented aggregate Table 1, missingness, performance, subgroup, capacity, and decision-curve files.
- ELSA remains reserved for later confirmatory validation with the frozen model and reporting code.
""".strip()
    if "2026-07-16 Phase 7 formal external-validation reporting update" not in text:
        master.write_text(text.rstrip() + "\n\n" + addition + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        performance = pd.read_csv(
            args.output_dir / "repeat_cognition_formal_absolute_performance.csv"
        )
        differences = pd.read_csv(
            args.output_dir / "repeat_cognition_formal_paired_differences.csv"
        )
        subgroups = pd.read_csv(
            args.output_dir / "repeat_cognition_formal_hrs_subgroups.csv"
        )
        capacity = pd.read_csv(
            args.output_dir / "repeat_cognition_formal_hrs_fixed_capacity.csv"
        )
        decision = pd.read_csv(
            args.output_dir / "repeat_cognition_formal_hrs_decision_curve.csv"
        )
        write_report(
            args.output_dir,
            performance,
            differences,
            subgroups,
            capacity,
            decision,
        )
        update_master_plan()
        print("Rebuilt formal report from aggregate CSVs.")
        return

    charls_all, hrs_all = pilot.load_cohorts(args.data_dir)
    charls, charls_y, _ = pilot.analysis_sample(charls_all, minimum_age=50)
    hrs, hrs_y, _ = pilot.analysis_sample(hrs_all, minimum_age=50)
    predictions = phase2.fit_predict_pair(
        charls, charls_y, hrs, args.cv_repeats
    )
    charls_m1 = predictions[("CHARLS repeated 5-fold CV", "M1_single_assessment")]
    charls_m2 = predictions[("CHARLS repeated 5-fold CV", "M2_repeat_assessment")]
    hrs_m1 = predictions[("HRS external validation", "M1_single_assessment")]
    hrs_m2 = predictions[("HRS external validation", "M2_repeat_assessment")]
    hrs_m1_recalibrated = phase2.cross_validated_recalibration(
        hrs_y, hrs_m1, "intercept_slope"
    )
    hrs_m2_recalibrated = phase2.cross_validated_recalibration(
        hrs_y, hrs_m2, "intercept_slope"
    )

    performance = pd.DataFrame(
        absolute_performance_rows(
            [
                (
                    "CHARLS repeated 5-fold CV",
                    "M1_single_assessment",
                    "none_internal_cv",
                    charls_y,
                    charls_m1,
                ),
                (
                    "CHARLS repeated 5-fold CV",
                    "M2_repeat_assessment",
                    "none_internal_cv",
                    charls_y,
                    charls_m2,
                ),
                (
                    "HRS external validation",
                    "M1_single_assessment",
                    "none_original_transport",
                    hrs_y,
                    hrs_m1,
                ),
                (
                    "HRS external validation",
                    "M2_repeat_assessment",
                    "none_original_transport",
                    hrs_y,
                    hrs_m2,
                ),
                (
                    "HRS external validation",
                    "M1_single_assessment",
                    "cv_intercept_slope",
                    hrs_y,
                    hrs_m1_recalibrated,
                ),
                (
                    "HRS external validation",
                    "M2_repeat_assessment",
                    "cv_intercept_slope",
                    hrs_y,
                    hrs_m2_recalibrated,
                ),
            ],
            args.bootstrap,
        )
    )

    difference_rows: list[dict[str, object]] = []
    difference_rows.extend(
        paired_difference_rows(
            "CHARLS repeated 5-fold CV",
            "none_internal_cv",
            charls_y,
            charls_m1,
            charls_m2,
            args.bootstrap,
            RANDOM_SEED + 700,
        )
    )
    difference_rows.extend(
        paired_difference_rows(
            "HRS external validation",
            "none_original_transport",
            hrs_y,
            hrs_m1,
            hrs_m2,
            args.bootstrap,
            RANDOM_SEED + 701,
        )
    )
    difference_rows.extend(
        paired_difference_rows(
            "HRS external validation",
            "cv_intercept_slope",
            hrs_y,
            hrs_m1_recalibrated,
            hrs_m2_recalibrated,
            args.bootstrap,
            RANDOM_SEED + 702,
        )
    )
    differences = pd.DataFrame(difference_rows)
    subgroups = pd.DataFrame(
        subgroup_rows(
            hrs,
            hrs_y,
            hrs_m1,
            hrs_m2,
            hrs_m2_recalibrated,
            args.bootstrap,
        )
    )
    capacity = pd.DataFrame(
        fixed_capacity_rows(
            hrs,
            hrs_y,
            {
                "M1_single_assessment": hrs_m1,
                "M2_repeat_assessment": hrs_m2,
            },
            args.bootstrap,
        )
    )
    decision = pd.DataFrame(
        decision_curve_rows(
            hrs_y,
            {
                "M1_single_assessment": hrs_m1_recalibrated,
                "M2_repeat_assessment": hrs_m2_recalibrated,
            },
        )
    )
    characteristics = pd.DataFrame(
        cohort_characteristic_rows(
            {"CHARLS": (charls, charls_y), "HRS": (hrs, hrs_y)}
        )
    )
    missingness = pd.DataFrame(
        predictor_missingness_rows(
            {"CHARLS": (charls, charls_y), "HRS": (hrs, hrs_y)}
        )
    )

    outputs = {
        "repeat_cognition_table1_cohort_characteristics.csv": characteristics,
        "repeat_cognition_supplement_predictor_missingness.csv": missingness,
        "repeat_cognition_formal_absolute_performance.csv": performance,
        "repeat_cognition_formal_paired_differences.csv": differences,
        "repeat_cognition_formal_hrs_subgroups.csv": subgroups,
        "repeat_cognition_formal_hrs_fixed_capacity.csv": capacity,
        "repeat_cognition_formal_hrs_decision_curve.csv": decision,
    }
    for filename, table in outputs.items():
        table.to_csv(args.output_dir / filename, index=False, encoding="utf-8-sig")

    write_report(
        args.output_dir, performance, differences, subgroups, capacity, decision
    )
    update_master_plan()
    print(performance[["dataset", "model", "calibration", "auroc", "auprc", "brier"]].to_string(index=False))
    print(differences.to_string(index=False))


if __name__ == "__main__":
    main()
