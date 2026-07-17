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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


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
        description="IPCW sensitivity analysis for repeat cognition prediction."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--cv-repeats", type=int, default=5)
    return parser.parse_args()


def eligible_with_observation(data: pd.DataFrame, minimum_age: int = 50) -> pd.DataFrame:
    eligible = (
        data["baseline_age"].ge(minimum_age)
        & data["previous_cognition"].notna()
        & data["index_cognition"].notna()
    )
    sample = data.loc[eligible].copy()
    index_sd = float(
        data.loc[data["baseline_age"].ge(minimum_age) & data["index_cognition"].notna(), "index_cognition"].std(ddof=1)
    )
    sample["outcome_observed"] = sample["outcome_cognition"].notna().astype(int)
    sample["cognitive_decline"] = (
        sample["outcome_cognition"] - sample["index_cognition"] <= -0.5 * index_sd
    ).astype(float)
    sample.loc[sample["outcome_cognition"].isna(), "cognitive_decline"] = np.nan
    sample["previous_change"] = sample["index_cognition"] - sample["previous_cognition"]
    return sample


def stabilized_ipcw(data: pd.DataFrame) -> tuple[pd.Series, dict[str, float]]:
    features = pilot.M3_FEATURES
    model = pilot.build_model(features)
    observed = data["outcome_observed"].to_numpy()
    model.fit(data[features], observed)
    probability = model.predict_proba(data[features])[:, 1]
    probability = np.clip(probability, 0.02, 0.99)
    stabilized = observed.mean() / probability
    observed_weights = pd.Series(stabilized, index=data.index).where(data["outcome_observed"].eq(1))
    low = float(observed_weights.dropna().quantile(0.01))
    high = float(observed_weights.dropna().quantile(0.99))
    truncated = observed_weights.clip(low, high)
    diagnostics = {
        "eligible_n": int(data.shape[0]),
        "observed_outcome_n": int(observed.sum()),
        "observed_outcome_percent": 100.0 * float(observed.mean()),
        "raw_weight_min": float(observed_weights.min()),
        "raw_weight_p01": low,
        "raw_weight_median": float(observed_weights.median()),
        "raw_weight_p99": high,
        "raw_weight_max": float(observed_weights.max()),
        "truncated_weight_mean": float(truncated.mean()),
        "truncated_weight_min": float(truncated.min()),
        "truncated_weight_max": float(truncated.max()),
        "effective_sample_size": float((truncated.sum() ** 2) / (truncated.pow(2).sum())),
    }
    return truncated, diagnostics


def calibration_parameters_weighted(
    y: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)
    model.fit(logits, y, sample_weight=weights)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def weighted_metrics(
    y: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    intercept, slope = calibration_parameters_weighted(y, probabilities, weights)
    return {
        "auroc": float(roc_auc_score(y, probabilities, sample_weight=weights)),
        "auprc": float(average_precision_score(y, probabilities, sample_weight=weights)),
        "brier": float(brier_score_loss(y, probabilities, sample_weight=weights)),
        "observed_percent": 100.0 * float(np.average(y, weights=weights)),
        "mean_predicted_percent": 100.0 * float(np.average(probabilities, weights=weights)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def bootstrap_weighted_difference(
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray,
    replicates: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    metric_functions = {
        "auroc": lambda yy, pp, ww: roc_auc_score(yy, pp, sample_weight=ww),
        "auprc": lambda yy, pp, ww: average_precision_score(yy, pp, sample_weight=ww),
        "brier": lambda yy, pp, ww: brier_score_loss(yy, pp, sample_weight=ww),
    }
    observed = {
        metric: function(y, candidate, weights) - function(y, reference, weights)
        for metric, function in metric_functions.items()
    }
    draws = {metric: [] for metric in metric_functions}
    for _ in range(replicates):
        index = rng.integers(0, y.size, y.size)
        if np.unique(y[index]).size < 2:
            continue
        for metric, function in metric_functions.items():
            draws[metric].append(
                function(y[index], candidate[index], weights[index])
                - function(y[index], reference[index], weights[index])
            )
    rows = []
    for metric, values in draws.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "difference_candidate_minus_reference": observed[metric],
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
                "bootstrap_replicates_used": int(arr.size),
            }
        )
    return rows


def model_predictions(
    charls_observed: pd.DataFrame,
    charls_y: np.ndarray,
    hrs_observed: pd.DataFrame,
    repeats: int,
) -> dict[tuple[str, str], np.ndarray]:
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for idx, model_name in enumerate(["M1_single_assessment", "M2_repeat_assessment"]):
        features = pilot.MODEL_FEATURES[model_name]
        charls_prob = pilot.repeated_oof_predictions(
            charls_observed,
            charls_y,
            features,
            repeats,
            RANDOM_SEED + idx,
        )
        model = pilot.build_model(features)
        model.fit(charls_observed[features], charls_y)
        hrs_prob = model.predict_proba(hrs_observed[features])[:, 1]
        predictions[("CHARLS repeated 5-fold CV", model_name)] = charls_prob
        predictions[("HRS external validation", model_name)] = hrs_prob
    return predictions


def run_dataset(
    dataset: str,
    data: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    predictions: dict[tuple[str, str], np.ndarray],
    bootstrap: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    performance_rows: list[dict[str, object]] = []
    for model_name in ["M1_single_assessment", "M2_repeat_assessment"]:
        probabilities = predictions[(dataset, model_name)]
        row = {
            "dataset": dataset,
            "model": model_name,
            "n": int(y.size),
            "events": int(y.sum()),
            "unweighted_event_percent": 100.0 * float(y.mean()),
            "weight_mean": float(weights.mean()),
            "effective_sample_size": float((weights.sum() ** 2) / np.square(weights).sum()),
        }
        row.update(weighted_metrics(y, probabilities, weights))
        performance_rows.append(row)

    difference_rows = bootstrap_weighted_difference(
        y,
        predictions[(dataset, "M1_single_assessment")],
        predictions[(dataset, "M2_repeat_assessment")],
        weights,
        bootstrap,
        RANDOM_SEED + len(dataset),
    )
    for row in difference_rows:
        row.update(
            {
                "dataset": dataset,
                "reference_model": "M1_single_assessment",
                "candidate_model": "M2_repeat_assessment",
            }
        )
    return performance_rows, difference_rows


def write_report(
    performance: pd.DataFrame,
    differences: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = [
        "# Repeat Cognition Phase 4: IPCW Sensitivity",
        "",
        "Date: 2026-07-16",
        "",
        "Scope: inverse-probability-of-cognitive-outcome-observation weighting among participants eligible at index. Models remain M1 and M2.",
        "",
        "## Observation Model Diagnostics",
        "",
        "| Dataset | Eligible | Observed outcome | Observed % | Effective n | Weight range after truncation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in diagnostics.iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['eligible_n'])} | {int(row['observed_outcome_n'])} | "
            f"{row['observed_outcome_percent']:.1f}% | {row['effective_sample_size']:.0f} | "
            f"{row['truncated_weight_min']:.2f} to {row['truncated_weight_max']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Weighted Performance",
            "",
            "| Dataset | Model | n | Weighted event | AUROC | AUPRC | Brier | Predicted |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in performance.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['model']} | {int(row['n'])} | "
            f"{row['observed_percent']:.1f}% | {row['auroc']:.3f} | "
            f"{row['auprc']:.3f} | {row['brier']:.3f} | "
            f"{row['mean_predicted_percent']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Weighted M2 vs M1 Differences",
            "",
            "| Dataset | Metric | Difference | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for _, row in differences.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['metric']} | "
            f"{row['difference_candidate_minus_reference']:.4f} | "
            f"{row['ci_low']:.4f} to {row['ci_high']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- IPCW is a sensitivity analysis for outcome observation, not a new primary model.",
            "- Persistence of M2 improvement after weighting supports robustness against measured attrition bias.",
        ]
    )
    (output_dir / "repeat_cognition_phase4_ipcw_sensitivity_2026-07-16.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def update_master_plan() -> None:
    master = PROJECT_DIR / "STUDY_MASTER_PLAN.md"
    text = master.read_text(encoding="utf-8")
    addition = """

### 2026-07-16 Phase 4 IPCW update

- Added `25_repeat_cognition_ipcw_sensitivity.py`.
- Completed inverse-probability-of-cognitive-outcome-observation weighted sensitivity analysis for CHARLS and HRS.
- M1 and M2 model training remained unchanged; IPCW was applied to performance estimation among observed outcomes.
- Weights were stabilized and truncated at the 1st and 99th percentiles.
- Decision-curve plots, manuscript figures and ELSA confirmatory validation remain pending.
"""
    if "2026-07-16 Phase 4 IPCW update" not in text:
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
    charls_eligible = eligible_with_observation(charls_all)
    hrs_eligible = eligible_with_observation(hrs_all)

    charls_weights_all, charls_diag = stabilized_ipcw(charls_eligible)
    hrs_weights_all, hrs_diag = stabilized_ipcw(hrs_eligible)
    charls_diag["dataset"] = "CHARLS"
    hrs_diag["dataset"] = "HRS"

    charls_observed = charls_eligible.loc[charls_eligible["outcome_observed"].eq(1)].copy()
    hrs_observed = hrs_eligible.loc[hrs_eligible["outcome_observed"].eq(1)].copy()
    charls_y = charls_observed["cognitive_decline"].astype(int).to_numpy()
    hrs_y = hrs_observed["cognitive_decline"].astype(int).to_numpy()
    charls_weights = charls_weights_all.loc[charls_observed.index].to_numpy(dtype=float)
    hrs_weights = hrs_weights_all.loc[hrs_observed.index].to_numpy(dtype=float)

    predictions = model_predictions(charls_observed, charls_y, hrs_observed, args.cv_repeats)

    performance_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    for dataset, data, y, weights in (
        ("CHARLS repeated 5-fold CV", charls_observed, charls_y, charls_weights),
        ("HRS external validation", hrs_observed, hrs_y, hrs_weights),
    ):
        perf, diffs = run_dataset(dataset, data, y, weights, predictions, args.bootstrap)
        performance_rows.extend(perf)
        difference_rows.extend(diffs)

    performance = pd.DataFrame(performance_rows)
    differences = pd.DataFrame(difference_rows)
    diagnostics = pd.DataFrame([charls_diag, hrs_diag])

    performance.to_csv(
        args.output_dir / "repeat_cognition_phase4_ipcw_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    differences.to_csv(
        args.output_dir / "repeat_cognition_phase4_ipcw_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    diagnostics.to_csv(
        args.output_dir / "repeat_cognition_phase4_ipcw_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_report(performance, differences, diagnostics, args.output_dir)
    update_master_plan()
    print(diagnostics.to_string(index=False))
    print(performance.to_string(index=False))
    print(differences.to_string(index=False))


if __name__ == "__main__":
    main()
