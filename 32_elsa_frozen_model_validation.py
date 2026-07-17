from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_PACKAGES = PROJECT_DIR / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import joblib
import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = PROJECT_DIR / "data_link"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
DEFAULT_MODEL_DIR = PROJECT_DIR / "model_artifacts"
RANDOM_SEED = 20260716


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


elsa_audit = load_module("31_elsa_data_inventory_and_validation.py", "elsa_audit")
pilot = load_module("22_repeat_cognition_primary_pilot.py", "pilot_elsa")
phase2 = load_module("23_repeat_cognition_sensitivity_recalibration.py", "phase2_elsa")
formal = load_module("28_formal_external_validation_and_tables.py", "formal_elsa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply frozen CHARLS M1/M2 models to the locked ELSA sample."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_models(model_dir: Path) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    metadata_path = model_dir / "frozen_model_metadata_2026-07-16.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    models: dict[str, object] = {}
    verification_rows = []
    for model_name, details in metadata["models"].items():
        path = PROJECT_DIR / details["artifact"]
        observed_hash = sha256(path)
        expected_hash = details["artifact_sha256"]
        if observed_hash != expected_hash:
            raise RuntimeError(f"SHA-256 verification failed for {path.name}.")
        models[model_name] = joblib.load(path)
        verification_rows.append(
            {
                "model": model_name,
                "artifact": str(path.relative_to(PROJECT_DIR)),
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "hash_verified": True,
                "freeze_date": metadata["freeze_date"],
                "development_cohort": metadata["development_cohort"],
            }
        )
    return models, pd.DataFrame(verification_rows), metadata


def subgroup_labels(data: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "sex": data["female"].map({0.0: "male", 1.0: "female"}).fillna("missing"),
        "age_group": pd.cut(
            data["age"],
            bins=[-np.inf, 64.999, 74.999, np.inf],
            labels=["50-64", "65-74", "75+"],
        )
        .astype("string")
        .fillna("missing"),
        "education": data["education_level"]
        .map({1.0: "low", 2.0: "middle", 3.0: "high"})
        .fillna("missing"),
    }


def subgroup_rows(
    data: pd.DataFrame,
    y: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    bootstrap: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_index = 0
    for subgroup, labels in subgroup_labels(data).items():
        for level in sorted(labels.unique().tolist()):
            mask = labels.eq(level).to_numpy()
            subgroup_y = y[mask]
            if subgroup_y.size < 100 or np.unique(subgroup_y).size < 2:
                continue
            m1_values = formal.metric_values(subgroup_y, m1[mask])
            m2_values = formal.metric_values(subgroup_y, m2[mask])
            differences = pilot.bootstrap_difference(
                subgroup_y,
                m1[mask],
                m2[mask],
                bootstrap,
                RANDOM_SEED + 700 + row_index,
            )
            difference_lookup = {row["metric"]: row for row in differences}
            rows.append(
                {
                    "subgroup": subgroup,
                    "level": level,
                    "n": int(subgroup_y.size),
                    "events": int(subgroup_y.sum()),
                    "event_percent": 100 * float(subgroup_y.mean()),
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
                    "bootstrap_replicates": bootstrap,
                }
            )
            row_index += 1
    return pd.DataFrame(rows)


def capacity_table(
    y: np.ndarray, prediction_sets: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for name, probabilities in prediction_sets.items():
        model_name, calibration = name.split("|", maxsplit=1)
        model_rows = pilot.capacity_rows(
            "ELSA confirmatory validation", model_name, y, probabilities
        )
        for row in model_rows:
            row["calibration"] = calibration
            rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(
    y: np.ndarray, prediction_sets: dict[str, np.ndarray], bins: int = 10
) -> pd.DataFrame:
    rows = []
    for name, probabilities in prediction_sets.items():
        model_name, calibration = name.split("|", maxsplit=1)
        frame = pd.DataFrame({"y": y, "p": probabilities})
        frame["bin"] = pd.qcut(
            frame["p"].rank(method="first"), bins, labels=False, duplicates="drop"
        )
        for bin_number, group in frame.groupby("bin", observed=True):
            rows.append(
                {
                    "model": model_name,
                    "calibration": calibration,
                    "risk_bin": int(bin_number) + 1,
                    "n": int(len(group)),
                    "events": int(group["y"].sum()),
                    "mean_predicted": float(group["p"].mean()),
                    "observed_proportion": float(group["y"].mean()),
                    "minimum_predicted": float(group["p"].min()),
                    "maximum_predicted": float(group["p"].max()),
                }
            )
    return pd.DataFrame(rows)


def decision_curve_table(
    y: np.ndarray, prediction_sets: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    n = y.size
    prevalence = float(y.mean())
    for threshold in np.arange(0.05, 0.601, 0.025):
        odds = threshold / (1 - threshold)
        treat_all = prevalence - (1 - prevalence) * odds
        rows.append(
            {
                "model": "treat_none",
                "calibration": "reference",
                "threshold": threshold,
                "net_benefit": 0.0,
            }
        )
        rows.append(
            {
                "model": "treat_all",
                "calibration": "reference",
                "threshold": threshold,
                "net_benefit": treat_all,
            }
        )
        for name, probabilities in prediction_sets.items():
            model_name, calibration = name.split("|", maxsplit=1)
            selected = probabilities >= threshold
            true_positive = int(((y == 1) & selected).sum())
            false_positive = int(((y == 0) & selected).sum())
            net_benefit = true_positive / n - false_positive / n * odds
            rows.append(
                {
                    "model": model_name,
                    "calibration": calibration,
                    "threshold": threshold,
                    "net_benefit": net_benefit,
                }
            )
    return pd.DataFrame(rows)


def value_with_ci(row: pd.Series, metric: str, digits: int = 3) -> str:
    return (
        f"{row[metric]:.{digits}f} "
        f"({row[f'{metric}_ci_low']:.{digits}f} to "
        f"{row[f'{metric}_ci_high']:.{digits}f})"
    )


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
    header = "| " + " | ".join(display.columns.astype(str)) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(row.astype(str).tolist()) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def write_report(
    output_dir: Path,
    absolute: pd.DataFrame,
    differences: pd.DataFrame,
    capacity: pd.DataFrame,
    subgroups: pd.DataFrame,
    index_sd: float,
    bootstrap: int,
) -> Path:
    original = absolute[absolute["calibration"].eq("none_original_transport")]
    m1 = original[original["model"].eq("M1_single_assessment")].iloc[0]
    m2 = original[original["model"].eq("M2_repeat_assessment")].iloc[0]
    recal_m2 = absolute[
        absolute["model"].eq("M2_repeat_assessment")
        & absolute["calibration"].eq("cv_intercept_slope")
    ].iloc[0]
    delta = differences.set_index("metric")
    auroc_delta = delta.loc["auroc"]
    auprc_delta = delta.loc["auprc"]
    brier_delta = delta.loc["brier"]
    cap20 = capacity[
        capacity["selected_fraction"].eq(0.20)
        & capacity["calibration"].eq("none_original_transport")
    ].set_index("model")
    supported = auroc_delta["ci_low"] > 0
    conclusion = (
        "The primary confirmatory comparison favored M2, and the 95% CI for "
        "the ELSA M2-minus-M1 AUROC difference excluded zero."
        if supported
        else "The primary confirmatory comparison did not clearly favor M2 because the 95% CI for the ELSA M2-minus-M1 AUROC difference included zero."
    )
    report = output_dir / "elsa_frozen_model_validation_2026-07-16.md"
    lines = [
        "# ELSA Confirmatory Validation of Frozen CHARLS Models",
        "",
        "Date: 2026-07-16",
        "",
        "## Locked analysis",
        "",
        "- Window: ELSA wave 6 -> wave 7 -> wave 8.",
        "- Primary contrast: frozen M2 repeat-assessment model versus frozen M1 single-assessment model.",
        "- Original transport performance is primary. ELSA-specific cross-validated intercept-plus-slope recalibration is secondary.",
        f"- Bootstrap replicates: {bootstrap:,}.",
        f"- ELSA wave 7 index-score SD: {index_sd:.4f}; decline threshold: {0.5*index_sd:.4f} points.",
        "",
        "## Confirmatory result",
        "",
        f"- Analysis sample: {int(m1['n']):,}; events: {int(m1['events']):,} ({m1['observed_percent']:.1f}%).",
        f"- M1 AUROC: {value_with_ci(m1, 'auroc')}.",
        f"- M2 AUROC: {value_with_ci(m2, 'auroc')}.",
        f"- M2 minus M1 AUROC: {auroc_delta['difference_candidate_minus_reference']:.4f} ({auroc_delta['ci_low']:.4f} to {auroc_delta['ci_high']:.4f}).",
        f"- M2 minus M1 AUPRC: {auprc_delta['difference_candidate_minus_reference']:.4f} ({auprc_delta['ci_low']:.4f} to {auprc_delta['ci_high']:.4f}).",
        f"- M2 minus M1 Brier score: {brier_delta['difference_candidate_minus_reference']:.4f} ({brier_delta['ci_low']:.4f} to {brier_delta['ci_high']:.4f}); negative favors M2.",
        f"- {conclusion}",
        "",
        "## Calibration",
        "",
        f"- Original M1: observed risk {m1['observed_percent']:.1f}%, mean predicted risk {m1['mean_predicted_percent']:.1f}%, Brier {m1['brier']:.3f}, calibration intercept {m1['calibration_intercept']:.3f}, slope {m1['calibration_slope']:.3f}.",
        f"- Original M2: observed risk {m2['observed_percent']:.1f}%, mean predicted risk {m2['mean_predicted_percent']:.1f}%, Brier {m2['brier']:.3f}, calibration intercept {m2['calibration_intercept']:.3f}, slope {m2['calibration_slope']:.3f}.",
        f"- Recalibrated M2: mean predicted risk {recal_m2['mean_predicted_percent']:.1f}%, Brier {recal_m2['brier']:.3f}, calibration intercept {recal_m2['calibration_intercept']:.3f}, slope {recal_m2['calibration_slope']:.3f}.",
        "- Recalibration was estimated within ELSA using fivefold cross-validation and does not replace the original-transport result.",
        "",
        "## Fixed 20% screening capacity",
        "",
        f"- M1: {int(cap20.loc['M1_single_assessment', 'true_positive_n']):,} events identified; sensitivity {100*cap20.loc['M1_single_assessment', 'sensitivity']:.1f}%; PPV {100*cap20.loc['M1_single_assessment', 'positive_predictive_value']:.1f}%.",
        f"- M2: {int(cap20.loc['M2_repeat_assessment', 'true_positive_n']):,} events identified; sensitivity {100*cap20.loc['M2_repeat_assessment', 'sensitivity']:.1f}%; PPV {100*cap20.loc['M2_repeat_assessment', 'positive_predictive_value']:.1f}%.",
        "",
        "## Prespecified descriptive subgroups",
        "",
        markdown_table(subgroups),
        "",
        "## Freeze protection",
        "",
        "- Frozen artifact hashes were verified before prediction.",
        "- No model coefficient, predictor, hyperparameter, preprocessing step, outcome threshold rule, or model-selection decision was altered using ELSA.",
        "- No person-level ELSA data or predictions were saved; all outputs are aggregate.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    root = elsa_audit.stata_dir(args.data_dir)
    harmonized = elsa_audit.load_harmonized(root)
    derived = elsa_audit.derive_elsa(harmonized)
    sample, y, index_sd = elsa_audit.analysis_sample(derived)
    models, artifact_verification, metadata = load_models(args.model_dir)

    feature_map = {
        "M1_single_assessment": metadata["models"]["M1_single_assessment"][
            "features_in_order"
        ],
        "M2_repeat_assessment": metadata["models"]["M2_repeat_assessment"][
            "features_in_order"
        ],
    }
    predictions = {
        model_name: models[model_name].predict_proba(sample[feature_map[model_name]])[
            :, 1
        ]
        for model_name in feature_map
    }
    recalibrated = {
        model_name: phase2.cross_validated_recalibration(
            y, probability, "intercept_slope", splits=5
        )
        for model_name, probability in predictions.items()
    }

    prediction_sets = []
    for model_name in feature_map:
        prediction_sets.append(
            (
                "ELSA confirmatory validation",
                model_name,
                "none_original_transport",
                y,
                predictions[model_name],
            )
        )
        prediction_sets.append(
            (
                "ELSA confirmatory validation",
                model_name,
                "cv_intercept_slope",
                y,
                recalibrated[model_name],
            )
        )
    absolute = pd.DataFrame(
        formal.absolute_performance_rows(prediction_sets, args.bootstrap)
    )
    differences = pd.DataFrame(
        formal.paired_difference_rows(
            "ELSA confirmatory validation",
            "none_original_transport",
            y,
            predictions["M1_single_assessment"],
            predictions["M2_repeat_assessment"],
            args.bootstrap,
            RANDOM_SEED + 500,
        )
    )
    subgroups = subgroup_rows(
        sample,
        y,
        predictions["M1_single_assessment"],
        predictions["M2_repeat_assessment"],
        args.bootstrap,
    )

    named_predictions = {
        "M1_single_assessment|none_original_transport": predictions[
            "M1_single_assessment"
        ],
        "M1_single_assessment|cv_intercept_slope": recalibrated[
            "M1_single_assessment"
        ],
        "M2_repeat_assessment|none_original_transport": predictions[
            "M2_repeat_assessment"
        ],
        "M2_repeat_assessment|cv_intercept_slope": recalibrated[
            "M2_repeat_assessment"
        ],
    }
    capacity = capacity_table(y, named_predictions)
    calibration = calibration_table(y, named_predictions)
    decision_curve = decision_curve_table(y, named_predictions)

    outputs = {
        "elsa_frozen_artifact_verification.csv": artifact_verification,
        "elsa_frozen_absolute_performance.csv": absolute,
        "elsa_frozen_paired_differences.csv": differences,
        "elsa_frozen_subgroups.csv": subgroups,
        "elsa_frozen_capacity.csv": capacity,
        "elsa_frozen_calibration_curve.csv": calibration,
        "elsa_frozen_decision_curve.csv": decision_curve,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False, encoding="utf-8-sig")

    report = write_report(
        args.output_dir,
        absolute,
        differences,
        capacity,
        subgroups,
        index_sd,
        args.bootstrap,
    )
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
