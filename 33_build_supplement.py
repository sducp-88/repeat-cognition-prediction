from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_PACKAGES = PROJECT_DIR / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import joblib
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
DEFAULT_SUPPLEMENT = PROJECT_DIR / "SUPPLEMENT_DRAFT_v0.1.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the aggregate 3-cohort supplement draft."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--supplement", type=Path, default=DEFAULT_SUPPLEMENT)
    return parser.parse_args()


def clean(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    header = "| " + " | ".join(clean(column) for column in display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def ci(value: float, low: float, high: float, digits: int = 3) -> str:
    return f"{value:.{digits}f} ({low:.{digits}f} to {high:.{digits}f})"


def model_label(value: str) -> str:
    return {
        "M1_single_assessment": "M1 single assessment",
        "M2_repeat_assessment": "M2 repeat assessment",
        "M3_expanded_clinical": "M3 expanded clinical",
        "M4_elastic_net": "M4 elastic net",
        "M4_gradient_boosting": "M4 gradient boosting",
    }.get(value, value)


def metric_label(value: str) -> str:
    return {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "brier": "Brier score",
    }.get(value, value)


def variable_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                "Previous/index/outcome waves",
                "W1 (2011) / W2 (2013) / W3 (2015)",
                "W11 (2012) / W12 (2014) / W13 (2016)",
                "W6 (2012/13) / W7 (2014/15) / W8 (2016/17)",
                "Approximately 2-year intervals",
            ),
            (
                "Person identifier",
                "ID",
                "hhidpn",
                "idauniq",
                "Used locally for linkage only",
            ),
            (
                "Age",
                "r1agey; r2agey",
                "r11agey_e; r12agey_e",
                "r6agey; r7agey",
                "Age >=50 at previous assessment; index age in model",
            ),
            (
                "Sex",
                "ragender",
                "ragender",
                "ragender",
                "Female indicator derived from harmonized sex",
            ),
            (
                "Education",
                "raeducl",
                "raeducl",
                "raeducl",
                "1 low, 2 middle, 3 high; cohort-harmonized categories",
            ),
            (
                "Immediate recall",
                "r1imrc; r2imrc; r3imrc",
                "r11imrc; r12imrc; r13imrc",
                "r6imrc; r7imrc; r8imrc",
                "Observed direct score, 0-10",
            ),
            (
                "Delayed recall",
                "r1dlrc; r2dlrc; r3dlrc",
                "r11dlrc; r12dlrc; r13dlrc",
                "r6dlrc; r7dlrc; r8dlrc",
                "Observed direct score, 0-10",
            ),
            (
                "Proxy/imputation exclusion",
                "No aligned proxy flag in Harmonized CHARLS D",
                "r11proxy-r13proxy and f-imrc/f-dlrc flags",
                "r6proxy-r8proxy and r6iwstat-r8iwstat",
                "HRS and ELSA required direct observed components",
            ),
            (
                "Primary memory score",
                "Immediate + delayed recall",
                "Immediate + delayed recall",
                "Immediate + delayed recall",
                "Complete sum, 0-20; no prorating",
            ),
            (
                "Primary outcome",
                "W3 minus W2 score",
                "W13 minus W12 score",
                "W8 minus W7 score",
                "Decline <= -0.5 x cohort-specific index-score SD",
            ),
        ],
        columns=["Construct", "CHARLS", "HRS", "ELSA", "Locked rule"],
    )


def cohort_flow(output_dir: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("CHARLS", "Age eligible at previous assessment", 13466, ""),
            ("CHARLS", "Previous memory complete", 10632, "79.0% of preceding stage"),
            ("CHARLS", "Index memory complete", 8359, "78.6%"),
            ("CHARLS", "Outcome memory complete: analysis sample", 7264, "86.9%"),
            ("CHARLS", "Primary decline events", 2886, "39.7% of analysis sample"),
            ("HRS", "Age eligible at previous assessment", 19866, ""),
            ("HRS", "Previous direct memory complete", 18129, "91.3%"),
            ("HRS", "Index direct memory complete", 15587, "86.0%"),
            ("HRS", "Outcome direct memory complete: analysis sample", 13118, "84.2%"),
            ("HRS", "Primary decline events", 4529, "34.5% of analysis sample"),
            ("ELSA", "Harmonized participants", 21679, ""),
            ("ELSA", "Age >=50 and index direct memory complete", 8214, "37.9% of all participants"),
            ("ELSA", "Also previous direct memory complete", 8152, "99.2%"),
            ("ELSA", "Also outcome direct memory complete: analysis sample", 6907, "84.7%"),
            ("ELSA", "Primary decline events", 2098, "30.4% of analysis sample"),
        ],
        columns=["Cohort", "Stage", "No.", "Retention or event rate"],
    )


def predictor_missingness(output_dir: Path) -> pd.DataFrame:
    common = pd.read_csv(
        output_dir / "repeat_cognition_supplement_predictor_missingness.csv"
    )
    common = common[
        common["variable"].isin(
            [
                "age",
                "female",
                "education_level",
                "index_cognition",
                "previous_cognition",
            ]
        )
    ].copy()
    elsa = pd.read_csv(output_dir / "elsa_feature_missingness.csv")
    elsa["cohort"] = "ELSA"
    elsa["n"] = 6907
    combined = pd.concat([common, elsa], ignore_index=True, sort=False)
    combined["Missing, No. (%)"] = combined.apply(
        lambda row: f"{int(row['missing_n'])} ({row['missing_percent']:.1f}%)", axis=1
    )
    pivot = combined.pivot(
        index="variable", columns="cohort", values="Missing, No. (%)"
    ).reset_index()
    pivot = pivot.rename(columns={"variable": "Predictor"})
    order = [
        "age",
        "female",
        "education_level",
        "index_cognition",
        "previous_cognition",
    ]
    pivot["_order"] = pivot["Predictor"].map({v: i for i, v in enumerate(order)})
    pivot = pivot.sort_values("_order").drop(columns="_order")
    return pivot[["Predictor", "CHARLS", "HRS", "ELSA"]]


def absolute_performance(output_dir: Path) -> pd.DataFrame:
    primary = pd.read_csv(output_dir / "repeat_cognition_formal_absolute_performance.csv")
    primary = primary[
        primary["calibration"].isin(["none_internal_cv", "none_original_transport"])
    ]
    elsa = pd.read_csv(output_dir / "elsa_frozen_absolute_performance.csv")
    elsa = elsa[elsa["calibration"].eq("none_original_transport")]
    data = pd.concat([primary, elsa], ignore_index=True)
    rows = []
    for row in data.itertuples():
        rows.append(
            {
                "Cohort/evaluation": row.dataset,
                "Model": model_label(row.model),
                "No. / events": f"{int(row.n):,} / {int(row.events):,}",
                "AUROC (95% CI)": ci(row.auroc, row.auroc_ci_low, row.auroc_ci_high),
                "AUPRC (95% CI)": ci(row.auprc, row.auprc_ci_low, row.auprc_ci_high),
                "Brier score (95% CI)": ci(row.brier, row.brier_ci_low, row.brier_ci_high),
                "Observed / predicted risk": f"{row.observed_percent:.1f}% / {row.mean_predicted_percent:.1f}%",
                "Calibration intercept (95% CI)": ci(
                    row.calibration_intercept,
                    row.calibration_intercept_ci_low,
                    row.calibration_intercept_ci_high,
                ),
                "Calibration slope (95% CI)": ci(
                    row.calibration_slope,
                    row.calibration_slope_ci_low,
                    row.calibration_slope_ci_high,
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_differences(output_dir: Path) -> pd.DataFrame:
    primary = pd.read_csv(output_dir / "repeat_cognition_formal_paired_differences.csv")
    primary = primary[
        primary["calibration"].isin(["none_internal_cv", "none_original_transport"])
    ]
    elsa = pd.read_csv(output_dir / "elsa_frozen_paired_differences.csv")
    data = pd.concat([primary, elsa], ignore_index=True)
    data["Estimate (95% CI)"] = data.apply(
        lambda row: ci(
            row["difference_candidate_minus_reference"],
            row["ci_low"],
            row["ci_high"],
            4,
        ),
        axis=1,
    )
    data["metric"] = data["metric"].map(metric_label)
    return data[
        ["dataset", "metric", "Estimate (95% CI)", "bootstrap_replicates_used"]
    ].rename(
        columns={
            "dataset": "Cohort/evaluation",
            "metric": "Metric",
            "bootstrap_replicates_used": "Bootstrap replicates",
        }
    )


def recalibration_table(output_dir: Path) -> pd.DataFrame:
    hrs = pd.read_csv(output_dir / "repeat_cognition_formal_absolute_performance.csv")
    hrs = hrs[
        hrs["dataset"].eq("HRS external validation")
        & hrs["model"].eq("M2_repeat_assessment")
    ]
    elsa = pd.read_csv(output_dir / "elsa_frozen_absolute_performance.csv")
    elsa = elsa[elsa["model"].eq("M2_repeat_assessment")]
    data = pd.concat([hrs, elsa], ignore_index=True)
    data["Analysis"] = data["calibration"].map(
        {
            "none_original_transport": "Original transport",
            "cv_intercept_slope": "5-fold CV intercept + slope recalibration",
        }
    )
    data["Observed / predicted risk"] = data.apply(
        lambda row: f"{row.observed_percent:.1f}% / {row.mean_predicted_percent:.1f}%",
        axis=1,
    )
    data["Brier (95% CI)"] = data.apply(
        lambda row: ci(row.brier, row.brier_ci_low, row.brier_ci_high), axis=1
    )
    data["Intercept (95% CI)"] = data.apply(
        lambda row: ci(
            row.calibration_intercept,
            row.calibration_intercept_ci_low,
            row.calibration_intercept_ci_high,
        ),
        axis=1,
    )
    data["Slope (95% CI)"] = data.apply(
        lambda row: ci(
            row.calibration_slope,
            row.calibration_slope_ci_low,
            row.calibration_slope_ci_high,
        ),
        axis=1,
    )
    return data[
        [
            "dataset",
            "Analysis",
            "Observed / predicted risk",
            "Brier (95% CI)",
            "Intercept (95% CI)",
            "Slope (95% CI)",
        ]
    ].rename(columns={"dataset": "Cohort"})


def capacity_table(output_dir: Path) -> pd.DataFrame:
    hrs = pd.read_csv(output_dir / "repeat_cognition_phase2_hrs_capacity_recalibrated.csv")
    hrs = hrs[hrs["calibration"].eq("none_original_transport")].copy()
    elsa = pd.read_csv(output_dir / "elsa_frozen_capacity.csv")
    elsa = elsa[elsa["calibration"].eq("none_original_transport")].copy()
    data = pd.concat([hrs, elsa], ignore_index=True, sort=False)
    data["Selected"] = data.apply(
        lambda row: f"{int(row.selected_n):,} ({100*row.selected_fraction:.0f}%)", axis=1
    )
    data["Sensitivity"] = data["sensitivity"].map(lambda value: f"{100*value:.1f}%")
    data["PPV"] = data["positive_predictive_value"].map(
        lambda value: f"{100*value:.1f}%"
    )
    data["true_positive_n"] = data["true_positive_n"].map(
        lambda value: f"{int(value):,}"
    )
    return data[
        ["dataset", "model", "Selected", "true_positive_n", "Sensitivity", "PPV"]
    ].rename(
        columns={
            "dataset": "Cohort",
            "model": "Model",
            "true_positive_n": "True-positive events",
        }
    ).assign(Model=lambda frame: frame["Model"].map(model_label))


def subgroup_table(output_dir: Path) -> pd.DataFrame:
    hrs = pd.read_csv(output_dir / "repeat_cognition_formal_hrs_subgroups.csv")
    hrs["Cohort"] = "HRS"
    elsa = pd.read_csv(output_dir / "elsa_frozen_subgroups.csv")
    elsa["Cohort"] = "ELSA"
    data = pd.concat([hrs, elsa], ignore_index=True, sort=False)
    data["Events, No. (%)"] = data.apply(
        lambda row: f"{int(row.events):,} ({row.event_percent:.1f}%)", axis=1
    )
    data["M1 / M2 AUROC"] = data.apply(
        lambda row: f"{row.m1_auroc:.3f} / {row.m2_auroc:.3f}", axis=1
    )
    data["AUROC difference (95% CI)"] = data.apply(
        lambda row: ci(
            row.delta_auroc, row.delta_auroc_ci_low, row.delta_auroc_ci_high, 4
        ),
        axis=1,
    )
    data["n"] = data["n"].map(lambda value: f"{int(value):,}")
    return data[
        [
            "Cohort",
            "subgroup",
            "level",
            "n",
            "Events, No. (%)",
            "M1 / M2 AUROC",
            "AUROC difference (95% CI)",
        ]
    ].rename(
        columns={"subgroup": "Subgroup", "level": "Level", "n": "No."}
    )


def sensitivity_table(output_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(output_dir / "repeat_cognition_phase2_sensitivity_differences.csv")
    data = data[data["metric"].eq("auroc")].copy()
    labels = {
        "age65_20pt_0.5sd": "Age >=65",
        "decline_20pt_1sd_age50": "1-SD decline",
        "hrs_no_dementia_ad_20pt_age50_0.5sd": "Exclude HRS dementia/Alzheimer disease",
        "score25_age50_0.5sd": "25-point cognition score",
    }
    data = data[data["scenario"].isin(labels)].copy()
    data = data[
        ~(
            data["scenario"].eq("hrs_no_dementia_ad_20pt_age50_0.5sd")
            & data["dataset"].str.startswith("CHARLS")
        )
    ]
    data["Analysis"] = data["scenario"].map(labels)
    data["AUROC difference (95% CI)"] = data.apply(
        lambda row: ci(
            row.difference_candidate_minus_reference, row.ci_low, row.ci_high, 4
        ),
        axis=1,
    )
    return data[["Analysis", "dataset", "AUROC difference (95% CI)"]].rename(
        columns={"dataset": "Cohort/evaluation"}
    )


def ipcw_table(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = pd.read_csv(output_dir / "repeat_cognition_phase4_ipcw_diagnostics.csv")
    diagnostics["Observed outcome"] = diagnostics.apply(
        lambda row: f"{int(row.observed_outcome_n):,}/{int(row.eligible_n):,} ({row.observed_outcome_percent:.1f}%)",
        axis=1,
    )
    diagnostics["Truncated weight range"] = diagnostics.apply(
        lambda row: f"{row.truncated_weight_min:.3f} to {row.truncated_weight_max:.3f}",
        axis=1,
    )
    diagnostics["Effective sample size"] = diagnostics["effective_sample_size"].map(
        lambda value: f"{value:,.0f}"
    )
    diagnostics_table = diagnostics[
        ["dataset", "Observed outcome", "Truncated weight range", "Effective sample size"]
    ].rename(columns={"dataset": "Cohort"})

    differences = pd.read_csv(output_dir / "repeat_cognition_phase4_ipcw_differences.csv")
    differences["Estimate (95% CI)"] = differences.apply(
        lambda row: ci(
            row.difference_candidate_minus_reference,
            row.ci_low,
            row.ci_high,
            4,
        ),
        axis=1,
    )
    differences["metric"] = differences["metric"].map(metric_label)
    differences_table = differences[
        ["dataset", "metric", "Estimate (95% CI)"]
    ].rename(columns={"dataset": "Cohort/evaluation", "metric": "Metric"})
    return diagnostics_table, differences_table


def ml_table(output_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(output_dir / "repeat_cognition_phase3_m4_differences.csv")
    data["Candidate"] = data["candidate_model"].map(model_label)
    data["Estimate (95% CI)"] = data.apply(
        lambda row: ci(
            row.difference_candidate_minus_reference,
            row.ci_low,
            row.ci_high,
            4,
        ),
        axis=1,
    )
    data["metric"] = data["metric"].map(metric_label)
    return data[["dataset", "Candidate", "metric", "Estimate (95% CI)"]].rename(
        columns={"dataset": "Cohort/evaluation", "metric": "Metric"}
    )


def raw_elsa_validation(output_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(output_dir / "elsa_raw_harmonized_validation.csv")
    return data.rename(
        columns={
            "wave": "Wave",
            "raw_variable": "Raw variable",
            "harmonized_variable": "Harmonized variable",
            "both_nonmissing_n": "Both nonmissing, No.",
            "exact_match_n": "Exact matches, No.",
            "mismatch_n": "Mismatches, No.",
        }
    )[
        [
            "Wave",
            "Raw variable",
            "Harmonized variable",
            "Both nonmissing, No.",
            "Exact matches, No.",
            "Mismatches, No.",
        ]
    ]


def frozen_model_specification() -> pd.DataFrame:
    artifacts = [
        (
            "M1 single assessment",
            PROJECT_DIR
            / "model_artifacts"
            / "m1_single_assessment_charls_frozen_2026-07-16.joblib",
        ),
        (
            "M2 repeat assessment",
            PROJECT_DIR
            / "model_artifacts"
            / "m2_repeat_assessment_charls_frozen_2026-07-16.joblib",
        ),
    ]
    rows: list[dict[str, str]] = []
    for label, path in artifacts:
        pipeline = joblib.load(path)
        preprocess = pipeline.named_steps["preprocess"]
        estimator = pipeline.named_steps["model"]
        feature_names = preprocess.get_feature_names_out()
        numeric = preprocess.named_transformers_["numeric"]
        numeric_columns = list(preprocess.transformers_[0][2])
        numeric_stats = {
            name: {
                "mean": numeric.named_steps["scaler"].mean_[index],
                "scale": numeric.named_steps["scaler"].scale_[index],
                "imputer": numeric.named_steps["imputer"].statistics_[index],
            }
            for index, name in enumerate(numeric_columns)
        }
        rows.append(
            {
                "Model": label,
                "Term": "Intercept",
                "Coefficient": f"{float(estimator.intercept_[0]):.8f}",
                "Training mean": "",
                "Training SD": "",
                "Imputation value": "",
                "Coding": "Logistic-regression intercept",
            }
        )
        for feature_name, coefficient in zip(feature_names, estimator.coef_[0]):
            source_name = feature_name.split("__", 1)[1]
            if feature_name.startswith("numeric__"):
                stats = numeric_stats[source_name]
                term = source_name.replace("_", " ")
                coding = "Standardized as (value - training mean) / training SD"
                mean = f"{stats['mean']:.8f}"
                scale = f"{stats['scale']:.8f}"
                imputer = f"{stats['imputer']:.8f}"
            else:
                category = source_name.rsplit("_", 1)[1]
                term = f"education level {category}"
                coding = "One-hot indicator for category; missing values imputed to level 1"
                mean = ""
                scale = ""
                imputer = "Most frequent category (1)"
            rows.append(
                {
                    "Model": label,
                    "Term": term,
                    "Coefficient": f"{float(coefficient):.8f}",
                    "Training mean": mean,
                    "Training SD": scale,
                    "Imputation value": imputer,
                    "Coding": coding,
                }
            )
    return pd.DataFrame(rows)


def artifact_verification(output_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(output_dir / "elsa_frozen_artifact_verification.csv")
    data["Model"] = data["model"].map(model_label)
    data["Hash verified"] = data["hash_verified"].map(
        lambda value: "Yes" if bool(value) else "No"
    )
    data["Artifact"] = data["artifact"].map(lambda value: Path(value).name)
    return data[
        [
            "Model",
            "Artifact",
            "freeze_date",
            "development_cohort",
            "expected_sha256",
            "Hash verified",
        ]
    ].rename(
        columns={
            "freeze_date": "Freeze date",
            "development_cohort": "Development cohort",
            "expected_sha256": "Expected SHA-256",
        }
    )


def build_supplement(output_dir: Path) -> str:
    ipcw_diagnostics, ipcw_differences = ipcw_table(output_dir)
    sections = [
        "# Supplement",
        "",
        "**Article title:** Repeat Cognitive Assessment for Predicting Memory Decline Across 3 National Aging Cohorts",
        "",
        "## Supplement Contents",
        "",
        "- eMethods 1. Data Sources, Study Design, and Governance",
        "- eMethods 2. Participants and Cognitive Measures",
        "- eMethods 3. Predictors, Model Development, and Freeze Protection",
        "- eMethods 4. Performance, Recalibration, and Decision-Curve Analysis",
        "- eMethods 5. Missing Outcomes, Sensitivity Analyses, and Machine Learning",
        "- eMethods 6. Model Use, Input Quality, and Implementation Considerations",
        "- eTable 1. Cross-Cohort Variable Mapping",
        "- eTable 2. Cohort Construction and Outcome Events",
        "- eTable 3. Missingness in M1 and M2 Predictors",
        "- eTable 4. Full Primary Model Performance",
        "- eTable 5. Paired M2 vs M1 Performance Differences",
        "- eTable 6. External-Cohort M2 Recalibration",
        "- eTable 7. Fixed-Capacity Performance",
        "- eTable 8. External-Cohort Subgroup Performance",
        "- eTable 9. Prespecified Sensitivity Analyses",
        "- eTable 10. IPCW Diagnostics and Performance",
        "- eTable 11. Machine-Learning Comparisons",
        "- eTable 12. Raw vs Harmonized ELSA Verification",
        "- eTable 13. Frozen Model Coefficients and Preprocessing Parameters",
        "- eTable 14. Frozen Artifact Verification Before ELSA Analysis",
        "",
        "# eMethods",
        "",
        "## eMethods 1. Data Sources, Study Design, and Governance",
        "",
        "This prediction model study used the China Health and Retirement Longitudinal Study (CHARLS), the US Health and Retirement Study (HRS), and the English Longitudinal Study of Ageing (ELSA).<sup>1-8</sup> CHARLS 2011, 2013, and 2015 formed the development sequence. HRS 2012, 2014, and 2016 formed the first cross-national external evaluation sequence. ELSA waves 6, 7, and 8 formed a locked confirmation sequence after the CHARLS models, preprocessing, primary outcome rule, and reporting code had been frozen and hashed. HRS had contributed to earlier exploratory work and was therefore not described as untouched confirmation. ELSA was not used to choose predictors, tune hyperparameters, revise score definitions, or refit model coefficients. Reporting followed TRIPOD+AI and STROBE guidance.<sup>9,10</sup>",
        "",
        "All analyses were conducted within the local data environment under the applicable cohort data-use agreements. Person identifiers were used only for local linkage. No person-level cohort records or individual predictions were written to the project output directory. Saved outputs comprised aggregate tables, figures, code, fitted model objects without source records, software metadata, and cryptographic hashes. CHARLS, HRS, and ELSA received ethics approval from their responsible review committees and obtained participant informed consent. Under the institutional policy of the Second Qilu Hospital of Shandong University, this secondary analysis did not require additional ethics committee review or approval; accordingly, no separate written exemption determination was issued. Patients and members of the public were not involved in the design, conduct, reporting, interpretation, or dissemination planning of this secondary analysis.",
        "",
        "## eMethods 2. Participants and Cognitive Measures",
        "",
        "Participants were eligible at age 50 years or older at the previous assessment and required a complete index memory score. The primary performance sample additionally required complete immediate and delayed word recall at the previous, index, and outcome assessments. Immediate recall and delayed recall each ranged from 0 to 10; both components were required and were summed without prorating to form the 20-point memory score. HRS proxy interviews and RAND-imputed immediate or delayed recall components were excluded at all 3 assessments. ELSA interview status, proxy status, and raw recall fields were cross-checked against Harmonized ELSA before the confirmatory analysis. Negative raw ELSA recall codes were treated as missing.",
        "",
        "For each cohort, the SD of the index score was calculated among age-eligible participants with an observed index score without using future outcome values. Primary decline was defined as an outcome score minus index score less than or equal to negative 0.5 times this cohort-specific SD. This standardized rule preserved the same construct while allowing different score distributions across languages and cohorts. A 1-SD decline, age 65 years or older, and a 25-point score adding serial 7 subtraction were prespecified sensitivity analyses in CHARLS and HRS.",
        "",
        "## eMethods 3. Predictors, Model Development, and Freeze Protection",
        "",
        "M1 included age at the index assessment, sex, harmonized 3-level education, and the index 20-point memory score. M2 additionally included the previous 20-point memory score. M3 additionally included prespecified health, function, depressive-symptom, and lifestyle predictors. Continuous predictors were median-imputed and standardized; categorical education was most-frequent-imputed and one-hot encoded. These transformations were estimated within CHARLS training folds and stored in the fitted pipeline.",
        "",
        "M1 through M3 used logistic regression. CHARLS internal performance used repeated stratified 5-fold cross-validation. Final M1 and M2 pipelines were fitted once to the full eligible CHARLS development sample and serialized on July 16, 2026. Reloaded artifacts reproduced HRS predictions exactly. Before ELSA prediction, stored SHA-256 hashes were verified. The ELSA analysis loaded and applied these pipelines unchanged. ELSA education codes outside the prespecified 1-to-3 harmonized categories were treated as missing and handled by the already-fitted categorical imputer.",
        "",
        "No a priori target sample size was imposed because all eligible participants in each prespecified wave sequence were included. The primary samples contained 7264 participants and 2886 events in CHARLS, 13 118 participants and 4529 events in HRS, and 6907 participants and 2098 events in ELSA, providing substantial event counts for the parsimonious prespecified models and their external evaluation.",
        "",
        "## eMethods 4. Performance, Recalibration, and Decision-Curve Analysis",
        "",
        "The primary performance measure was the paired M2 minus M1 difference in area under the receiver operating characteristic curve (AUROC). Additional measures were area under the precision-recall curve (AUPRC), Brier score, calibration intercept and slope, observed and mean predicted risk, and sensitivity and positive predictive value when the highest-risk 10%, 20%, or 30% of participants were selected. Participant-level paired bootstrap resampling with 1000 replicates provided percentile 95% CIs.",
        "",
        "Original CHARLS-to-HRS and CHARLS-to-ELSA transport performance was reported before model updating. Intercept-plus-slope recalibration was then estimated separately within HRS and ELSA using stratified 5-fold cross-validation. Each recalibrated probability was generated for a participant held outside the fold used to estimate recalibration parameters. Recalibrated probabilities were used for decision curves involving absolute risk thresholds; original transport remained the primary external-performance analysis. Net benefit was calculated across prespecified thresholds and compared with treat-all and treat-none strategies.",
        "",
        "## eMethods 5. Missing Outcomes, Sensitivity Analyses, and Machine Learning",
        "",
        "The primary analysis evaluated participants with observed 3-wave cognition. In CHARLS and HRS, inverse-probability-of-cognitive-outcome-observation weights were estimated among index-eligible participants. Stabilized weights were truncated at the cohort-specific 1st and 99th percentiles and applied to performance estimation without changing model training. Subgroup estimates by sex, age, education, and residence were descriptive and were not used for model selection. ELSA subgroup reporting included sex, age, and education; residence was not required for the frozen M1/M2 comparison. Race and ethnicity were not analyzed because comparable harmonized categories were unavailable across the Chinese, US, and English cohort contexts; country was not interpreted as a proxy for race or ethnicity.<sup>11-14</sup>",
        "",
        "Elastic-net logistic regression and histogram gradient boosting used the M3 candidate information. Hyperparameters were selected only within nested CHARLS training folds. HRS was used solely for external comparison of the selected algorithms. Machine learning was secondary because it did not improve external performance over the transparent M2 model.",
        "",
        "## eMethods 6. Model Use, Input Quality, and Implementation Considerations",
        "",
        "M2 produces a probability of memory decline over the next assessment interval from age, sex, education, and complete direct 20-point memory scores at the previous and index assessments. The model is not yet intended for unsupervised clinical use. At the point of future implementation, unavailable previous or current memory scores should not be silently imputed; the assessment should be repeated or the model should not be applied. Recall testing should follow a standardized direct-interview protocol, and locally different education categories require prespecified mapping and validation.",
        "",
        "A validated software implementation can calculate predictions without specialized machine-learning expertise, but users need training in standardized cognitive testing and interpretation of probabilistic risk. Because original transport overpredicted absolute risk in both external cohorts, any new setting should evaluate calibration and perform local recalibration before using absolute-risk thresholds. Prospective evaluation, assessment of downstream clinical consequences, and further evaluation in underrepresented sociodemographic groups are required before clinical implementation.",
        "",
        "# eTables",
        "",
        "## eTable 1. Cross-Cohort Variable Mapping",
        "",
        markdown_table(variable_mapping()),
        "",
        "## eTable 2. Cohort Construction and Outcome Events",
        "",
        markdown_table(cohort_flow(output_dir)),
        "",
        "## eTable 3. Missingness in M1 and M2 Predictors",
        "",
        markdown_table(predictor_missingness(output_dir)),
        "",
        "## eTable 4. Full Primary Model Performance",
        "",
        markdown_table(absolute_performance(output_dir)),
        "",
        "## eTable 5. Paired M2 vs M1 Performance Differences",
        "",
        markdown_table(paired_differences(output_dir)),
        "",
        "Negative Brier-score differences favor M2. All differences used paired participant-level bootstrap resampling.",
        "",
        "## eTable 6. External-Cohort M2 Recalibration",
        "",
        markdown_table(recalibration_table(output_dir)),
        "",
        "Recalibration estimates were evaluated out of fold and do not replace original transport performance.",
        "",
        "## eTable 7. Fixed-Capacity Performance",
        "",
        markdown_table(capacity_table(output_dir)),
        "",
        "Selection was based on original-transport ranking. PPV indicates positive predictive value.",
        "",
        "## eTable 8. External-Cohort Subgroup Performance",
        "",
        markdown_table(subgroup_table(output_dir)),
        "",
        "Subgroup results are descriptive. The ELSA education-missing level is shown for transparency but was not a prespecified substantive education stratum.",
        "",
        "## eTable 9. Prespecified Sensitivity Analyses",
        "",
        markdown_table(sensitivity_table(output_dir)),
        "",
        "## eTable 10. IPCW Diagnostics and Performance",
        "",
        "### eTable 10A. Weight Diagnostics",
        "",
        markdown_table(ipcw_diagnostics),
        "",
        "### eTable 10B. IPCW M2 vs M1 Differences",
        "",
        markdown_table(ipcw_differences),
        "",
        "## eTable 11. Machine-Learning Comparisons",
        "",
        markdown_table(ml_table(output_dir)),
        "",
        "M4 differences use M2 as the reference. Negative Brier-score differences favor the candidate model.",
        "",
        "## eTable 12. Raw vs Harmonized ELSA Verification",
        "",
        markdown_table(raw_elsa_validation(output_dir)),
        "",
        "Negative raw recall codes were treated as missing before comparison. No mismatches were observed among jointly nonmissing values.",
        "",
        "## eTable 13. Frozen Model Coefficients and Preprocessing Parameters",
        "",
        markdown_table(frozen_model_specification()),
        "",
        "Coefficients are on the logistic-regression log-odds scale. Continuous-variable training means, SDs, and median imputation values were estimated in the full CHARLS development sample and stored in each fitted pipeline. Education used most-frequent imputation followed by one-hot encoding.",
        "",
        "## eTable 14. Frozen Artifact Verification Before ELSA Analysis",
        "",
        markdown_table(artifact_verification(output_dir)),
        "",
        "Expected and observed SHA-256 hashes were identical for both artifacts before any ELSA prediction was generated.",
        "",
        "# Supplement References",
        "",
        "1. Zhao Y, Hu Y, Smith JP, Strauss J, Yang G. Cohort profile: the China Health and Retirement Longitudinal Study (CHARLS). Int J Epidemiol. 2014;43(1):61-68. doi:10.1093/ije/dys203",
        "",
        "2. Sonnega A, Faul JD, Ofstedal MB, Langa KM, Phillips JWR, Weir DR. Cohort profile: the Health and Retirement Study (HRS). Int J Epidemiol. 2014;43(2):576-585. doi:10.1093/ije/dyu067",
        "",
        "3. Steptoe A, Breeze E, Banks J, Nazroo J. Cohort profile: the English Longitudinal Study of Ageing. Int J Epidemiol. 2013;42(6):1640-1648. doi:10.1093/ije/dys168",
        "",
        "4. Gateway to Global Aging Data. Harmonized CHARLS, Version D. University of Southern California; 2021. Accessed July 17, 2026. https://g2aging.org/",
        "",
        "5. Health and Retirement Study. RAND HRS Longitudinal File 2022 (V1) public use dataset. University of Michigan; May 2025.",
        "",
        "6. Wang Y, Cole A, Green H, Wilkens J, Phillips D, Lee J. Harmonized HRS, Version D. Gateway to Global Aging Data; 2023. doi:10.25549/4smz-hp46",
        "",
        "7. Markot M, Xie C, Cole A, et al. Gateway Harmonized ELSA, Version H. Gateway to Global Aging Data; 2025. doi:10.25553/h52b-8869",
        "",
        "8. Banks J, Cribb J, Coughlin K, et al. English Longitudinal Study of Ageing: Waves 0-11, 1998-2024. 47th ed. UK Data Service; 2025. doi:10.5255/UKDA-SN-5050-34",
        "",
        "9. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378",
        "",
        "10. von Elm E, Altman DG, Egger M, et al. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies. Lancet. 2007;370(9596):1453-1457. doi:10.1016/S0140-6736(07)61602-X",
        "",
        "11. Glymour MM, Manly JJ. Lifecourse social conditions and racial and ethnic patterns of cognitive aging. Neuropsychol Rev. 2008;18(3):223-254. doi:10.1007/s11065-008-9064-z",
        "",
        "12. Avila JF, Arce Renteria M, Witkiewitz K, Verney SP, Vonk JMJ, Manly JJ. Measurement invariance of neuropsychological measures of cognitive aging across race/ethnicity by sex/gender groups. Neuropsychology. 2020;34(1):3-14. doi:10.1037/neu0000584",
        "",
        "13. Merkley TL, Esopenko C, Zizak VS, et al. Challenges and opportunities for harmonization of cross-cultural neuropsychological data. Neuropsychology. 2023;37(3):237-246. doi:10.1037/neu0000818",
        "",
        "14. Iveniuk J, Zhong S, Wilder J, et al. Race/ethnicity and the measurement of cognition in the National Social Life, Health, and Aging Project: recommendations for robustness. J Gerontol B Psychol Sci Soc Sci. 2025;80(suppl 1):S55-S65. doi:10.1093/geronb/gbae043",
    ]
    return "\n".join(sections) + "\n"


def main() -> None:
    args = parse_args()
    text = build_supplement(args.output_dir)
    args.supplement.write_text(text, encoding="utf-8")
    print(f"Wrote supplement draft to {args.supplement}")


if __name__ == "__main__":
    main()
