from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


def fmt_ci(row: pd.Series) -> str:
    return f"{row['difference_candidate_minus_reference']:.4f} ({row['ci_low']:.4f} to {row['ci_high']:.4f})"


def select_diff(table: pd.DataFrame, scenario: str, dataset: str, metric: str) -> pd.Series:
    rows = table[
        table["scenario"].eq(scenario)
        & table["dataset"].eq(dataset)
        & table["metric"].eq(metric)
    ]
    if rows.shape[0] != 1:
        raise RuntimeError(f"Expected one row for {scenario}, {dataset}, {metric}")
    return rows.iloc[0]


def select_m4(table: pd.DataFrame, dataset: str, candidate: str, metric: str) -> pd.Series:
    rows = table[
        table["dataset"].eq(dataset)
        & table["candidate_model"].eq(candidate)
        & table["metric"].eq(metric)
    ]
    if rows.shape[0] != 1:
        raise RuntimeError(f"Expected one row for {dataset}, {candidate}, {metric}")
    return rows.iloc[0]


def main() -> None:
    sensitivity = pd.read_csv(OUTPUT_DIR / "repeat_cognition_phase2_sensitivity_differences.csv")
    recalibration = pd.read_csv(OUTPUT_DIR / "repeat_cognition_phase2_hrs_recalibration.csv")
    m4 = pd.read_csv(OUTPUT_DIR / "repeat_cognition_phase3_m4_differences.csv")
    ipcw = pd.read_csv(OUTPUT_DIR / "repeat_cognition_phase4_ipcw_differences.csv")
    ipcw_diag = pd.read_csv(OUTPUT_DIR / "repeat_cognition_phase4_ipcw_diagnostics.csv")

    primary_charls = select_diff(
        sensitivity, "primary_20pt_age50_0.5sd", "CHARLS repeated 5-fold CV", "auroc"
    )
    primary_hrs = select_diff(
        sensitivity, "primary_20pt_age50_0.5sd", "HRS external validation", "auroc"
    )
    primary_hrs_auprc = select_diff(
        sensitivity, "primary_20pt_age50_0.5sd", "HRS external validation", "auprc"
    )
    primary_hrs_brier = select_diff(
        sensitivity, "primary_20pt_age50_0.5sd", "HRS external validation", "brier"
    )
    age65_hrs = select_diff(sensitivity, "age65_20pt_0.5sd", "HRS external validation", "auroc")
    sd1_hrs = select_diff(sensitivity, "decline_20pt_1sd_age50", "HRS external validation", "auroc")
    score25_hrs = select_diff(sensitivity, "score25_age50_0.5sd", "HRS external validation", "auroc")

    m2_original = recalibration[
        recalibration["model"].eq("M2_repeat_assessment")
        & recalibration["calibration"].eq("none_original_transport")
    ].iloc[0]
    m2_recal = recalibration[
        recalibration["model"].eq("M2_repeat_assessment")
        & recalibration["calibration"].eq("cv_intercept_slope")
    ].iloc[0]

    ipcw_hrs_auroc = ipcw[
        ipcw["dataset"].eq("HRS external validation") & ipcw["metric"].eq("auroc")
    ].iloc[0]
    m4_elastic_hrs = select_m4(m4, "HRS external validation", "M4_elastic_net", "auroc")
    m4_gb_hrs = select_m4(m4, "HRS external validation", "M4_gradient_boosting", "auroc")

    lines = [
        "# Current Results and Manuscript Positioning",
        "",
        "Date: 2026-07-16",
        "",
        "## Bottom Line",
        "",
        "The current data support a transparent prediction manuscript centered on repeat cognitive assessment. "
        "The strongest claim is not that a complex AI model predicts cognition, but that one prior objective memory assessment adds stable, externally transportable predictive information beyond a single current assessment.",
        "",
        "This is a stronger and cleaner story for a JAMA Network Open-style submission than the earlier CKM-centered topic, because the main contrast is clinically simple, methodologically defensible, externally validated in HRS, and robust across prespecified sensitivity analyses.",
        "",
        "## Core Results",
        "",
        f"- Primary CHARLS internal M2 vs M1 AUROC difference: {fmt_ci(primary_charls)}.",
        f"- Primary HRS external M2 vs M1 AUROC difference: {fmt_ci(primary_hrs)}.",
        f"- Primary HRS external AUPRC difference: {fmt_ci(primary_hrs_auprc)}.",
        f"- Primary HRS external Brier difference: {fmt_ci(primary_hrs_brier)}.",
        f"- HRS age >=65 AUROC difference: {fmt_ci(age65_hrs)}.",
        f"- HRS 1-SD decline AUROC difference: {fmt_ci(sd1_hrs)}.",
        f"- HRS 25-point cognition score AUROC difference: {fmt_ci(score25_hrs)}.",
        "",
        "## Calibration",
        "",
        f"- HRS original M2 observed risk: {m2_original['observed_percent']:.1f}%; mean predicted risk: {m2_original['mean_predicted_percent']:.1f}%; Brier: {m2_original['brier']:.3f}.",
        f"- HRS recalibrated M2 observed risk: {m2_recal['observed_percent']:.1f}%; mean predicted risk: {m2_recal['mean_predicted_percent']:.1f}%; Brier: {m2_recal['brier']:.3f}.",
        "- Interpretation: transport discrimination is good, but absolute risk requires local recalibration before clinical deployment.",
        "",
        "## Attrition/IPCW",
        "",
    ]
    for _, row in ipcw_diag.iterrows():
        lines.append(
            f"- {row['dataset']}: eligible n={int(row['eligible_n'])}, observed outcome n={int(row['observed_outcome_n'])} "
            f"({row['observed_outcome_percent']:.1f}%), effective weighted n={row['effective_sample_size']:.0f}."
        )
    lines.extend(
        [
            f"- IPCW HRS M2 vs M1 AUROC difference: {fmt_ci(ipcw_hrs_auroc)}.",
            "- Interpretation: the main finding is not explained by measured probability of outcome observation.",
            "",
            "## AI/Machine Learning",
            "",
            f"- HRS Elastic Net vs M2 AUROC difference: {fmt_ci(m4_elastic_hrs)}.",
            f"- HRS gradient boosting vs M2 AUROC difference: {fmt_ci(m4_gb_hrs)}.",
            "- Interpretation: machine learning does not improve external performance. It should be framed as a secondary robustness analysis, not as the central innovation.",
            "",
            "## Recommended Title Direction",
            "",
            "Incremental Value of Repeat Cognitive Assessment for Predicting Subsequent Memory Decline: Development in CHARLS and External Validation in HRS",
            "",
            "## Current Recommendation",
            "",
            "Proceed toward manuscript drafting with this structure:",
            "",
            "1. Main text: M1 vs M2, CHARLS development/internal validation, HRS external validation, calibration/recalibration.",
            "2. Main figure: sensitivity forest plot plus HRS calibration plot.",
            "3. Supplement: M3 clinical variables, M4 machine learning, IPCW, 25-point score, age >=65, 1-SD decline, HRS dementia/AD exclusion.",
            "4. Defer ELSA until the analysis code and manuscript tables are frozen; use it as confirmatory validation if access is ready.",
            "",
            "## Remaining Work",
            "",
            "- Polish figures in final journal style.",
            "- Draft STROBE/TRIPOD-aligned Methods and Results.",
            "- Decide whether to include ELSA before first submission.",
            "- Prepare a reproducibility appendix with variable mapping and code execution order.",
        ]
    )
    output = OUTPUT_DIR / "repeat_cognition_current_results_and_positioning_2026-07-16.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
