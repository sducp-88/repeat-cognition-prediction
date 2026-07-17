from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_PACKAGES = PROJECT_DIR / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = PROJECT_DIR / "data_link"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
WAVES = (6, 7, 8)
HARMONIZED_COLUMNS = [
    "idauniq",
    "ragender",
    "raeducl",
    "r6iwstat",
    "r7iwstat",
    "r8iwstat",
    "r6proxy",
    "r7proxy",
    "r8proxy",
    "r6agey",
    "r7agey",
    "r8agey",
    "r6imrc",
    "r7imrc",
    "r8imrc",
    "r6dlrc",
    "r7dlrc",
    "r8dlrc",
]
RAW_WAVE_FILES = {
    6: "wave_6_elsa_data_eul.dta",
    7: "wave_7_elsa_data_eul.dta",
    8: "wave_8_elsa_data_eul_v2.dta",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit ELSA wave 6-8 data and build the locked validation sample."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def stata_dir(data_dir: Path) -> Path:
    candidates = [
        data_dir / "UKDA-5050-stata" / "stata" / "stata13_se",
        data_dir / "stata" / "stata13_se",
        data_dir,
    ]
    for candidate in candidates:
        if (candidate / "gh_elsa_h.dta").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find gh_elsa_h.dta under the supplied ELSA data directory."
    )


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def strict_component(data: pd.DataFrame, wave: int, component: str) -> pd.Series:
    value = numeric(data[f"r{wave}{component}"]).where(lambda x: x.between(0, 10))
    interviewed = numeric(data[f"r{wave}iwstat"]).eq(1)
    direct = numeric(data[f"r{wave}proxy"]).eq(0)
    return value.where(interviewed & direct)


def complete_sum(first: pd.Series, second: pd.Series) -> pd.Series:
    values = pd.concat([first, second], axis=1)
    return values.sum(axis=1).where(values.notna().all(axis=1))


def derive_elsa(harmonized: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=harmonized.index)
    result["idauniq"] = harmonized["idauniq"]
    result["baseline_age"] = numeric(harmonized["r6agey"])
    result["age"] = numeric(harmonized["r7agey"])
    gender = numeric(harmonized["ragender"])
    result["female"] = gender.eq(2).where(gender.isin([1, 2])).astype(float)
    education = numeric(harmonized["raeducl"])
    result["education_level"] = education.where(education.isin([1, 2, 3]))

    score_names = {
        6: "previous_cognition",
        7: "index_cognition",
        8: "outcome_cognition",
    }
    for wave, name in score_names.items():
        result[name] = complete_sum(
            strict_component(harmonized, wave, "imrc"),
            strict_component(harmonized, wave, "dlrc"),
        )
    return result


def analysis_sample(
    derived: pd.DataFrame, minimum_age: int = 50, decline_sd: float = 0.5
) -> tuple[pd.DataFrame, np.ndarray, float]:
    eligible = derived["baseline_age"].ge(minimum_age) & derived[
        "index_cognition"
    ].notna()
    index_sd = float(derived.loc[eligible, "index_cognition"].std(ddof=1))
    complete = (
        eligible
        & derived["previous_cognition"].notna()
        & derived["outcome_cognition"].notna()
    )
    sample = derived.loc[complete].copy()
    sample["previous_change"] = (
        sample["index_cognition"] - sample["previous_cognition"]
    )
    outcome = (
        sample["outcome_cognition"] - sample["index_cognition"]
        <= -decline_sd * index_sd
    ).astype(int)
    return sample, outcome.to_numpy(), index_sd


def load_harmonized(root: Path) -> pd.DataFrame:
    path = root / "gh_elsa_h.dta"
    data = pd.read_stata(path, columns=HARMONIZED_COLUMNS, convert_categoricals=False)
    if data["idauniq"].duplicated().any():
        raise RuntimeError("Harmonized ELSA contains duplicate idauniq values.")
    return data


def raw_harmonized_validation(
    root: Path, harmonized: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for wave in WAVES:
        raw_columns = [
            "idauniq",
            f"w{wave}indout",
            "askpx",
            "cflisen",
            "cflisd",
        ]
        raw = pd.read_stata(
            root / RAW_WAVE_FILES[wave],
            columns=raw_columns,
            convert_categoricals=False,
        )
        if raw["idauniq"].duplicated().any():
            raise RuntimeError(f"Wave {wave} raw data contain duplicate idauniq values.")
        comparison = raw.merge(
            harmonized[
                ["idauniq", f"r{wave}proxy", f"r{wave}imrc", f"r{wave}dlrc"]
            ],
            on="idauniq",
            how="left",
            indicator=True,
            validate="one_to_one",
        )
        pairs = [
            ("askpx", f"r{wave}proxy", None),
            ("cflisen", f"r{wave}imrc", (0, 10)),
            ("cflisd", f"r{wave}dlrc", (0, 10)),
        ]
        for raw_name, harmonized_name, valid_range in pairs:
            raw_value = numeric(comparison[raw_name])
            harmonized_value = numeric(comparison[harmonized_name])
            if valid_range is not None:
                raw_value = raw_value.where(raw_value.between(*valid_range))
                harmonized_value = harmonized_value.where(
                    harmonized_value.between(*valid_range)
                )
            both = raw_value.notna() & harmonized_value.notna()
            mismatch = both & raw_value.ne(harmonized_value)
            rows.append(
                {
                    "wave": wave,
                    "raw_file": RAW_WAVE_FILES[wave],
                    "raw_rows": int(len(raw)),
                    "raw_ids_not_in_harmonized": int(
                        comparison["_merge"].eq("left_only").sum()
                    ),
                    "raw_variable": raw_name,
                    "harmonized_variable": harmonized_name,
                    "both_nonmissing_n": int(both.sum()),
                    "exact_match_n": int((both & ~mismatch).sum()),
                    "mismatch_n": int(mismatch.sum()),
                }
            )
    return pd.DataFrame(rows)


def cohort_flow(derived: pd.DataFrame, outcome: np.ndarray, index_sd: float) -> pd.DataFrame:
    age_eligible_index = derived["baseline_age"].ge(50) & derived[
        "index_cognition"
    ].notna()
    with_previous = age_eligible_index & derived["previous_cognition"].notna()
    final = with_previous & derived["outcome_cognition"].notna()
    stages = [
        ("Harmonized ELSA participants", pd.Series(True, index=derived.index)),
        ("Direct complete wave 6 memory", derived["previous_cognition"].notna()),
        ("Age >=50 at wave 6 and direct complete wave 7 memory", age_eligible_index),
        ("Also direct complete wave 6 memory", with_previous),
        ("Also direct complete wave 8 memory: final validation sample", final),
    ]
    rows = [
        {"stage": stage, "n": int(mask.sum()), "percent_of_total": 100 * mask.mean()}
        for stage, mask in stages
    ]
    rows.append(
        {
            "stage": "Primary decline events in final validation sample",
            "n": int(outcome.sum()),
            "percent_of_total": 100 * outcome.mean(),
        }
    )
    rows.append(
        {
            "stage": "Index-score SD used for outcome threshold",
            "n": np.nan,
            "percent_of_total": index_sd,
        }
    )
    rows.append(
        {
            "stage": "Decline threshold in memory-score points",
            "n": np.nan,
            "percent_of_total": 0.5 * index_sd,
        }
    )
    return pd.DataFrame(rows)


def file_inventory(root: Path) -> pd.DataFrame:
    package_root = root.parents[2]
    rows = []
    for path in sorted(package_root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(package_root)),
                    "extension": path.suffix.lower(),
                    "bytes": path.stat().st_size,
                    "modified": path.stat().st_mtime,
                }
            )
    return pd.DataFrame(rows)


def mapping_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Identifier", "idauniq", "Link wave 6, 7, and 8 records"),
            ("Interview status", "r6iwstat/r7iwstat/r8iwstat", "Require 1: respondent alive"),
            ("Proxy status", "r6proxy/r7proxy/r8proxy", "Require 0: direct interview"),
            ("Immediate recall", "r6imrc/r7imrc/r8imrc", "Require observed value 0-10"),
            ("Delayed recall", "r6dlrc/r7dlrc/r8dlrc", "Require observed value 0-10"),
            ("Memory score", "immediate + delayed recall", "Complete sum, range 0-20"),
            ("Age", "r6agey and r7agey", "Wave 6 eligibility; wave 7 model predictor"),
            ("Sex", "ragender", "1 man, 2 woman; derive female indicator"),
            ("Education", "raeducl", "1 low, 2 middle, 3 high; other codes missing"),
            ("Primary outcome", "wave 8 score - wave 7 score", "Decline <= -0.5 wave 7 SD"),
        ],
        columns=["construct", "elsa_variables", "locked_rule"],
    )


def three_cohort_table(
    output_dir: Path, sample: pd.DataFrame, outcome: np.ndarray
) -> pd.DataFrame:
    source = output_dir / "repeat_cognition_table1_cohort_characteristics.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing existing CHARLS/HRS Table 1 source: {source}")
    table = pd.read_csv(source, dtype=str).fillna("")
    education = sample["education_level"].dropna()
    previous_change = sample["index_cognition"] - sample["previous_cognition"]
    elsa_values = {
        "Participants, No.": f"{len(sample)}",
        "Memory decline, No. (%)": f"{int(outcome.sum())} ({100*outcome.mean():.1f}%)",
        "Age at index assessment, mean (SD), y": f"{sample['age'].mean():.1f} ({sample['age'].std(ddof=1):.1f})",
        "Female sex, No. (%)": f"{int(sample['female'].sum())} ({100*sample['female'].mean():.1f}%)",
        "Education: low, No. (%)": f"{int(education.eq(1).sum())} ({100*education.eq(1).mean():.1f}%)",
        "Education: middle, No. (%)": f"{int(education.eq(2).sum())} ({100*education.eq(2).mean():.1f}%)",
        "Education: high, No. (%)": f"{int(education.eq(3).sum())} ({100*education.eq(3).mean():.1f}%)",
        "Previous memory score, mean (SD)": f"{sample['previous_cognition'].mean():.1f} ({sample['previous_cognition'].std(ddof=1):.1f})",
        "Index memory score, mean (SD)": f"{sample['index_cognition'].mean():.1f} ({sample['index_cognition'].std(ddof=1):.1f})",
        "Previous-to-index change, mean (SD)": f"{previous_change.mean():.1f} ({previous_change.std(ddof=1):.1f})",
        "Outcome memory score, mean (SD)": f"{sample['outcome_cognition'].mean():.1f} ({sample['outcome_cognition'].std(ddof=1):.1f})",
    }
    table["ELSA"] = table["characteristic"].map(elsa_values).fillna("")
    return table


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
    root: Path,
    validation: pd.DataFrame,
    flow: pd.DataFrame,
    missingness: pd.DataFrame,
    sample: pd.DataFrame,
    outcome: np.ndarray,
    index_sd: float,
) -> Path:
    mismatch_n = int(validation["mismatch_n"].sum())
    report = output_dir / "elsa_data_audit_2026-07-16.md"
    lines = [
        "# ELSA Data Audit and Locked Sample",
        "",
        "Date: 2026-07-16",
        "",
        "## Data package",
        "",
        f"- Stata directory: `{root}`",
        "- Required files found: Harmonized ELSA H and raw ELSA wave 6, 7, and 8 interviewer files.",
        "- The analysis uses wave 6 -> wave 7 -> wave 8, corresponding approximately to 2012/13 -> 2014/15 -> 2016/17.",
        "",
        "## Cross-source verification",
        "",
        f"- Total raw-versus-harmonized mismatches across proxy, immediate recall, and delayed recall fields: {mismatch_n}.",
        "- Negative raw recall codes were treated as missing before comparison.",
        "",
        "## Locked validation sample",
        "",
        f"- Final N: {len(sample):,}",
        f"- Primary events: {int(outcome.sum()):,} ({100*outcome.mean():.1f}%)",
        f"- Wave 7 index-score SD: {index_sd:.4f}",
        f"- Decline threshold: {0.5*index_sd:.4f} points",
        "- Scores require direct interviews and complete immediate plus delayed recall at all three waves.",
        "- Age eligibility is age 50 years or older at wave 6, matching the frozen analysis convention.",
        "",
        "## Privacy and freeze protection",
        "",
        "- No person-level ELSA data or predictions were written to the outputs directory.",
        "- ELSA was not used to select predictors, alter score definitions, tune hyperparameters, or refit CHARLS coefficients.",
        "- Only aggregate audit tables were saved.",
        "",
        "## Cohort flow",
        "",
        markdown_table(flow),
        "",
        "## Predictor missingness in the final sample",
        "",
        markdown_table(missingness),
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = stata_dir(args.data_dir)
    required = [root / "gh_elsa_h.dta"] + [
        root / RAW_WAVE_FILES[wave] for wave in WAVES
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required ELSA files: " + ", ".join(missing))

    harmonized = load_harmonized(root)
    derived = derive_elsa(harmonized)
    sample, outcome, index_sd = analysis_sample(derived)
    validation = raw_harmonized_validation(root, harmonized)
    if validation["mismatch_n"].sum() != 0:
        raise RuntimeError("Raw and harmonized ELSA core variables do not agree exactly.")

    flow = cohort_flow(derived, outcome, index_sd)
    feature_columns = [
        "age",
        "female",
        "education_level",
        "index_cognition",
        "previous_cognition",
    ]
    missingness = pd.DataFrame(
        [
            {
                "variable": column,
                "missing_n": int(sample[column].isna().sum()),
                "missing_percent": 100 * float(sample[column].isna().mean()),
            }
            for column in feature_columns
        ]
    )

    inventory = file_inventory(root)
    inventory.to_csv(
        args.output_dir / "elsa_file_inventory.csv", index=False, encoding="utf-8-sig"
    )
    mapping_table().to_csv(
        args.output_dir / "elsa_variable_mapping.csv", index=False, encoding="utf-8-sig"
    )
    three_cohort_table(args.output_dir, sample, outcome).to_csv(
        args.output_dir / "repeat_cognition_table1_three_cohort_characteristics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation.to_csv(
        args.output_dir / "elsa_raw_harmonized_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    flow.to_csv(
        args.output_dir / "elsa_cohort_flow.csv", index=False, encoding="utf-8-sig"
    )
    missingness.to_csv(
        args.output_dir / "elsa_feature_missingness.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = write_report(
        args.output_dir,
        root,
        validation,
        flow,
        missingness,
        sample,
        outcome,
        index_sd,
    )
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
