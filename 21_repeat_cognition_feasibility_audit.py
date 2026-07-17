from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data_link"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"

CHARLS_FILE = Path("H_CHARLS_D_Data") / "H_CHARLS_D_Data.dta"
HRS_FILE = Path("randhrs1992_2022v1.dta")

CHARLS_COLUMNS = [
    "ID",
    "ragender",
    "raeducl",
    "raeduc_c",
    "r1agey",
    "r1iwstat",
    "r2iwstat",
    "r3iwstat",
    "r4iwstat",
    "r1wtrespb",
    "r2wtrespb",
    "r3wtrespb",
    "r4wtrespb",
    "r1imrc",
    "r1dlrc",
    "r1ser7",
    "r2imrc",
    "r2dlrc",
    "r2ser7",
    "r3imrc",
    "r3dlrc",
    "r3ser7",
    "r4imrc",
    "r4dlrc",
    "r4ser7",
    "r1stroke",
    "r2stroke",
    "r3stroke",
    "r4stroke",
]

HRS_COLUMNS = [
    "hhidpn",
    "ragender",
    "raeduc",
    "raedyrs",
    "r11agey_e",
    "r11iwstat",
    "r12iwstat",
    "r13iwstat",
    "r14iwstat",
    "r11proxy",
    "r12proxy",
    "r13proxy",
    "r14proxy",
    "r11wtcrnh",
    "r12wtcrnh",
    "r13wtcrnh",
    "r14wtcrnh",
    "r11imrc",
    "r11dlrc",
    "r11ser7",
    "r11fimrc",
    "r11fdlrc",
    "r11fser7",
    "r12imrc",
    "r12dlrc",
    "r12ser7",
    "r12fimrc",
    "r12fdlrc",
    "r12fser7",
    "r13imrc",
    "r13dlrc",
    "r13ser7",
    "r13fimrc",
    "r13fdlrc",
    "r13fser7",
    "r14imrcp",
    "r14imrcw",
    "r14dlrcp",
    "r14dlrcw",
    "r14ser7p",
    "r14ser7w",
    "r14fimrcp",
    "r14fimrcw",
    "r14fdlrcp",
    "r14fdlrcw",
    "r14fser7p",
    "r14fser7w",
    "r12demene",
    "r12alzhee",
    "r11stroke",
    "r12stroke",
    "r13stroke",
    "r14stroke",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit three-wave repeated cognition feasibility in CHARLS and HRS."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def metadata(path: Path) -> dict[str, str]:
    reader = pd.io.stata.StataReader(path)
    return reader.variable_labels()


def require_columns(
    labels: dict[str, str], columns: list[str], source: str
) -> None:
    missing = [column for column in columns if column not in labels]
    if missing:
        raise KeyError(f"Missing columns in {source}: {missing}")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def complete_sum(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = data[columns].apply(numeric)
    complete = values.notna().all(axis=1)
    return values.sum(axis=1).where(complete)


def observed_component(
    data: pd.DataFrame, value_column: str, flag_column: str
) -> pd.Series:
    value = numeric(data[value_column])
    flag = numeric(data[flag_column])
    return value.where(flag.eq(0))


def combined_mode_component(
    data: pd.DataFrame,
    p_column: str,
    w_column: str,
    p_flag: str,
    w_flag: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    p_value = numeric(data[p_column])
    w_value = numeric(data[w_column])
    p_strict = p_value.where(numeric(data[p_flag]).eq(0))
    w_strict = w_value.where(numeric(data[w_flag]).eq(0))
    all_modes = p_value.combine_first(w_value)
    strict_all_modes = p_strict.combine_first(w_strict)
    strict_p_only = p_strict
    return all_modes, strict_all_modes, strict_p_only


def prepare_charls(raw: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=raw.index)
    result["id"] = raw["ID"]
    result["age"] = numeric(raw["r1agey"])
    result["education"] = numeric(raw["raeducl"])
    result["female"] = numeric(raw["ragender"]).eq(2).astype(float)

    for wave in (1, 2, 3, 4):
        result[f"iwstat_{wave}"] = numeric(raw[f"r{wave}iwstat"])
        result[f"proxy_{wave}"] = np.nan
        result[f"weight_{wave}"] = numeric(raw[f"r{wave}wtrespb"])
        result[f"imrc_{wave}"] = numeric(raw[f"r{wave}imrc"])
        result[f"dlrc_{wave}"] = numeric(raw[f"r{wave}dlrc"])
        result[f"ser7_{wave}"] = numeric(raw[f"r{wave}ser7"])
        result[f"memory20_{wave}"] = complete_sum(
            raw, [f"r{wave}imrc", f"r{wave}dlrc"]
        )
        result[f"cog25_{wave}"] = complete_sum(
            raw, [f"r{wave}imrc", f"r{wave}dlrc", f"r{wave}ser7"]
        )
        result[f"stroke_{wave}"] = numeric(raw[f"r{wave}stroke"])

    result["index_dementia_or_alzheimer"] = np.nan
    result["w14_mode"] = "not_applicable"
    return result


def prepare_hrs(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = pd.DataFrame(index=raw.index)
    result["id"] = raw["hhidpn"]
    result["age"] = numeric(raw["r11agey_e"])
    result["education"] = numeric(raw["raeduc"])
    result["female"] = numeric(raw["ragender"]).eq(2).astype(float)

    flag_rows: list[dict[str, object]] = []
    for wave, source_wave in ((1, 11), (2, 12), (3, 13)):
        result[f"iwstat_{wave}"] = numeric(raw[f"r{source_wave}iwstat"])
        result[f"proxy_{wave}"] = numeric(raw[f"r{source_wave}proxy"])
        result[f"weight_{wave}"] = numeric(raw[f"r{source_wave}wtcrnh"])
        result[f"stroke_{wave}"] = numeric(raw[f"r{source_wave}stroke"])

        component_series: dict[str, pd.Series] = {}
        strict_component_series: dict[str, pd.Series] = {}
        for component in ("imrc", "dlrc", "ser7"):
            value_column = f"r{source_wave}{component}"
            flag_column = f"r{source_wave}f{component}"
            component_series[component] = numeric(raw[value_column])
            strict_component_series[component] = observed_component(
                raw, value_column, flag_column
            )
            for value, count in (
                numeric(raw[flag_column]).value_counts(dropna=False).sort_index().items()
            ):
                flag_rows.append(
                    {
                        "cohort": "HRS",
                        "wave": source_wave,
                        "component": component,
                        "imputation_flag": value,
                        "n": int(count),
                    }
                )

        result[f"imrc_{wave}"] = component_series["imrc"]
        result[f"dlrc_{wave}"] = component_series["dlrc"]
        result[f"ser7_{wave}"] = component_series["ser7"]
        result[f"memory20_{wave}_rand"] = (
            pd.concat([component_series["imrc"], component_series["dlrc"]], axis=1)
            .sum(axis=1)
            .where(
                pd.concat(
                    [component_series["imrc"], component_series["dlrc"]], axis=1
                )
                .notna()
                .all(axis=1)
            )
        )
        result[f"memory20_{wave}"] = (
            pd.concat(
                [strict_component_series["imrc"], strict_component_series["dlrc"]],
                axis=1,
            )
            .sum(axis=1)
            .where(
                pd.concat(
                    [
                        strict_component_series["imrc"],
                        strict_component_series["dlrc"],
                    ],
                    axis=1,
                )
                .notna()
                .all(axis=1)
            )
        )
        result[f"cog25_{wave}"] = (
            pd.concat(list(strict_component_series.values()), axis=1)
            .sum(axis=1)
            .where(
                pd.concat(list(strict_component_series.values()), axis=1)
                .notna()
                .all(axis=1)
            )
        )

    result["iwstat_4"] = numeric(raw["r14iwstat"])
    result["proxy_4"] = numeric(raw["r14proxy"])
    result["weight_4"] = numeric(raw["r14wtcrnh"])
    result["stroke_4"] = numeric(raw["r14stroke"])

    w14: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}
    for component in ("imrc", "dlrc", "ser7"):
        w14[component] = combined_mode_component(
            raw,
            f"r14{component}p",
            f"r14{component}w",
            f"r14f{component}p",
            f"r14f{component}w",
        )
        for mode, flag_column in (
            ("p", f"r14f{component}p"),
            ("w", f"r14f{component}w"),
        ):
            for value, count in (
                numeric(raw[flag_column]).value_counts(dropna=False).sort_index().items()
            ):
                flag_rows.append(
                    {
                        "cohort": "HRS",
                        "wave": 14,
                        "component": f"{component}_{mode}",
                        "imputation_flag": value,
                        "n": int(count),
                    }
                )

    for component in ("imrc", "dlrc", "ser7"):
        result[f"{component}_4"] = w14[component][1]

    result["memory20_4"] = (
        pd.concat([w14["imrc"][1], w14["dlrc"][1]], axis=1)
        .sum(axis=1)
        .where(
            pd.concat([w14["imrc"][1], w14["dlrc"][1]], axis=1)
            .notna()
            .all(axis=1)
        )
    )
    result["memory20_4_p_only"] = (
        pd.concat([w14["imrc"][2], w14["dlrc"][2]], axis=1)
        .sum(axis=1)
        .where(
            pd.concat([w14["imrc"][2], w14["dlrc"][2]], axis=1)
            .notna()
            .all(axis=1)
        )
    )
    result["cog25_4"] = (
        pd.concat([w14["imrc"][1], w14["dlrc"][1], w14["ser7"][1]], axis=1)
        .sum(axis=1)
        .where(
            pd.concat(
                [w14["imrc"][1], w14["dlrc"][1], w14["ser7"][1]], axis=1
            )
            .notna()
            .all(axis=1)
        )
    )

    p_observed = numeric(raw["r14imrcp"]).notna() | numeric(raw["r14dlrcp"]).notna()
    w_observed = numeric(raw["r14imrcw"]).notna() | numeric(raw["r14dlrcw"]).notna()
    result["w14_mode"] = np.select(
        [p_observed & ~w_observed, w_observed & ~p_observed, p_observed & w_observed],
        ["ftf_or_phone", "web", "both"],
        default="none",
    )

    dementia = numeric(raw["r12demene"])
    alzheimer = numeric(raw["r12alzhee"])
    result["index_dementia_or_alzheimer"] = np.where(
        dementia.eq(1) | alzheimer.eq(1),
        1.0,
        np.where(dementia.eq(0) & alzheimer.eq(0), 0.0, np.nan),
    )
    return result, pd.DataFrame(flag_rows)


def enforce_direct_scores(data: pd.DataFrame, cohort: str) -> pd.DataFrame:
    result = data.copy()
    if cohort == "HRS":
        for wave in (1, 2, 3, 4):
            direct = result[f"proxy_{wave}"].eq(0)
            for score in ("memory20", "cog25"):
                column = f"{score}_{wave}"
                result[column] = result[column].where(direct)
            if wave == 4:
                result["memory20_4_p_only"] = result["memory20_4_p_only"].where(
                    direct
                )
    return result


def add_score_summary(
    rows: list[dict[str, object]],
    cohort: str,
    data: pd.DataFrame,
    age_min: int,
    score_name: str,
    waves: list[tuple[int, str]],
) -> None:
    eligible = data["age"].ge(age_min)
    for wave, year in waves:
        column = f"{score_name}_{wave}"
        observed = numeric(data.loc[eligible, column]).dropna()
        rows.append(
            {
                "cohort": cohort,
                "age_min": age_min,
                "wave": wave,
                "year": year,
                "score": score_name,
                "eligible_n": int(eligible.sum()),
                "observed_n": int(observed.size),
                "observed_percent": (
                    100.0 * observed.size / eligible.sum() if eligible.sum() else np.nan
                ),
                "mean": float(observed.mean()) if not observed.empty else np.nan,
                "sd": float(observed.std(ddof=1)) if observed.size > 1 else np.nan,
                "min": float(observed.min()) if not observed.empty else np.nan,
                "max": float(observed.max()) if not observed.empty else np.nan,
            }
        )


def audit_window(
    cohort: str,
    horizon: str,
    data: pd.DataFrame,
    age_min: int,
    first_column: str,
    second_column: str,
    outcome_column: str,
    outcome_note: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    age_known = data["age"].notna()
    eligible = age_known & data["age"].ge(age_min)
    first = eligible & data[first_column].notna()
    second = first & data[second_column].notna()
    complete = second & data[outcome_column].notna()

    stages = [
        ("all_persons", pd.Series(True, index=data.index)),
        ("age_known", age_known),
        (f"age_ge_{age_min}", eligible),
        ("first_assessment_complete", first),
        ("second_assessment_complete", second),
        ("outcome_assessment_complete", complete),
    ]
    flow_rows: list[dict[str, object]] = []
    previous_n: int | None = None
    for stage, mask in stages:
        n = int(mask.sum())
        flow_rows.append(
            {
                "cohort": cohort,
                "horizon": horizon,
                "age_min": age_min,
                "stage": stage,
                "n": n,
                "percent_of_previous": (
                    100.0 * n / previous_n if previous_n not in (None, 0) else np.nan
                ),
            }
        )
        previous_n = n

    index_sd = float(data.loc[second, second_column].std(ddof=1))
    threshold = 0.5 * index_sd
    threshold_1sd = index_sd
    change = data.loc[complete, outcome_column] - data.loc[complete, second_column]
    decline = change.le(-threshold)
    decline_1sd = change.le(-threshold_1sd)
    outcome_row = {
        "cohort": cohort,
        "horizon": horizon,
        "age_min": age_min,
        "score": second_column.rsplit("_", 1)[0],
        "complete_n": int(complete.sum()),
        "index_score_sd": index_sd,
        "simple_0_5sd_threshold_points": threshold,
        "mean_change_outcome_minus_second": float(change.mean()),
        "change_sd": float(change.std(ddof=1)),
        "simple_0_5sd_decline_events": int(decline.sum()),
        "simple_0_5sd_decline_percent": (
            100.0 * decline.mean() if not decline.empty else np.nan
        ),
        "simple_1sd_threshold_points": threshold_1sd,
        "simple_1sd_decline_events": int(decline_1sd.sum()),
        "simple_1sd_decline_percent": (
            100.0 * decline_1sd.mean() if not decline_1sd.empty else np.nan
        ),
        "note": outcome_note,
    }
    return flow_rows, outcome_row


def status_rows(
    cohort: str, data: pd.DataFrame, wave_years: list[tuple[int, str]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for wave, year in wave_years:
        counts = data[f"iwstat_{wave}"].value_counts(dropna=False).sort_index()
        for value, count in counts.items():
            rows.append(
                {
                    "cohort": cohort,
                    "wave": wave,
                    "year": year,
                    "iwstat_value": value,
                    "n": int(count),
                }
            )
    return rows


def missingness_rows(
    cohort: str,
    data: pd.DataFrame,
    age_min: int,
    wave_years: list[tuple[int, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    eligible = data["age"].ge(age_min)
    eligible_n = int(eligible.sum())
    for wave, year in wave_years:
        for variable in ("imrc", "dlrc", "ser7", "memory20", "cog25"):
            column = f"{variable}_{wave}"
            missing_n = int(data.loc[eligible, column].isna().sum())
            rows.append(
                {
                    "cohort": cohort,
                    "age_min": age_min,
                    "wave": wave,
                    "year": year,
                    "variable": variable,
                    "eligible_n": eligible_n,
                    "missing_n": missing_n,
                    "missing_percent": (
                        100.0 * missing_n / eligible_n if eligible_n else np.nan
                    ),
                }
            )
    return rows


def proxy_rows(
    cohort: str,
    data: pd.DataFrame,
    age_min: int,
    wave_years: list[tuple[int, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    eligible = data["age"].ge(age_min)
    for wave, year in wave_years:
        proxy = data.loc[eligible, f"proxy_{wave}"]
        if proxy.notna().sum() == 0:
            rows.append(
                {
                    "cohort": cohort,
                    "age_min": age_min,
                    "wave": wave,
                    "year": year,
                    "proxy_status": "not_available",
                    "n": int(eligible.sum()),
                }
            )
            continue
        for status, count in (
            proxy.map({0.0: "direct", 1.0: "proxy"})
            .fillna("missing_or_not_interviewed")
            .value_counts()
            .items()
        ):
            rows.append(
                {
                    "cohort": cohort,
                    "age_min": age_min,
                    "wave": wave,
                    "year": year,
                    "proxy_status": status,
                    "n": int(count),
                }
            )
    return rows


def format_n(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{int(value):,}"


def format_pct(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.1f}%"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    charls_path = args.data_dir / CHARLS_FILE
    hrs_path = args.data_dir / HRS_FILE
    if not charls_path.exists() or not hrs_path.exists():
        raise FileNotFoundError(f"Missing {charls_path} or {hrs_path}")

    charls_labels = metadata(charls_path)
    hrs_labels = metadata(hrs_path)
    require_columns(charls_labels, CHARLS_COLUMNS, str(CHARLS_FILE))
    require_columns(hrs_labels, HRS_COLUMNS, str(HRS_FILE))

    variable_rows: list[dict[str, object]] = []
    for cohort, labels, columns in (
        ("CHARLS", charls_labels, CHARLS_COLUMNS),
        ("HRS", hrs_labels, HRS_COLUMNS),
    ):
        for column in columns:
            variable_rows.append(
                {
                    "cohort": cohort,
                    "variable": column,
                    "label": labels[column],
                    "available": True,
                }
            )
    pd.DataFrame(variable_rows).to_csv(
        args.output_dir / "repeat_cognition_variable_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    charls_raw = pd.read_stata(
        charls_path, columns=CHARLS_COLUMNS, convert_categoricals=False
    )
    hrs_raw = pd.read_stata(
        hrs_path, columns=HRS_COLUMNS, convert_categoricals=False
    )
    if charls_raw["ID"].duplicated().any():
        raise ValueError("CHARLS ID is not unique.")
    if hrs_raw["hhidpn"].duplicated().any():
        raise ValueError("HRS hhidpn is not unique.")

    charls = enforce_direct_scores(prepare_charls(charls_raw), "CHARLS")
    hrs, flag_table = prepare_hrs(hrs_raw)
    hrs = enforce_direct_scores(hrs, "HRS")
    flag_table.to_csv(
        args.output_dir / "repeat_cognition_hrs_imputation_flags.csv",
        index=False,
        encoding="utf-8-sig",
    )

    score_rows: list[dict[str, object]] = []
    for age_min in (50, 65):
        add_score_summary(
            score_rows,
            "CHARLS",
            charls,
            age_min,
            "memory20",
            [(1, "2011"), (2, "2013"), (3, "2015"), (4, "2018")],
        )
        add_score_summary(
            score_rows,
            "CHARLS",
            charls,
            age_min,
            "cog25",
            [(1, "2011"), (2, "2013"), (3, "2015"), (4, "2018")],
        )
        add_score_summary(
            score_rows,
            "HRS",
            hrs,
            age_min,
            "memory20",
            [(1, "2012"), (2, "2014"), (3, "2016"), (4, "2018")],
        )
        add_score_summary(
            score_rows,
            "HRS",
            hrs,
            age_min,
            "cog25",
            [(1, "2012"), (2, "2014"), (3, "2016"), (4, "2018")],
        )
    score_table = pd.DataFrame(score_rows)
    score_table.to_csv(
        args.output_dir / "repeat_cognition_score_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    flow_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    windows = [
        (
            "CHARLS",
            "primary_2011_2013_to_2015",
            charls,
            "memory20_1",
            "memory20_2",
            "memory20_3",
            "Prespecified primary 0.5-SD feasibility definition; formal models are not fitted.",
        ),
        (
            "CHARLS",
            "longterm_2011_2013_to_2018",
            charls,
            "memory20_1",
            "memory20_2",
            "memory20_4",
            "Raw feasibility only; CHARLS 2018 measurement changes require equating.",
        ),
        (
            "HRS",
            "primary_2012_2014_to_2016",
            hrs,
            "memory20_1",
            "memory20_2",
            "memory20_3",
            "Strictly observed RAND cognition components and direct interviews.",
        ),
        (
            "HRS",
            "longterm_2012_2014_to_2018",
            hrs,
            "memory20_1",
            "memory20_2",
            "memory20_4",
            "Includes direct 2018 ftf/telephone and web modes; mode sensitivity needed.",
        ),
    ]
    for age_min in (50, 65):
        for (
            cohort,
            horizon,
            data,
            first_column,
            second_column,
            outcome_column,
            note,
        ) in windows:
            flow, outcome = audit_window(
                cohort,
                horizon,
                data,
                age_min,
                first_column,
                second_column,
                outcome_column,
                note,
            )
            flow_rows.extend(flow)
            outcome_rows.append(outcome)

    flow_table = pd.DataFrame(flow_rows)
    outcome_table = pd.DataFrame(outcome_rows)
    flow_table.to_csv(
        args.output_dir / "repeat_cognition_sample_flow.csv",
        index=False,
        encoding="utf-8-sig",
    )
    outcome_table.to_csv(
        args.output_dir / "repeat_cognition_outcome_feasibility.csv",
        index=False,
        encoding="utf-8-sig",
    )

    interview_status = pd.DataFrame(
        status_rows(
            "CHARLS", charls, [(1, "2011"), (2, "2013"), (3, "2015"), (4, "2018")]
        )
        + status_rows(
            "HRS", hrs, [(1, "2012"), (2, "2014"), (3, "2016"), (4, "2018")]
        )
    )
    interview_status.to_csv(
        args.output_dir / "repeat_cognition_interview_status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missingness = pd.DataFrame(
        missingness_rows(
            "CHARLS",
            charls,
            50,
            [(1, "2011"), (2, "2013"), (3, "2015"), (4, "2018")],
        )
        + missingness_rows(
            "CHARLS",
            charls,
            65,
            [(1, "2011"), (2, "2013"), (3, "2015"), (4, "2018")],
        )
        + missingness_rows(
            "HRS",
            hrs,
            50,
            [(1, "2012"), (2, "2014"), (3, "2016"), (4, "2018")],
        )
        + missingness_rows(
            "HRS",
            hrs,
            65,
            [(1, "2012"), (2, "2014"), (3, "2016"), (4, "2018")],
        )
    )
    missingness.to_csv(
        args.output_dir / "repeat_cognition_missingness.csv",
        index=False,
        encoding="utf-8-sig",
    )

    proxy_summary = pd.DataFrame(
        proxy_rows(
            "CHARLS",
            charls,
            50,
            [(1, "2011"), (2, "2013"), (3, "2015"), (4, "2018")],
        )
        + proxy_rows(
            "HRS",
            hrs,
            50,
            [(1, "2012"), (2, "2014"), (3, "2016"), (4, "2018")],
        )
    )
    proxy_summary.to_csv(
        args.output_dir / "repeat_cognition_proxy_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hrs_mode_rows = []
    for age_min in (50, 65):
        eligible = hrs["age"].ge(age_min)
        for mode, count in hrs.loc[eligible, "w14_mode"].value_counts().items():
            hrs_mode_rows.append(
                {
                    "age_min": age_min,
                    "w14_mode": mode,
                    "n": int(count),
                }
            )
    pd.DataFrame(hrs_mode_rows).to_csv(
        args.output_dir / "repeat_cognition_hrs_2018_mode.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report_lines = [
        "# CHARLS-HRS重复认知测评三波可行性审计",
        "",
        "日期：2026-07-15",
        "",
        "本审计只输出汇总统计，不保存个体级数据，不比较候选预测模型性能。",
        "",
        "## 主要20分记忆评分的三波样本与简单事件数",
        "",
        "| 队列 | 时间窗 | 年龄 | 三波完整N | 简单0.5 SD下降事件 | 事件率 | 平均变化 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in outcome_table.iterrows():
        report_lines.append(
            "| {cohort} | {horizon} | >= {age_min} | {n} | {events} | {rate} | {change:.2f} |".format(
                cohort=row["cohort"],
                horizon=row["horizon"],
                age_min=int(row["age_min"]),
                n=format_n(row["complete_n"]),
                events=format_n(row["simple_0_5sd_decline_events"]),
                rate=format_pct(row["simple_0_5sd_decline_percent"]),
                change=float(row["mean_change_outcome_minus_second"]),
            )
        )

    hrs_age50 = hrs["age"].ge(50)
    hrs_index_observed = hrs_age50 & hrs["memory20_2"].notna()
    hrs_dementia_known = hrs_index_observed & hrs[
        "index_dementia_or_alzheimer"
    ].notna()
    hrs_dementia_yes = hrs_index_observed & hrs[
        "index_dementia_or_alzheimer"
    ].eq(1)
    report_lines.extend(
        [
            "",
            "## 测量与纳入标准审计结论",
            "",
            "- CHARLS和HRS均具备主要三波20分记忆评分，且可构建25分敏感性评分。",
            "- HRS主要评分已限制为直接访谈且两个记忆组成部分均未使用RAND插补值。",
            "- CHARLS Harmonized Version D未发现与HRS可直接对齐的代理访谈标记或痴呆/阿尔茨海默病诊断字段；主要评分完整本身可排除无法直接完成认知测验者，但显式诊断排除不能跨队列对称实施。",
            f"- HRS年龄>=50且2014记忆评分完整者中，痴呆/阿尔茨海默病状态已知{format_n(hrs_dementia_known.sum())}人，报告任一诊断{format_n(hrs_dementia_yes.sum())}人；建议仅作HRS敏感性排除，不进入跨队列主要纳入规则。",
            "- CHARLS 2018原始长期事件数只用于样本量审计，正式分析前必须处理词表/实施方式变化。",
            "- HRS 2018同时存在面对面/电话与网络认知测验，长期分析需做访谈模式限制或校正。",
            "- ELSA未读取，继续保留为模型冻结后的确认性队列。",
            "",
            "## 下一步",
            "",
            "1. 显式痴呆诊断排除已移至HRS敏感性分析，并记录为方案修订1。",
            "2. 冻结单次、重复测评、扩展临床和机器学习模型的精确变量与训练流程。",
            "3. 建立主要分析数据；所有可更新临床变量取第二次测评/预测起点。",
        ]
    )
    (args.output_dir / "repeat_cognition_feasibility_audit_2026-07-15.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(outcome_table.to_string(index=False))


if __name__ == "__main__":
    main()
