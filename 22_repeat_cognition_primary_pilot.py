from __future__ import annotations

import argparse
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DATA_DIR = PROJECT_DIR / "data_link"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
RANDOM_SEED = 20260715

CHARLS_COLUMNS = [
    "ragender",
    "raeducl",
    "r1agey",
    "r2agey",
    "r1imrc",
    "r1dlrc",
    "r2imrc",
    "r2dlrc",
    "r3imrc",
    "r3dlrc",
    "r2mstat",
    "h2rural",
    "r2shlt",
    "r2cesd10",
    "r2smokev",
    "r2smoken",
    "r2drinkl",
    "r2hibpe",
    "r2diabe",
    "r2hearte",
    "r2stroke",
    "r2adlfive",
    "r2work",
]

HRS_MAIN_COLUMNS = [
    "hhidpn",
    "ragender",
    "r11agey_e",
    "r12agey_e",
    "r11proxy",
    "r12proxy",
    "r13proxy",
    "r11imrc",
    "r11dlrc",
    "r11fimrc",
    "r11fdlrc",
    "r12imrc",
    "r12dlrc",
    "r12fimrc",
    "r12fdlrc",
    "r13imrc",
    "r13dlrc",
    "r13fimrc",
    "r13fdlrc",
    "r12mstat",
    "r12shlt",
    "r12cesd",
    "r12smokev",
    "r12smoken",
    "r12drink",
    "r12hibpe",
    "r12diabe",
    "r12hearte",
    "r12stroke",
    "r12adl5a",
    "r12work",
    "r12demene",
    "r12alzhee",
]

HRS_SUPPLEMENT_COLUMNS = ["hhidpn", "raeducl", "h12rural", "r12adlfive"]

M0_FEATURES = ["age", "female", "education_level"]
M1_FEATURES = M0_FEATURES + ["index_cognition"]
M2_FEATURES = M1_FEATURES + ["previous_cognition"]
M3_FEATURES = M2_FEATURES + [
    "married_partnered",
    "rural",
    "self_rated_health",
    "cesd_fraction",
    "ever_smoked",
    "current_smoker",
    "drinks_alcohol",
    "hypertension",
    "diabetes",
    "heart_disease",
    "stroke",
    "adl_difficulty_count",
    "working",
]

MODEL_FEATURES = {
    "M0_demographics": M0_FEATURES,
    "M1_single_assessment": M1_FEATURES,
    "M2_repeat_assessment": M2_FEATURES,
    "M3_expanded_clinical": M3_FEATURES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pilot the primary single-versus-repeat cognition comparison."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--cv-repeats", type=int, default=5)
    return parser.parse_args()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def complete_sum(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = data[columns].apply(numeric)
    return values.sum(axis=1).where(values.notna().all(axis=1))


def strict_hrs_component(
    data: pd.DataFrame, value_column: str, flag_column: str, proxy_column: str
) -> pd.Series:
    value = numeric(data[value_column])
    flag = numeric(data[flag_column])
    proxy = numeric(data[proxy_column])
    return value.where(flag.eq(0) & proxy.eq(0))


def yes_no(series: pd.Series) -> pd.Series:
    value = numeric(series)
    return value.where(value.isin([0, 1]))


def derive_charls(raw: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=raw.index)
    result["baseline_age"] = numeric(raw["r1agey"])
    result["age"] = numeric(raw["r2agey"])
    gender = numeric(raw["ragender"])
    result["female"] = gender.eq(2).where(gender.notna()).astype(float)
    result["education_level"] = numeric(raw["raeducl"])
    result["previous_cognition"] = complete_sum(raw, ["r1imrc", "r1dlrc"])
    result["index_cognition"] = complete_sum(raw, ["r2imrc", "r2dlrc"])
    result["outcome_cognition"] = complete_sum(raw, ["r3imrc", "r3dlrc"])

    marital = numeric(raw["r2mstat"])
    result["married_partnered"] = (
        marital.isin([1, 2, 3]).where(marital.notna()).astype(float)
    )
    result["rural"] = yes_no(raw["h2rural"])
    result["self_rated_health"] = numeric(raw["r2shlt"])
    result["cesd_fraction"] = numeric(raw["r2cesd10"]) / 30.0
    result["ever_smoked"] = yes_no(raw["r2smokev"])
    result["current_smoker"] = yes_no(raw["r2smoken"])
    result["drinks_alcohol"] = yes_no(raw["r2drinkl"])
    result["hypertension"] = yes_no(raw["r2hibpe"])
    result["diabetes"] = yes_no(raw["r2diabe"])
    result["heart_disease"] = yes_no(raw["r2hearte"])
    result["stroke"] = yes_no(raw["r2stroke"])
    result["adl_difficulty_count"] = numeric(raw["r2adlfive"]).where(
        lambda value: value.between(0, 5)
    )
    result["working"] = yes_no(raw["r2work"])
    result["index_dementia_or_alzheimer"] = np.nan
    return result


def derive_hrs(raw: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=raw.index)
    result["baseline_age"] = numeric(raw["r11agey_e"])
    result["age"] = numeric(raw["r12agey_e"])
    gender = numeric(raw["ragender"])
    result["female"] = gender.eq(2).where(gender.notna()).astype(float)
    result["education_level"] = numeric(raw["raeducl"])

    strict_components: dict[tuple[int, str], pd.Series] = {}
    for wave in (11, 12, 13):
        for component in ("imrc", "dlrc"):
            strict_components[(wave, component)] = strict_hrs_component(
                raw,
                f"r{wave}{component}",
                f"r{wave}f{component}",
                f"r{wave}proxy",
            )
    result["previous_cognition"] = (
        pd.concat(
            [strict_components[(11, "imrc")], strict_components[(11, "dlrc")]],
            axis=1,
        )
        .sum(axis=1)
        .where(
            pd.concat(
                [
                    strict_components[(11, "imrc")],
                    strict_components[(11, "dlrc")],
                ],
                axis=1,
            )
            .notna()
            .all(axis=1)
        )
    )
    result["index_cognition"] = (
        pd.concat(
            [strict_components[(12, "imrc")], strict_components[(12, "dlrc")]],
            axis=1,
        )
        .sum(axis=1)
        .where(
            pd.concat(
                [
                    strict_components[(12, "imrc")],
                    strict_components[(12, "dlrc")],
                ],
                axis=1,
            )
            .notna()
            .all(axis=1)
        )
    )
    result["outcome_cognition"] = (
        pd.concat(
            [strict_components[(13, "imrc")], strict_components[(13, "dlrc")]],
            axis=1,
        )
        .sum(axis=1)
        .where(
            pd.concat(
                [
                    strict_components[(13, "imrc")],
                    strict_components[(13, "dlrc")],
                ],
                axis=1,
            )
            .notna()
            .all(axis=1)
        )
    )

    marital = numeric(raw["r12mstat"])
    result["married_partnered"] = (
        marital.isin([1, 2, 3]).where(marital.notna()).astype(float)
    )
    result["rural"] = yes_no(raw["h12rural"])
    result["self_rated_health"] = numeric(raw["r12shlt"])
    result["cesd_fraction"] = numeric(raw["r12cesd"]) / 8.0
    result["ever_smoked"] = yes_no(raw["r12smokev"])
    result["current_smoker"] = yes_no(raw["r12smoken"])
    result["drinks_alcohol"] = yes_no(raw["r12drink"])
    result["hypertension"] = yes_no(raw["r12hibpe"])
    result["diabetes"] = yes_no(raw["r12diabe"])
    result["heart_disease"] = yes_no(raw["r12hearte"])
    result["stroke"] = yes_no(raw["r12stroke"])
    adl = raw["r12adlfive"].combine_first(raw["r12adl5a"])
    result["adl_difficulty_count"] = numeric(adl).where(
        lambda value: value.between(0, 5)
    )
    result["working"] = yes_no(raw["r12work"])
    dementia = numeric(raw["r12demene"])
    alzheimer = numeric(raw["r12alzhee"])
    result["index_dementia_or_alzheimer"] = np.where(
        dementia.eq(1) | alzheimer.eq(1),
        1.0,
        np.where(dementia.eq(0) & alzheimer.eq(0), 0.0, np.nan),
    )
    return result


def load_cohorts(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    charls = pd.read_stata(
        data_dir / "H_CHARLS_D_Data" / "H_CHARLS_D_Data.dta",
        columns=CHARLS_COLUMNS,
        convert_categoricals=False,
    )
    hrs_main = pd.read_stata(
        data_dir / "randhrs1992_2022v1.dta",
        columns=HRS_MAIN_COLUMNS,
        convert_categoricals=False,
    )
    hrs_supplement = pd.read_stata(
        data_dir / "H_HRS_d.dta",
        columns=HRS_SUPPLEMENT_COLUMNS,
        convert_categoricals=False,
    )
    hrs = hrs_main.merge(
        hrs_supplement, on="hhidpn", how="left", validate="one_to_one"
    )
    return derive_charls(charls), derive_hrs(hrs)


def analysis_sample(
    data: pd.DataFrame, minimum_age: int = 50
) -> tuple[pd.DataFrame, np.ndarray, float]:
    eligible = data["baseline_age"].ge(minimum_age) & data[
        "index_cognition"
    ].notna()
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
        <= -0.5 * index_sd
    ).astype(int)
    return sample, outcome.to_numpy(), index_sd


def preprocessing(features: list[str]) -> ColumnTransformer:
    categorical = ["education_level"]
    numeric_features = [feature for feature in features if feature not in categorical]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def build_model(features: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", preprocessing(features)),
            (
                "model",
                LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000),
            ),
        ]
    )


def repeated_oof_predictions(
    data: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    repeats: int,
    seed: int,
) -> np.ndarray:
    splitter = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=repeats, random_state=seed
    )
    probability_sum = np.zeros(y.size, dtype=float)
    prediction_count = np.zeros(y.size, dtype=int)
    for train_index, test_index in splitter.split(data, y):
        model = build_model(features)
        model.fit(data.iloc[train_index][features], y[train_index])
        probability_sum[test_index] += model.predict_proba(
            data.iloc[test_index][features]
        )[:, 1]
        prediction_count[test_index] += 1
    if not np.all(prediction_count == repeats):
        raise RuntimeError("Each participant must receive one prediction per repeat.")
    return probability_sum / prediction_count


def calibration_parameters(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)
    model.fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def metric_values(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration_parameters(y, probabilities)
    return {
        "auroc": float(roc_auc_score(y, probabilities)),
        "auprc": float(average_precision_score(y, probabilities)),
        "brier": float(brier_score_loss(y, probabilities)),
        "observed_percent": 100.0 * float(y.mean()),
        "mean_predicted_percent": 100.0 * float(probabilities.mean()),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def bootstrap_difference(
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    replicates: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    metric_functions = {
        "auroc": lambda yy, pp: roc_auc_score(yy, pp),
        "auprc": lambda yy, pp: average_precision_score(yy, pp),
        "brier": lambda yy, pp: brier_score_loss(yy, pp),
    }
    observed = {
        metric: function(y, candidate) - function(y, reference)
        for metric, function in metric_functions.items()
    }
    draws = {metric: [] for metric in metric_functions}
    for _ in range(replicates):
        index = rng.integers(0, y.size, y.size)
        if np.unique(y[index]).size < 2:
            continue
        for metric, function in metric_functions.items():
            draws[metric].append(
                function(y[index], candidate[index])
                - function(y[index], reference[index])
            )
    rows = []
    for metric, values in draws.items():
        values_array = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "difference_candidate_minus_reference": observed[metric],
                "ci_low": float(np.quantile(values_array, 0.025)),
                "ci_high": float(np.quantile(values_array, 0.975)),
                "bootstrap_replicates_used": int(values_array.size),
            }
        )
    return rows


def capacity_rows(
    dataset: str,
    model_name: str,
    y: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    n = y.size
    order = np.argsort(-probabilities)
    for fraction in (0.10, 0.20, 0.30):
        selected_n = int(np.ceil(fraction * n))
        selected = np.zeros(n, dtype=bool)
        selected[order[:selected_n]] = True
        true_positive = int(((y == 1) & selected).sum())
        false_positive = int(((y == 0) & selected).sum())
        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "selected_fraction": fraction,
                "selected_n": selected_n,
                "true_positive_n": true_positive,
                "false_positive_n": false_positive,
                "sensitivity": true_positive / y.sum(),
                "positive_predictive_value": true_positive / selected_n,
            }
        )
    return rows


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
        "residence": data["rural"]
        .map({0.0: "urban", 1.0: "rural"})
        .fillna("missing"),
    }


def fairness_rows(
    model_name: str,
    data: pd.DataFrame,
    y: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for subgroup, labels in subgroup_labels(data).items():
        for level in sorted(labels.unique().tolist()):
            mask = labels.eq(level).to_numpy()
            subgroup_y = y[mask]
            subgroup_p = probabilities[mask]
            if subgroup_y.size < 100 or np.unique(subgroup_y).size < 2:
                continue
            rows.append(
                {
                    "model": model_name,
                    "subgroup": subgroup,
                    "level": level,
                    "n": int(subgroup_y.size),
                    "events": int(subgroup_y.sum()),
                    "event_percent": 100.0 * subgroup_y.mean(),
                    "mean_predicted_percent": 100.0 * subgroup_p.mean(),
                    "auroc": float(roc_auc_score(subgroup_y, subgroup_p)),
                    "auprc": float(average_precision_score(subgroup_y, subgroup_p)),
                    "brier": float(brier_score_loss(subgroup_y, subgroup_p)),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    charls_all, hrs_all = load_cohorts(args.data_dir)
    charls, charls_y, charls_index_sd = analysis_sample(charls_all)
    hrs, hrs_y, hrs_index_sd = analysis_sample(hrs_all)

    performance_rows: list[dict[str, object]] = []
    capacity_output: list[dict[str, object]] = []
    fairness_output: list[dict[str, object]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}

    for model_index, (model_name, features) in enumerate(MODEL_FEATURES.items()):
        charls_probabilities = repeated_oof_predictions(
            charls,
            charls_y,
            features,
            args.cv_repeats,
            RANDOM_SEED + model_index,
        )
        final_model = build_model(features)
        final_model.fit(charls[features], charls_y)
        hrs_probabilities = final_model.predict_proba(hrs[features])[:, 1]
        predictions[("CHARLS repeated 5-fold CV", model_name)] = charls_probabilities
        predictions[("HRS external validation", model_name)] = hrs_probabilities

        for dataset, y, probabilities in (
            ("CHARLS repeated 5-fold CV", charls_y, charls_probabilities),
            ("HRS external validation", hrs_y, hrs_probabilities),
        ):
            row = {
                "dataset": dataset,
                "model": model_name,
                "n": int(y.size),
                "events": int(y.sum()),
            }
            row.update(metric_values(y, probabilities))
            performance_rows.append(row)
            capacity_output.extend(
                capacity_rows(dataset, model_name, y, probabilities)
            )
        if model_name in ("M1_single_assessment", "M2_repeat_assessment"):
            fairness_output.extend(
                fairness_rows(model_name, hrs, hrs_y, hrs_probabilities)
            )

    comparisons: list[dict[str, object]] = []
    for dataset, y in (
        ("CHARLS repeated 5-fold CV", charls_y),
        ("HRS external validation", hrs_y),
    ):
        for reference_name, candidate_name in (
            ("M1_single_assessment", "M2_repeat_assessment"),
            ("M2_repeat_assessment", "M3_expanded_clinical"),
        ):
            rows = bootstrap_difference(
                y,
                predictions[(dataset, reference_name)],
                predictions[(dataset, candidate_name)],
                args.bootstrap,
                RANDOM_SEED + len(comparisons),
            )
            for row in rows:
                row.update(
                    {
                        "dataset": dataset,
                        "reference_model": reference_name,
                        "candidate_model": candidate_name,
                    }
                )
                comparisons.append(row)

    performance = pd.DataFrame(performance_rows)
    comparison_table = pd.DataFrame(comparisons)
    capacity_table = pd.DataFrame(capacity_output)
    fairness_table = pd.DataFrame(fairness_output)

    performance.to_csv(
        args.output_dir / "repeat_cognition_primary_pilot_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison_table.to_csv(
        args.output_dir / "repeat_cognition_primary_pilot_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    capacity_table.to_csv(
        args.output_dir / "repeat_cognition_primary_pilot_capacity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fairness_table.to_csv(
        args.output_dir / "repeat_cognition_primary_pilot_fairness_hrs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = [
        "# 重复认知测评主要比较：第一轮预测试",
        "",
        "日期：2026-07-15",
        "",
        f"CHARLS：N={charls_y.size:,}，事件={charls_y.sum():,}，2013评分SD={charls_index_sd:.3f}。",
        f"HRS：N={hrs_y.size:,}，事件={hrs_y.sum():,}，2014评分SD={hrs_index_sd:.3f}。",
        "",
        "本轮只运行预先规定的逻辑回归层级，用于验证核心研究假设；尚未运行机器学习、IPCW、调查权重和全部敏感性分析。",
        "",
        "## 性能",
        "",
        "| 数据集 | 模型 | AUROC | AUPRC | Brier | 实际率 | 平均预测率 | 校准截距 | 校准斜率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in performance.iterrows():
        report.append(
            "| {dataset} | {model} | {auroc:.3f} | {auprc:.3f} | {brier:.3f} | {observed:.1f}% | {predicted:.1f}% | {intercept:.3f} | {slope:.3f} |".format(
                dataset=row["dataset"],
                model=row["model"],
                auroc=row["auroc"],
                auprc=row["auprc"],
                brier=row["brier"],
                observed=row["observed_percent"],
                predicted=row["mean_predicted_percent"],
                intercept=row["calibration_intercept"],
                slope=row["calibration_slope"],
            )
        )

    report.extend(
        [
            "",
            "## 预先指定的增量比较",
            "",
            "| 数据集 | 候选模型 vs 参照 | 指标 | 差值 | 95% CI |",
            "|---|---|---|---:|---:|",
        ]
    )
    for _, row in comparison_table.iterrows():
        report.append(
            "| {dataset} | {candidate} vs {reference} | {metric} | {difference:.4f} | {low:.4f} to {high:.4f} |".format(
                dataset=row["dataset"],
                candidate=row["candidate_model"],
                reference=row["reference_model"],
                metric=row["metric"],
                difference=row["difference_candidate_minus_reference"],
                low=row["ci_low"],
                high=row["ci_high"],
            )
        )

    report.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 本结果是按冻结方案执行的第一轮预测试，不替代最终嵌套验证和确认性ELSA分析。",
            "- HRS未用于重新训练或选择逻辑回归变量；其结果为原样外部应用。",
            "- 所有个体级数据只在内存中使用，保存结果均为汇总统计。",
        ]
    )
    (args.output_dir / "repeat_cognition_primary_pilot_2026-07-15.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(performance.to_string(index=False))
    print(comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()
