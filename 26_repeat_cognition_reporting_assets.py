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
from PIL import Image, ImageDraw, ImageFont


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


pilot = load_module("22_repeat_cognition_primary_pilot.py", "pilot")
phase2 = load_module("23_repeat_cognition_sensitivity_recalibration.py", "phase2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create aggregate reporting assets for repeat cognition analyses."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cv-repeats", type=int, default=5)
    return parser.parse_args()


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def primary_predictions(data_dir: Path, repeats: int):
    charls_all, hrs_all = pilot.load_cohorts(data_dir)
    charls, charls_y, _ = pilot.analysis_sample(charls_all, minimum_age=50)
    hrs, hrs_y, _ = pilot.analysis_sample(hrs_all, minimum_age=50)
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for idx, model_name in enumerate(["M1_single_assessment", "M2_repeat_assessment"]):
        features = pilot.MODEL_FEATURES[model_name]
        charls_prob = pilot.repeated_oof_predictions(
            charls, charls_y, features, repeats, RANDOM_SEED + idx
        )
        model = pilot.build_model(features)
        model.fit(charls[features], charls_y)
        hrs_prob = model.predict_proba(hrs[features])[:, 1]
        predictions[("CHARLS repeated 5-fold CV", model_name)] = charls_prob
        predictions[("HRS external validation", model_name)] = hrs_prob
    predictions[("HRS external validation", "M2_recalibrated")] = (
        phase2.cross_validated_recalibration(
            hrs_y,
            predictions[("HRS external validation", "M2_repeat_assessment")],
            "intercept_slope",
        )
    )
    return charls_y, hrs_y, predictions


def decision_curve(y: np.ndarray, probabilities: np.ndarray, model_name: str, dataset: str) -> list[dict[str, object]]:
    rows = []
    n = y.size
    prevalence = float(y.mean())
    for threshold in np.arange(0.05, 0.61, 0.05):
        selected = probabilities >= threshold
        tp = int(((y == 1) & selected).sum())
        fp = int(((y == 0) & selected).sum())
        odds = threshold / (1 - threshold)
        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "threshold": round(float(threshold), 2),
                "net_benefit": tp / n - fp / n * odds,
                "treat_all_net_benefit": prevalence - (1 - prevalence) * odds,
                "treat_none_net_benefit": 0.0,
            }
        )
    return rows


def calibration_deciles(y: np.ndarray, probabilities: np.ndarray, model_name: str, dataset: str) -> list[dict[str, object]]:
    frame = pd.DataFrame({"y": y, "p": probabilities})
    frame["decile"] = pd.qcut(frame["p"], 10, labels=False, duplicates="drop") + 1
    rows = []
    for decile, group in frame.groupby("decile", observed=True):
        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "decile": int(decile),
                "n": int(group.shape[0]),
                "events": int(group["y"].sum()),
                "mean_predicted": float(group["p"].mean()),
                "observed": float(group["y"].mean()),
            }
        )
    return rows


def draw_axes(draw: ImageDraw.ImageDraw, origin: tuple[int, int], size: tuple[int, int], title: str, x_label: str, y_label: str) -> None:
    x0, y0 = origin
    w, h = size
    font = get_font(22)
    small = get_font(16)
    title_font = get_font(28, bold=True)
    draw.text((x0, 24), title, fill=(20, 20, 20), font=title_font)
    draw.line((x0, y0, x0, y0 - h, x0 + w, y0 - h), fill=(30, 30, 30), width=2)
    draw.line((x0, y0, x0 + w, y0), fill=(30, 30, 30), width=2)
    draw.text((x0 + w // 2 - 60, y0 + 42), x_label, fill=(20, 20, 20), font=font)
    draw.text((x0, y0 - h - 34), y_label, fill=(20, 20, 20), font=small)


def save_decision_curve_plot(decision: pd.DataFrame, output_dir: Path) -> None:
    img = Image.new("RGB", (1200, 760), "white")
    draw = ImageDraw.Draw(img)
    x0, y0, w, h = 110, 650, 980, 520
    draw_axes(draw, (x0, y0), (w, h), "Decision curve: HRS external validation", "Risk threshold", "Net benefit")
    subset = decision[decision["dataset"].eq("HRS external validation")]
    y_min = -0.08
    y_max = max(float(subset["net_benefit"].max()), 0.30)
    colors = {
        "M1_single_assessment": (90, 90, 90),
        "M2_repeat_assessment": (30, 105, 170),
        "M2_recalibrated": (190, 80, 45),
    }
    labels = {
        "M1_single_assessment": "M1 single",
        "M2_repeat_assessment": "M2 repeat",
        "M2_recalibrated": "M2 recalibrated",
    }

    def xy(threshold: float, benefit: float) -> tuple[int, int]:
        benefit = min(max(benefit, y_min), y_max)
        x = x0 + int((threshold - 0.05) / 0.55 * w)
        y = y0 - int((benefit - y_min) / (y_max - y_min) * h)
        return x, y

    small = get_font(16)
    for tick in np.arange(0.1, 0.61, 0.1):
        x, _ = xy(float(tick), y_min)
        draw.line((x, y0, x, y0 + 6), fill=(30, 30, 30), width=1)
        draw.text((x - 14, y0 + 10), f"{tick:.1f}", fill=(30, 30, 30), font=small)
    for tick in np.linspace(y_min, y_max, 6):
        _, y = xy(0.05, float(tick))
        draw.line((x0 - 6, y, x0, y), fill=(30, 30, 30), width=1)
        draw.text((x0 - 84, y - 8), f"{tick:.2f}", fill=(30, 30, 30), font=small)

    for model_name, color in colors.items():
        model_data = subset[subset["model"].eq(model_name)].sort_values("threshold")
        points = [xy(row.threshold, row.net_benefit) for row in model_data.itertuples()]
        draw.line(points, fill=color, width=4)
    all_points = [xy(row.threshold, row.treat_all_net_benefit) for row in subset[subset["model"].eq("M2_repeat_assessment")].sort_values("threshold").itertuples()]
    draw.line(all_points, fill=(120, 120, 120), width=2)
    draw.line((x0, xy(0.05, 0.0)[1], x0 + w, xy(0.05, 0.0)[1]), fill=(160, 160, 160), width=2)

    legend_x, legend_y = 790, 105
    for i, (model_name, color) in enumerate(colors.items()):
        y = legend_y + i * 30
        draw.line((legend_x, y + 10, legend_x + 45, y + 10), fill=color, width=4)
        draw.text((legend_x + 55, y), labels[model_name], fill=(20, 20, 20), font=small)
    draw.line((legend_x, legend_y + 100, legend_x + 45, legend_y + 100), fill=(120, 120, 120), width=2)
    draw.text((legend_x + 55, legend_y + 90), "Treat all", fill=(20, 20, 20), font=small)
    img.save(output_dir / "fig_repeat_cognition_decision_curve_hrs.png")


def save_calibration_plot(calibration: pd.DataFrame, output_dir: Path) -> None:
    img = Image.new("RGB", (1000, 760), "white")
    draw = ImageDraw.Draw(img)
    x0, y0, w, h = 110, 650, 760, 520
    draw_axes(draw, (x0, y0), (w, h), "Calibration by decile: HRS", "Mean predicted risk", "Observed risk")
    subset = calibration[calibration["dataset"].eq("HRS external validation")]
    colors = {
        "M2_repeat_assessment": (30, 105, 170),
        "M2_recalibrated": (190, 80, 45),
    }
    labels = {"M2_repeat_assessment": "M2 original", "M2_recalibrated": "M2 recalibrated"}

    def xy(value_x: float, value_y: float) -> tuple[int, int]:
        x = x0 + int(value_x / 0.85 * w)
        y = y0 - int(value_y / 0.85 * h)
        return x, y

    small = get_font(16)
    for tick in np.arange(0, 0.86, 0.2):
        x, _ = xy(float(tick), 0)
        _, y = xy(0, float(tick))
        draw.line((x, y0, x, y0 + 6), fill=(30, 30, 30), width=1)
        draw.line((x0 - 6, y, x0, y), fill=(30, 30, 30), width=1)
        draw.text((x - 12, y0 + 10), f"{tick:.1f}", fill=(30, 30, 30), font=small)
        draw.text((x0 - 55, y - 8), f"{tick:.1f}", fill=(30, 30, 30), font=small)
    draw.line((x0, y0, x0 + w, y0 - h), fill=(150, 150, 150), width=2)
    for model_name, color in colors.items():
        model_data = subset[subset["model"].eq(model_name)].sort_values("mean_predicted")
        points = [xy(row.mean_predicted, row.observed) for row in model_data.itertuples()]
        draw.line(points, fill=color, width=4)
        for point in points:
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=color)
    legend_x, legend_y = 675, 112
    for i, (model_name, color) in enumerate(colors.items()):
        y = legend_y + i * 32
        draw.line((legend_x, y + 10, legend_x + 45, y + 10), fill=color, width=4)
        draw.text((legend_x + 55, y), labels[model_name], fill=(20, 20, 20), font=small)
    img.save(output_dir / "fig_repeat_cognition_calibration_hrs.png")


def save_forest_plot(differences: pd.DataFrame, output_dir: Path) -> None:
    auroc = differences[differences["metric"].eq("auroc")].copy()
    auroc["label"] = auroc["scenario"] + " / " + auroc["dataset"].str.replace(" validation", "", regex=False)
    img = Image.new("RGB", (1300, 900), "white")
    draw = ImageDraw.Draw(img)
    title = get_font(28, bold=True)
    font = get_font(15)
    draw.text((40, 25), "M2 vs M1 AUROC difference across sensitivity analyses", fill=(20, 20, 20), font=title)
    x0, y0, w = 560, 810, 650
    x_min, x_max = -0.005, 0.06

    def x_pos(value: float) -> int:
        return x0 + int((value - x_min) / (x_max - x_min) * w)

    draw.line((x0, y0, x0 + w, y0), fill=(30, 30, 30), width=2)
    zero = x_pos(0.0)
    draw.line((zero, 85, zero, y0), fill=(180, 180, 180), width=2)
    for tick in np.arange(0, 0.061, 0.01):
        x = x_pos(float(tick))
        draw.line((x, y0, x, y0 + 6), fill=(30, 30, 30), width=1)
        draw.text((x - 16, y0 + 10), f"{tick:.2f}", fill=(30, 30, 30), font=font)
    row_gap = 46
    for i, row in enumerate(auroc.itertuples()):
        y = 105 + i * row_gap
        label = row.label[:64]
        draw.text((40, y - 10), label, fill=(20, 20, 20), font=font)
        low, high, est = x_pos(row.ci_low), x_pos(row.ci_high), x_pos(row.difference_candidate_minus_reference)
        color = (30, 105, 170) if "HRS" in row.dataset else (80, 130, 80)
        draw.line((low, y, high, y), fill=color, width=3)
        draw.ellipse((est - 6, y - 6, est + 6, y + 6), fill=color)
        draw.text((x0 + w + 18, y - 10), f"{row.difference_candidate_minus_reference:.3f}", fill=(20, 20, 20), font=font)
    img.save(output_dir / "fig_repeat_cognition_sensitivity_forest.png")


def update_master_plan() -> None:
    master = PROJECT_DIR / "STUDY_MASTER_PLAN.md"
    text = master.read_text(encoding="utf-8")
    addition = """

### 2026-07-16 Phase 5 reporting assets update

- Added `26_repeat_cognition_reporting_assets.py`.
- Created decision-curve data, HRS calibration-decile data, and preliminary PNG figures.
- Figures are reporting drafts generated from aggregate or regenerated in-memory predictions; no person-level data were saved.
- Remaining before manuscript drafting: final 1000+ bootstrap rerun, optional ELSA confirmatory validation, and final journal-style figure polishing.
"""
    if "2026-07-16 Phase 5 reporting assets update" not in text:
        marker = "## Progress Log"
        if marker in text:
            text = text.replace(marker, addition.strip() + "\n\n" + marker)
        else:
            text = text.rstrip() + "\n\n" + addition.strip() + "\n"
        master.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    charls_y, hrs_y, predictions = primary_predictions(args.data_dir, args.cv_repeats)

    decision_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for dataset, y in (
        ("CHARLS repeated 5-fold CV", charls_y),
        ("HRS external validation", hrs_y),
    ):
        for model_name in ["M1_single_assessment", "M2_repeat_assessment"]:
            probabilities = predictions[(dataset, model_name)]
            decision_rows.extend(decision_curve(y, probabilities, model_name, dataset))
            calibration_rows.extend(calibration_deciles(y, probabilities, model_name, dataset))
    decision_rows.extend(
        decision_curve(
            hrs_y,
            predictions[("HRS external validation", "M2_recalibrated")],
            "M2_recalibrated",
            "HRS external validation",
        )
    )
    calibration_rows.extend(
        calibration_deciles(
            hrs_y,
            predictions[("HRS external validation", "M2_recalibrated")],
            "M2_recalibrated",
            "HRS external validation",
        )
    )

    decision = pd.DataFrame(decision_rows)
    calibration = pd.DataFrame(calibration_rows)
    decision.to_csv(
        args.output_dir / "repeat_cognition_phase5_decision_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calibration.to_csv(
        args.output_dir / "repeat_cognition_phase5_calibration_deciles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    differences = pd.read_csv(args.output_dir / "repeat_cognition_phase2_sensitivity_differences.csv")
    save_decision_curve_plot(decision, args.output_dir)
    save_calibration_plot(calibration, args.output_dir)
    save_forest_plot(differences, args.output_dir)
    update_master_plan()
    print(decision.head(12).to_string(index=False))
    print(calibration[calibration["dataset"].eq("HRS external validation")].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
