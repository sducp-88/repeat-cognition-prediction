from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"

INK = (32, 35, 38)
MUTED = (104, 109, 115)
LIGHT = (222, 225, 228)
CHARLS = (47, 126, 80)
HRS = (27, 105, 170)
ELSA = (190, 76, 47)
RECALIBRATED = (139, 79, 155)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render high-resolution 3-cohort journal figures."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf")
        if bold
        else Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf")
        if bold
        else Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    raise FileNotFoundError("No supported Windows font found.")


def save_figure(image: Image.Image, output_dir: Path, stem: str) -> None:
    image.save(output_dir / f"{stem}.png", dpi=(300, 300), optimize=True)
    image.save(
        output_dir / f"{stem}.tiff", dpi=(300, 300), compression="tiff_lzw"
    )


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox(
        (0, 0), text, font=text_font, spacing=8, align="center"
    )
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.multiline_text(
        (
            left + (right - left - width) / 2,
            top + (bottom - top - height) / 2,
        ),
        text,
        font=text_font,
        fill=fill,
        spacing=8,
        align="center",
    )


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: tuple[int, int, int] = MUTED,
) -> None:
    draw.line((*start, *end), fill=fill, width=5)
    x, y = end
    draw.polygon([(x, y), (x - 14, y - 22), (x + 14, y - 22)], fill=fill)


def figure1_flow(output_dir: Path) -> None:
    image = Image.new("RGB", (3000, 1600), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (100, 60),
        "Figure 1. Cohort Construction for the Primary Analysis",
        font=font(58, bold=True),
        fill=INK,
    )
    draw.text(
        (100, 142),
        "Three-wave direct objective memory assessment",
        font=font(34),
        fill=MUTED,
    )

    cohorts = {
        "CHARLS": {
            "color": CHARLS,
            "x": 100,
            "stages": [
                ("Age eligible in 2011", 13466),
                ("2011 memory complete", 10632),
                ("2013 index memory complete", 8359),
                ("2015 outcome memory complete", 7264),
            ],
        },
        "HRS": {
            "color": HRS,
            "x": 1080,
            "stages": [
                ("Age eligible in 2012", 19866),
                ("2012 memory complete", 18129),
                ("2014 index memory complete", 15587),
                ("2016 outcome memory complete", 13118),
            ],
        },
        "ELSA": {
            "color": ELSA,
            "x": 2060,
            "stages": [
                ("Harmonized participants", 21679),
                ("Age >=50 and wave 7\nindex memory complete", 8214),
                ("Also wave 6 previous\nmemory complete", 8152),
                ("Also wave 8 outcome\nmemory complete", 6907),
            ],
        },
    }
    top = 310
    box_width = 820
    box_height = 190
    gap = 100
    for cohort, values in cohorts.items():
        x = int(values["x"])
        color = values["color"]
        stages = values["stages"]
        draw.text((x, 225), cohort, font=font(48, bold=True), fill=color)
        for index, (label, n) in enumerate(stages):
            y = top + index * (box_height + gap)
            box = (x, y, x + box_width, y + box_height)
            draw.rectangle(box, fill=(249, 250, 251), outline=color, width=5)
            draw.rectangle((x, y, x + 18, y + box_height), fill=color)
            centered_text(
                draw,
                (x + 28, y, x + box_width - 18, y + box_height),
                f"{label}\nn = {n:,}",
                font(34, bold=index == 3),
                INK,
            )
            if index < len(stages) - 1:
                next_n = stages[index + 1][1]
                arrow_x = x + box_width // 2
                draw_arrow(
                    draw,
                    (arrow_x, y + box_height + 8),
                    (arrow_x, y + box_height + gap - 18),
                )
                draw.text(
                    (arrow_x + 28, y + box_height + 31),
                    f"Excluded: {n - next_n:,}",
                    font=font(25),
                    fill=MUTED,
                )
    draw.text(
        (100, 1510),
        "CHARLS indicates China Health and Retirement Longitudinal Study; HRS, Health and Retirement Study; ELSA, English Longitudinal Study of Ageing.",
        font=font(25),
        fill=MUTED,
    )
    save_figure(image, output_dir, "figure1_primary_cohort_flow")


def figure2_forest(output_dir: Path) -> None:
    data = pd.read_csv(
        output_dir / "repeat_cognition_phase2_sensitivity_differences.csv"
    )
    data = data[data["metric"].eq("auroc")].copy()
    elsa = pd.read_csv(output_dir / "elsa_frozen_paired_differences.csv")
    elsa = elsa[elsa["metric"].eq("auroc")].copy()
    elsa["scenario"] = "primary_20pt_age50_0.5sd"
    elsa["dataset"] = "ELSA locked confirmation"
    data = pd.concat(
        [
            data,
            elsa[
                [
                    "scenario",
                    "dataset",
                    "metric",
                    "difference_candidate_minus_reference",
                    "ci_low",
                    "ci_high",
                ]
            ],
        ],
        ignore_index=True,
        sort=False,
    )
    scenario_labels = {
        "primary_20pt_age50_0.5sd": "Primary: 0.5-SD decline, age >=50 y",
        "age65_20pt_0.5sd": "Age >=65 y",
        "decline_20pt_1sd_age50": "1-SD decline",
        "hrs_no_dementia_ad_20pt_age50_0.5sd": "Exclude HRS dementia/Alzheimer disease",
        "score25_age50_0.5sd": "25-point cognition score",
    }
    dataset_labels = {
        "CHARLS repeated 5-fold CV": "CHARLS internal evaluation",
        "HRS external validation": "HRS external evaluation",
        "ELSA locked confirmation": "ELSA locked confirmation",
    }
    colors = {
        "CHARLS repeated 5-fold CV": CHARLS,
        "HRS external validation": HRS,
        "ELSA locked confirmation": ELSA,
    }
    data = data[data["scenario"].isin(scenario_labels)].copy()
    duplicate = (
        data["scenario"].eq("hrs_no_dementia_ad_20pt_age50_0.5sd")
        & data["dataset"].eq("CHARLS repeated 5-fold CV")
    )
    data = data[~duplicate]
    data["scenario_order"] = data["scenario"].map(
        {name: i for i, name in enumerate(scenario_labels)}
    )
    data["dataset_order"] = data["dataset"].map(
        {
            "CHARLS repeated 5-fold CV": 0,
            "HRS external validation": 1,
            "ELSA locked confirmation": 2,
        }
    )
    data = data.sort_values(["scenario_order", "dataset_order"])

    image = Image.new("RGB", (2800, 1900), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (100, 55),
        "Figure 2. Incremental Discrimination From Repeat Cognitive Assessment",
        font=font(55, bold=True),
        fill=INK,
    )
    draw.text(
        (100, 132),
        "Difference in AUROC for M2 repeat assessment versus M1 single assessment",
        font=font(32),
        fill=MUTED,
    )

    plot_left, plot_right = 1160, 2260
    plot_top, plot_bottom = 280, 1660
    x_min, x_max = -0.005, 0.060

    def x_pos(value: float) -> int:
        return plot_left + int(
            (value - x_min) / (x_max - x_min) * (plot_right - plot_left)
        )

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=INK, width=4)
    draw.line((x_pos(0.0), plot_top, x_pos(0.0), plot_bottom), fill=LIGHT, width=5)
    for tick in np.arange(0.0, 0.061, 0.01):
        x = x_pos(float(tick))
        draw.line((x, plot_bottom, x, plot_bottom + 13), fill=INK, width=3)
        label = f"{tick:.2f}"
        bounds = draw.textbbox((0, 0), label, font=font(27))
        draw.text(
            (x - (bounds[2] - bounds[0]) / 2, plot_bottom + 22),
            label,
            font=font(27),
            fill=INK,
        )
    draw.text(
        (plot_left + 260, plot_bottom + 80),
        "AUROC difference (95% CI)",
        font=font(34),
        fill=INK,
    )

    y = 310
    previous_scenario = None
    for row in data.itertuples():
        if previous_scenario is not None and row.scenario != previous_scenario:
            y += 32
        if row.scenario != previous_scenario:
            draw.text(
                (100, y - 30),
                scenario_labels[row.scenario],
                font=font(31, bold=True),
                fill=INK,
            )
        row_y = y + 38
        draw.text(
            (150, row_y - 16),
            dataset_labels[row.dataset],
            font=font(28),
            fill=MUTED,
        )
        color = colors[row.dataset]
        low = x_pos(float(row.ci_low))
        high = x_pos(float(row.ci_high))
        estimate = x_pos(float(row.difference_candidate_minus_reference))
        draw.line((low, row_y, high, row_y), fill=color, width=8)
        draw.line((low, row_y - 14, low, row_y + 14), fill=color, width=5)
        draw.line((high, row_y - 14, high, row_y + 14), fill=color, width=5)
        draw.ellipse((estimate - 12, row_y - 12, estimate + 12, row_y + 12), fill=color)
        draw.text(
            (2300, row_y - 17),
            f"{row.difference_candidate_minus_reference:.3f} "
            f"({row.ci_low:.3f} to {row.ci_high:.3f})",
            font=font(25),
            fill=INK,
        )
        y += 108
        previous_scenario = row.scenario

    legend_y = 1800
    legend_items = [(CHARLS, "CHARLS"), (HRS, "HRS"), (ELSA, "ELSA")]
    for index, (color, label) in enumerate(legend_items):
        x = 120 + index * 300
        draw.ellipse((x, legend_y, x + 24, legend_y + 24), fill=color)
        draw.text((x + 42, legend_y - 5), label, font=font(28), fill=INK)
    save_figure(image, output_dir, "figure2_sensitivity_forest")


def draw_panel_axes(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    width: int,
    height: int,
    x_max: float,
    y_min: float,
    y_max: float,
    x_label: str,
    y_label: str,
    x_ticks: list[float],
    y_ticks: list[float],
) -> tuple[callable, callable]:
    bottom = top + height
    right = left + width
    draw.line((left, top, left, bottom, right, bottom), fill=INK, width=4)

    def x_pos(value: float) -> int:
        return min(right, max(left, left + int(value / x_max * width)))

    def y_pos(value: float) -> int:
        mapped = bottom - int((value - y_min) / (y_max - y_min) * height)
        return min(bottom, max(top, mapped))

    tick_font = font(24)
    for tick in x_ticks:
        x = x_pos(tick)
        draw.line((x, bottom, x, bottom + 10), fill=INK, width=3)
        draw.text((x - 18, bottom + 14), f"{tick:.1f}", font=tick_font, fill=INK)
    for tick in y_ticks:
        y = y_pos(tick)
        draw.line((left - 10, y, left, y), fill=INK, width=3)
        draw.text((left - 65, y - 13), f"{tick:.1f}", font=tick_font, fill=INK)
    x_bounds = draw.textbbox((0, 0), x_label, font=font(29))
    draw.text(
        (left + (width - (x_bounds[2] - x_bounds[0])) / 2, bottom + 58),
        x_label,
        font=font(29),
        fill=INK,
    )
    draw.text((left, top - 44), y_label, font=font(27), fill=INK)
    return x_pos, y_pos


def draw_calibration_panel(
    draw: ImageDraw.ImageDraw,
    data: pd.DataFrame,
    left: int,
    top: int,
    panel: str,
    title: str,
    cohort_color: tuple[int, int, int],
) -> None:
    draw.text((left - 60, top - 125), panel, font=font(44, bold=True), fill=INK)
    draw.text((left, top - 118), title, font=font(34, bold=True), fill=INK)
    x_pos, y_pos = draw_panel_axes(
        draw,
        left,
        top,
        1180,
        650,
        0.85,
        0.0,
        0.85,
        "Mean predicted risk",
        "Observed risk",
        [0.0, 0.2, 0.4, 0.6, 0.8],
        [0.0, 0.2, 0.4, 0.6, 0.8],
    )
    draw.line((x_pos(0.0), y_pos(0.0), x_pos(0.85), y_pos(0.85)), fill=LIGHT, width=5)
    styles = {
        "original": (cohort_color, "M2 original transport"),
        "recalibrated": (RECALIBRATED, "M2 recalibrated"),
    }
    for key, (color, _) in styles.items():
        group = data[data["curve"].eq(key)].sort_values("mean_predicted")
        points = [
            (x_pos(float(row.mean_predicted)), y_pos(float(row.observed)))
            for row in group.itertuples()
        ]
        draw.line(points, fill=color, width=7)
        for x, y in points:
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color)
    legend_x, legend_y = left + 600, top + 28
    for index, (_, (color, label)) in enumerate(styles.items()):
        y = legend_y + index * 48
        draw.line((legend_x, y, legend_x + 65, y), fill=color, width=7)
        draw.text((legend_x + 82, y - 16), label, font=font(24), fill=INK)


def draw_decision_panel(
    draw: ImageDraw.ImageDraw,
    data: pd.DataFrame,
    left: int,
    top: int,
    panel: str,
    title: str,
    cohort_color: tuple[int, int, int],
) -> None:
    draw.text((left - 60, top - 125), panel, font=font(44, bold=True), fill=INK)
    draw.text((left, top - 118), title, font=font(34, bold=True), fill=INK)
    x_pos, y_pos = draw_panel_axes(
        draw,
        left,
        top,
        1180,
        650,
        0.50,
        -0.05,
        0.30,
        "Risk threshold",
        "Net benefit",
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        [0.0, 0.1, 0.2, 0.3],
    )
    styles = {
        "M1_single_assessment": (MUTED, "M1 single assessment"),
        "M2_repeat_assessment": (cohort_color, "M2 repeat assessment"),
    }
    for model_name, (color, _) in styles.items():
        group = data[data["model"].eq(model_name)].sort_values("threshold")
        points = [
            (x_pos(float(row.threshold)), y_pos(float(row.net_benefit)))
            for row in group.itertuples()
            if 0.10 <= float(row.threshold) <= 0.50
        ]
        draw.line(points, fill=color, width=7)
    treat_all = data[data["model"].eq("treat_all")].sort_values("threshold")
    treat_all_points = [
        (x_pos(float(row.threshold)), y_pos(float(row.net_benefit)))
        for row in treat_all.itertuples()
        if 0.10 <= float(row.threshold) <= 0.50
        and float(row.net_benefit) >= -0.05
    ]
    draw.line(treat_all_points, fill=LIGHT, width=5)
    draw.line((x_pos(0.10), y_pos(0.0), x_pos(0.50), y_pos(0.0)), fill=(160, 164, 168), width=4)
    legend_x, legend_y = left + 610, top + 28
    legend = [
        (MUTED, "M1 single assessment"),
        (cohort_color, "M2 repeat assessment"),
        (LIGHT, "Treat all"),
        ((160, 164, 168), "Treat none"),
    ]
    for index, (color, label) in enumerate(legend):
        y = legend_y + index * 44
        draw.line((legend_x, y, legend_x + 65, y), fill=color, width=6)
        draw.text((legend_x + 82, y - 15), label, font=font(23), fill=INK)


def figure3_validation(output_dir: Path) -> None:
    hrs_calibration = pd.read_csv(
        output_dir / "repeat_cognition_phase5_calibration_deciles.csv"
    )
    hrs_calibration = hrs_calibration[
        hrs_calibration["dataset"].eq("HRS external validation")
        & hrs_calibration["model"].isin(["M2_repeat_assessment", "M2_recalibrated"])
    ].copy()
    hrs_calibration["curve"] = hrs_calibration["model"].map(
        {
            "M2_repeat_assessment": "original",
            "M2_recalibrated": "recalibrated",
        }
    )

    elsa_calibration = pd.read_csv(
        output_dir / "elsa_frozen_calibration_curve.csv"
    )
    elsa_calibration = elsa_calibration[
        elsa_calibration["model"].eq("M2_repeat_assessment")
    ].copy()
    elsa_calibration["curve"] = elsa_calibration["calibration"].map(
        {
            "none_original_transport": "original",
            "cv_intercept_slope": "recalibrated",
        }
    )
    elsa_calibration = elsa_calibration.rename(
        columns={"observed_proportion": "observed"}
    )

    hrs_decision_raw = pd.read_csv(
        output_dir / "repeat_cognition_formal_hrs_decision_curve.csv"
    )
    hrs_models = hrs_decision_raw[
        hrs_decision_raw["calibration"].eq("cv_intercept_slope")
    ][["model", "threshold", "net_benefit"]].copy()
    hrs_reference = hrs_decision_raw[
        hrs_decision_raw["model"].eq("M2_repeat_assessment")
        & hrs_decision_raw["calibration"].eq("cv_intercept_slope")
    ][["threshold", "treat_all_net_benefit"]].copy()
    hrs_reference["model"] = "treat_all"
    hrs_reference = hrs_reference.rename(
        columns={"treat_all_net_benefit": "net_benefit"}
    )
    hrs_decision = pd.concat([hrs_models, hrs_reference], ignore_index=True)

    elsa_decision_raw = pd.read_csv(output_dir / "elsa_frozen_decision_curve.csv")
    elsa_models = elsa_decision_raw[
        elsa_decision_raw["calibration"].eq("cv_intercept_slope")
        & elsa_decision_raw["model"].isin(
            ["M1_single_assessment", "M2_repeat_assessment"]
        )
    ][["model", "threshold", "net_benefit"]].copy()
    elsa_reference = elsa_decision_raw[
        elsa_decision_raw["model"].eq("treat_all")
    ][["model", "threshold", "net_benefit"]].copy()
    elsa_decision = pd.concat([elsa_models, elsa_reference], ignore_index=True)

    image = Image.new("RGB", (3200, 2350), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (100, 55),
        "Figure 3. Calibration and Clinical Utility in External Cohorts",
        font=font(56, bold=True),
        fill=INK,
    )
    draw.text(
        (100, 135),
        "Original CHARLS transport and 5-fold cross-validated local recalibration",
        font=font(32),
        fill=MUTED,
    )

    draw_calibration_panel(
        draw, hrs_calibration, 190, 330, "A", "HRS calibration of M2", HRS
    )
    draw_calibration_panel(
        draw, elsa_calibration, 1770, 330, "B", "ELSA calibration of M2", ELSA
    )
    draw_decision_panel(
        draw, hrs_decision, 190, 1340, "C", "HRS decision curve", HRS
    )
    draw_decision_panel(
        draw, elsa_decision, 1770, 1340, "D", "ELSA decision curve", ELSA
    )
    draw.text(
        (100, 2275),
        "Decision curves compare 5-fold out-of-fold recalibrated M1 and M2 predictions; original transport remains the primary external-performance analysis.",
        font=font(25),
        fill=MUTED,
    )
    save_figure(
        image,
        output_dir,
        "figure3_external_validation_calibration_decision_curve",
    )


def write_legends(output_dir: Path) -> None:
    text = """# Figure Legends

## Figure 1. Cohort Construction for the Primary Analysis

Participants were required to be aged 50 years or older at the first assessment and to have complete direct immediate and delayed recall scores at the previous, index, and outcome assessments. HRS proxy interviews and RAND-imputed recall components were excluded. ELSA direct-interview status and recall components were verified against wave-specific raw files. CHARLS indicates China Health and Retirement Longitudinal Study; HRS, Health and Retirement Study; ELSA, English Longitudinal Study of Ageing.

## Figure 2. Incremental Discrimination From Repeat Cognitive Assessment

Points show the difference in area under the receiver operating characteristic curve (AUROC) between the repeat-assessment model (M2) and the single-assessment model (M1); error bars show participant-level paired bootstrap 95% CIs based on 1000 replicates. Circles, squares, and diamonds indicate CHARLS, HRS, and ELSA, respectively. M1 included age, sex, education, and index memory score. M2 additionally included the previous memory score. ELSA was analyzed only after the CHARLS model and reporting code were frozen. Sensitivity analyses were prespecified in CHARLS and HRS; the HRS dementia/Alzheimer disease exclusion applies only to HRS. CHARLS indicates China Health and Retirement Longitudinal Study; HRS, Health and Retirement Study; ELSA, English Longitudinal Study of Ageing.

## Figure 3. Calibration and Clinical Utility in External Cohorts

Panels A and B show observed versus mean predicted event risk by tenths of predicted risk for the original CHARLS-trained M2 model and after 5-fold cross-validated intercept-plus-slope recalibration in HRS and ELSA, respectively. Original transport is shown with solid lines and circles; recalibrated estimates, with dashed lines and squares. Panels C and D show decision-curve net benefit for locally recalibrated M1 and M2 predictions evaluated out of fold; dashed lines indicate M1, solid cohort-colored lines indicate M2, dotted lines indicate treat all, and dash-dot lines indicate treat none. Original transport performance was the primary external analysis; recalibration was a secondary implementation analysis. The outcome was a decline of at least 0.5 SD in the cohort-specific 20-point immediate-plus-delayed recall score from the index to outcome assessment. HRS indicates Health and Retirement Study; ELSA, English Longitudinal Study of Ageing; M1, single-assessment model; M2, repeat-assessment model.
"""
    (output_dir / "FIGURE_LEGENDS.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure1_flow(args.output_dir)
    figure2_forest(args.output_dir)
    figure3_validation(args.output_dir)
    write_legends(args.output_dir)
    print("Rendered 3 three-cohort journal figures in PNG and TIFF formats.")


if __name__ == "__main__":
    main()
