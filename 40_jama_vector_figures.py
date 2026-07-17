from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import (
    Circle,
    Drawing,
    Line,
    Polygon,
    PolyLine,
    Rect,
    String,
)
from reportlab.lib.colors import Color, HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "outputs"
DEFAULT_OUTPUT = DEFAULT_INPUT / "jama_vector_figures"

INK = HexColor("#202326")
MUTED = HexColor("#656B73")
LIGHT = HexColor("#D7DCE1")
PALE = HexColor("#F7F8F9")
CHARLS = HexColor("#2F7E50")
HRS = HexColor("#1B69AA")
ELSA = HexColor("#BE4C2F")
RECALIBRATED = HexColor("#8B4F9B")
GRAY = HexColor("#666B70")
ZERO = HexColor("#9DA2A6")

FONT = "Helvetica"
BOLD = "Helvetica-Bold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build JAMA-style vector figures and aggregate source tables."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def register_fonts() -> None:
    global FONT, BOLD
    font_pairs = [
        (
            "Arial",
            Path(r"C:\Windows\Fonts\arial.ttf"),
            "Arial-Bold",
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
        ),
        (
            "LiberationSans",
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
            "LiberationSans-Bold",
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
        ),
    ]
    for regular_name, regular_path, bold_name, bold_path in font_pairs:
        if regular_path.exists() and bold_path.exists():
            pdfmetrics.registerFont(TTFont(regular_name, str(regular_path)))
            pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))
            FONT, BOLD = regular_name, bold_name
            return
    # ReportLab's built-in Helvetica pair keeps the script portable when no
    # compatible TrueType sans-serif font is installed.
    FONT, BOLD = "Helvetica", "Helvetica-Bold"


def text(
    drawing: Drawing,
    x: float,
    y: float,
    value: str,
    size: float = 8,
    *,
    bold: bool = False,
    color: Color = INK,
    anchor: str = "start",
) -> None:
    drawing.add(
        String(
            x,
            y,
            value,
            fontName=BOLD if bold else FONT,
            fontSize=size,
            fillColor=color,
            textAnchor=anchor,
        )
    )


def multiline_center(
    drawing: Drawing,
    x: float,
    center_y: float,
    lines: list[str],
    size: float = 8,
    *,
    bold: bool = False,
    color: Color = INK,
    leading: float | None = None,
) -> None:
    leading = leading or size * 1.25
    baseline = center_y + (len(lines) - 1) * leading / 2 - size * 0.35
    for index, line in enumerate(lines):
        text(
            drawing,
            x,
            baseline - index * leading,
            line,
            size,
            bold=bold,
            color=color,
            anchor="middle",
        )


def dashed_polyline(
    drawing: Drawing,
    points: list[tuple[float, float]],
    color: Color,
    width: float,
    dash: list[float] | None = None,
) -> None:
    line = PolyLine(points, strokeColor=color, strokeWidth=width, fillColor=None)
    if dash:
        line.strokeDashArray = dash
    drawing.add(line)


def marker(
    drawing: Drawing,
    x: float,
    y: float,
    shape: str,
    color: Color,
    radius: float = 3.2,
) -> None:
    if shape == "circle":
        drawing.add(Circle(x, y, radius, fillColor=color, strokeColor=color))
    elif shape == "square":
        drawing.add(
            Rect(
                x - radius,
                y - radius,
                radius * 2,
                radius * 2,
                fillColor=color,
                strokeColor=color,
            )
        )
    elif shape == "diamond":
        drawing.add(
            Polygon(
                [x, y + radius * 1.2, x + radius * 1.2, y, x, y - radius * 1.2, x - radius * 1.2, y],
                fillColor=color,
                strokeColor=color,
            )
        )
    else:
        raise ValueError(f"Unsupported marker: {shape}")


def save_vector(drawing: Drawing, output_dir: Path, stem: str) -> None:
    renderPDF.drawToFile(drawing, str(output_dir / f"{stem}.pdf"))
    renderSVG.drawToFile(drawing, str(output_dir / f"{stem}.svg"))


def build_figure1(output_dir: Path) -> pd.DataFrame:
    drawing = Drawing(518.4, 500)
    cohorts = [
        (
            "CHARLS",
            CHARLS,
            [
                ("Age eligible in 2011", 13466),
                ("2011 memory complete", 10632),
                ("2013 index memory complete", 8359),
                ("2015 outcome memory complete", 7264),
            ],
        ),
        (
            "HRS",
            HRS,
            [
                ("Age eligible in 2012", 19866),
                ("2012 memory complete", 18129),
                ("2014 index memory complete", 15587),
                ("2016 outcome memory complete", 13118),
            ],
        ),
        (
            "ELSA",
            ELSA,
            [
                ("Harmonized participants", 21679),
                ("Age ≥50 y and wave 7|index memory complete", 8214),
                ("Also wave 6 previous|memory complete", 8152),
                ("Also wave 8 outcome|memory complete", 6907),
            ],
        ),
    ]
    lefts = [8, 180, 352]
    box_width = 158
    box_height = 70
    box_bottoms = [365, 250, 135, 20]
    source_rows: list[dict[str, object]] = []

    for cohort_index, (cohort, color, stages) in enumerate(cohorts):
        left = lefts[cohort_index]
        text(drawing, left, 480, cohort, 11, bold=True, color=color)
        for index, (label, n_value) in enumerate(stages):
            bottom = box_bottoms[index]
            drawing.add(
                Rect(
                    left,
                    bottom,
                    box_width,
                    box_height,
                    fillColor=PALE,
                    strokeColor=color,
                    strokeWidth=1.2,
                )
            )
            drawing.add(
                Rect(
                    left,
                    bottom,
                    4,
                    box_height,
                    fillColor=color,
                    strokeColor=color,
                )
            )
            lines = label.split("|") + [f"n = {n_value:,}"]
            multiline_center(
                drawing,
                left + box_width / 2 + 2,
                bottom + box_height / 2,
                lines,
                8.2,
                bold=index == len(stages) - 1,
            )
            excluded = None
            if index < len(stages) - 1:
                excluded = n_value - stages[index + 1][1]
                center_x = left + box_width / 2
                drawing.add(
                    Line(
                        center_x,
                        bottom - 7,
                        center_x,
                        bottom - 34,
                        strokeColor=MUTED,
                        strokeWidth=1.2,
                    )
                )
                drawing.add(
                    Polygon(
                        [
                            center_x,
                            bottom - 39,
                            center_x - 3.5,
                            bottom - 32,
                            center_x + 3.5,
                            bottom - 32,
                        ],
                        fillColor=MUTED,
                        strokeColor=MUTED,
                    )
                )
                text(
                    drawing,
                    center_x + 7,
                    bottom - 25,
                    f"Excluded: {excluded:,}",
                    6.8,
                    color=MUTED,
                )
            source_rows.append(
                {
                    "cohort": cohort,
                    "stage_order": index + 1,
                    "stage_label": label.replace("|", " "),
                    "n": n_value,
                    "excluded_after_stage": excluded,
                }
            )

    save_vector(drawing, output_dir, "Figure_1")
    return pd.DataFrame(source_rows)


def load_figure2(input_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(input_dir / "repeat_cognition_phase2_sensitivity_differences.csv")
    data = data[data["metric"].eq("auroc")].copy()
    elsa = pd.read_csv(input_dir / "elsa_frozen_paired_differences.csv")
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
    )
    scenario_labels = {
        "primary_20pt_age50_0.5sd": "Primary: 0.5-SD decline, age ≥50 y",
        "age65_20pt_0.5sd": "Age ≥65 y",
        "decline_20pt_1sd_age50": "1-SD decline",
        "hrs_no_dementia_ad_20pt_age50_0.5sd": "Exclusion of reported dementia or Alzheimer disease",
        "score25_age50_0.5sd": "25-point cognition score",
    }
    dataset_labels = {
        "CHARLS repeated 5-fold CV": "CHARLS internal evaluation",
        "HRS external validation": "HRS external evaluation",
        "ELSA locked confirmation": "ELSA locked confirmation",
    }
    markers = {
        "CHARLS repeated 5-fold CV": "circle",
        "HRS external validation": "square",
        "ELSA locked confirmation": "diamond",
    }
    colors = {
        "CHARLS repeated 5-fold CV": "#2F7E50",
        "HRS external validation": "#1B69AA",
        "ELSA locked confirmation": "#BE4C2F",
    }
    data = data[data["scenario"].isin(scenario_labels)].copy()
    duplicate = data["scenario"].eq("hrs_no_dementia_ad_20pt_age50_0.5sd") & data[
        "dataset"
    ].eq("CHARLS repeated 5-fold CV")
    data = data[~duplicate].copy()
    data["scenario_order"] = data["scenario"].map(
        {name: index for index, name in enumerate(scenario_labels)}
    )
    data["dataset_order"] = data["dataset"].map(
        {
            "CHARLS repeated 5-fold CV": 0,
            "HRS external validation": 1,
            "ELSA locked confirmation": 2,
        }
    )
    data["scenario_label"] = data["scenario"].map(scenario_labels)
    data["dataset_label"] = data["dataset"].map(dataset_labels)
    data["marker"] = data["dataset"].map(markers)
    data["color_hex"] = data["dataset"].map(colors)
    return data.sort_values(["scenario_order", "dataset_order"]).reset_index(drop=True)


def build_figure2(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    data = load_figure2(input_dir)
    drawing = Drawing(518.4, 480)
    plot_left, plot_right = 232, 386
    plot_bottom, plot_top = 48, 438
    x_min, x_max = -0.005, 0.060

    def x_pos(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

    drawing.add(
        Line(plot_left, plot_bottom, plot_right, plot_bottom, strokeColor=INK, strokeWidth=1)
    )
    drawing.add(
        Line(x_pos(0), plot_bottom, x_pos(0), plot_top, strokeColor=LIGHT, strokeWidth=1.2)
    )
    for tick in np.arange(0.0, 0.061, 0.01):
        x = x_pos(float(tick))
        drawing.add(Line(x, plot_bottom, x, plot_bottom - 4, strokeColor=INK, strokeWidth=0.8))
        text(drawing, x, plot_bottom - 14, f"{tick:.2f}", 6.8, anchor="middle")
    text(
        drawing,
        (plot_left + plot_right) / 2,
        12,
        "AUROC difference (95% CI)",
        8,
        anchor="middle",
    )

    legend = [
        ("circle", CHARLS, "CHARLS"),
        ("square", HRS, "HRS"),
        ("diamond", ELSA, "ELSA"),
    ]
    for index, (shape, color, label) in enumerate(legend):
        x = 275 + index * 72
        marker(drawing, x, 463, shape, color, radius=2.8)
        text(drawing, x + 7, 460.5, label, 7.2)

    current_scenario = None
    row_y = 430
    for row in data.itertuples():
        if row.scenario != current_scenario:
            if current_scenario is not None:
                row_y -= 4
            text(drawing, 10, row_y, row.scenario_label, 8.2, bold=True)
            row_y -= 16
            current_scenario = row.scenario
        text(drawing, 24, row_y - 2.4, row.dataset_label, 7.2, color=MUTED)
        color = HexColor(row.color_hex)
        low = x_pos(float(row.ci_low))
        high = x_pos(float(row.ci_high))
        estimate = x_pos(float(row.difference_candidate_minus_reference))
        drawing.add(Line(low, row_y, high, row_y, strokeColor=color, strokeWidth=2.2))
        drawing.add(Line(low, row_y - 3, low, row_y + 3, strokeColor=color, strokeWidth=1.2))
        drawing.add(Line(high, row_y - 3, high, row_y + 3, strokeColor=color, strokeWidth=1.2))
        marker(drawing, estimate, row_y, row.marker, color, radius=3)
        text(
            drawing,
            405,
            row_y - 2.6,
            f"{row.difference_candidate_minus_reference:.3f} "
            f"({row.ci_low:.3f} to {row.ci_high:.3f})",
            6.9,
        )
        row_y -= 28

    save_vector(drawing, output_dir, "Figure_2")
    return data[
        [
            "scenario_order",
            "scenario",
            "scenario_label",
            "dataset_order",
            "dataset",
            "dataset_label",
            "difference_candidate_minus_reference",
            "ci_low",
            "ci_high",
            "marker",
            "color_hex",
        ]
    ]


def axes(
    drawing: Drawing,
    left: float,
    bottom: float,
    width: float,
    height: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_ticks: list[float],
    y_ticks: list[float],
    x_label: str,
    y_label: str,
) -> tuple[callable, callable]:
    def xp(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * width

    def yp(value: float) -> float:
        return bottom + (value - y_min) / (y_max - y_min) * height

    drawing.add(Line(left, bottom, left, bottom + height, strokeColor=INK, strokeWidth=1))
    drawing.add(Line(left, bottom, left + width, bottom, strokeColor=INK, strokeWidth=1))
    for tick in x_ticks:
        x = xp(tick)
        drawing.add(Line(x, bottom, x, bottom - 3, strokeColor=INK, strokeWidth=0.7))
        text(drawing, x, bottom - 11, f"{tick:.1f}", 6.5, anchor="middle")
    for tick in y_ticks:
        y = yp(tick)
        drawing.add(Line(left - 3, y, left, y, strokeColor=INK, strokeWidth=0.7))
        tick_label = f"{tick:.2f}" if abs(tick) == 0.05 else f"{tick:.1f}"
        text(drawing, left - 6, y - 2.2, tick_label, 7, anchor="end")
    text(drawing, left + width / 2, bottom - 24, x_label, 7.5, anchor="middle")
    text(drawing, left, bottom + height + 7, y_label, 7.2)
    return xp, yp


def panel_title(
    drawing: Drawing, left: float, bottom: float, height: float, panel: str, title_value: str
) -> None:
    text(drawing, left - 26, bottom + height + 32, panel, 12, bold=True)
    text(drawing, left, bottom + height + 32, title_value, 9, bold=True)


def build_calibration_panel(
    drawing: Drawing,
    data: pd.DataFrame,
    left: float,
    bottom: float,
    panel: str,
    title_value: str,
    cohort_color: Color,
) -> None:
    width, height = 190, 155
    panel_title(drawing, left, bottom, height, panel, title_value)
    xp, yp = axes(
        drawing,
        left,
        bottom,
        width,
        height,
        0,
        0.85,
        0,
        0.85,
        [0, 0.2, 0.4, 0.6, 0.8],
        [0, 0.2, 0.4, 0.6, 0.8],
        "Mean predicted risk",
        "Observed risk",
    )
    dashed_polyline(
        drawing,
        [(xp(0), yp(0)), (xp(0.85), yp(0.85))],
        LIGHT,
        1.1,
        [3, 2],
    )
    styles = {
        "original": (cohort_color, "circle", None, "M2 original transport"),
        "recalibrated": (RECALIBRATED, "square", [4, 2], "M2 recalibrated"),
    }
    for curve, (color, shape, dash, _) in styles.items():
        group = data[data["curve"].eq(curve)].sort_values("mean_predicted")
        points = [
            (xp(float(row.mean_predicted)), yp(float(row.observed)))
            for row in group.itertuples()
        ]
        dashed_polyline(drawing, points, color, 1.7, dash)
        for x, y in points:
            marker(drawing, x, y, shape, color, radius=2.3)
    legend_x, legend_y = left + 95, bottom + height - 17
    for index, (_, (color, shape, dash, label)) in enumerate(styles.items()):
        y = legend_y - index * 16
        dashed_polyline(drawing, [(legend_x, y), (legend_x + 20, y)], color, 1.5, dash)
        marker(drawing, legend_x + 10, y, shape, color, radius=1.8)
        text(drawing, legend_x + 25, y - 2.4, label, 7)


def build_decision_panel(
    drawing: Drawing,
    data: pd.DataFrame,
    left: float,
    bottom: float,
    panel: str,
    title_value: str,
    cohort_color: Color,
) -> None:
    width, height = 190, 155
    panel_title(drawing, left, bottom, height, panel, title_value)
    xp, yp = axes(
        drawing,
        left,
        bottom,
        width,
        height,
        0.10,
        0.50,
        -0.05,
        0.30,
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [-0.05, 0.0, 0.1, 0.2, 0.3],
        "Risk threshold",
        "Net benefit",
    )
    styles = {
        "M1_single_assessment": (GRAY, [5, 2], "M1 single assessment"),
        "M2_repeat_assessment": (cohort_color, None, "M2 repeat assessment"),
        "treat_all": (LIGHT, [2, 2], "Treat all"),
        "treat_none": (ZERO, [7, 2, 1, 2], "Treat none"),
    }
    for model_name, (color, dash, _) in styles.items():
        if model_name == "treat_none":
            points = [(xp(0.10), yp(0)), (xp(0.50), yp(0))]
        else:
            group = data[data["model"].eq(model_name)].sort_values("threshold")
            points = [
                (xp(float(row.threshold)), yp(float(row.net_benefit)))
                for row in group.itertuples()
                if 0.10 <= float(row.threshold) <= 0.50
                and -0.05 <= float(row.net_benefit) <= 0.30
            ]
        dashed_polyline(drawing, points, color, 1.5, dash)
    legend_x, legend_y = left + 95, bottom + height - 17
    for index, (_, (color, dash, label)) in enumerate(styles.items()):
        y = legend_y - index * 15
        dashed_polyline(drawing, [(legend_x, y), (legend_x + 20, y)], color, 1.5, dash)
        text(drawing, legend_x + 25, y - 2.4, label, 7)


def load_figure3(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    hrs_calibration = pd.read_csv(input_dir / "repeat_cognition_phase5_calibration_deciles.csv")
    hrs_calibration = hrs_calibration[
        hrs_calibration["dataset"].eq("HRS external validation")
        & hrs_calibration["model"].isin(["M2_repeat_assessment", "M2_recalibrated"])
    ].copy()
    hrs_calibration["cohort"] = "HRS"
    hrs_calibration["curve"] = hrs_calibration["model"].map(
        {"M2_repeat_assessment": "original", "M2_recalibrated": "recalibrated"}
    )

    elsa_calibration = pd.read_csv(input_dir / "elsa_frozen_calibration_curve.csv")
    elsa_calibration = elsa_calibration[
        elsa_calibration["model"].eq("M2_repeat_assessment")
    ].copy()
    elsa_calibration["cohort"] = "ELSA"
    elsa_calibration["curve"] = elsa_calibration["calibration"].map(
        {"none_original_transport": "original", "cv_intercept_slope": "recalibrated"}
    )
    elsa_calibration = elsa_calibration.rename(columns={"observed_proportion": "observed"})
    calibration = pd.concat(
        [
            hrs_calibration[["cohort", "curve", "mean_predicted", "observed"]],
            elsa_calibration[["cohort", "curve", "mean_predicted", "observed"]],
        ],
        ignore_index=True,
    )
    calibration["line_style"] = calibration["curve"].map(
        {"original": "solid", "recalibrated": "dashed"}
    )
    calibration["marker"] = calibration["curve"].map(
        {"original": "circle", "recalibrated": "square"}
    )

    hrs_raw = pd.read_csv(input_dir / "repeat_cognition_formal_hrs_decision_curve.csv")
    hrs_models = hrs_raw[
        hrs_raw["calibration"].eq("cv_intercept_slope")
    ][["model", "threshold", "net_benefit"]].copy()
    hrs_reference = hrs_raw[
        hrs_raw["model"].eq("M2_repeat_assessment")
        & hrs_raw["calibration"].eq("cv_intercept_slope")
    ][["threshold", "treat_all_net_benefit"]].copy()
    hrs_reference["model"] = "treat_all"
    hrs_reference = hrs_reference.rename(columns={"treat_all_net_benefit": "net_benefit"})
    hrs_decision = pd.concat([hrs_models, hrs_reference], ignore_index=True)
    hrs_decision["cohort"] = "HRS"

    elsa_raw = pd.read_csv(input_dir / "elsa_frozen_decision_curve.csv")
    elsa_models = elsa_raw[
        elsa_raw["calibration"].eq("cv_intercept_slope")
        & elsa_raw["model"].isin(["M1_single_assessment", "M2_repeat_assessment"])
    ][["model", "threshold", "net_benefit"]].copy()
    elsa_reference = elsa_raw[elsa_raw["model"].eq("treat_all")][
        ["model", "threshold", "net_benefit"]
    ].copy()
    elsa_decision = pd.concat([elsa_models, elsa_reference], ignore_index=True)
    elsa_decision["cohort"] = "ELSA"
    decision = pd.concat([hrs_decision, elsa_decision], ignore_index=True)
    decision["line_style"] = decision["model"].map(
        {
            "M1_single_assessment": "dashed",
            "M2_repeat_assessment": "solid",
            "treat_all": "dotted",
        }
    )
    return calibration, decision


def build_figure3(input_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration, decision = load_figure3(input_dir)
    drawing = Drawing(518.4, 520)
    build_calibration_panel(
        drawing,
        calibration[calibration["cohort"].eq("HRS")],
        45,
        300,
        "A",
        "HRS calibration of M2",
        HRS,
    )
    build_calibration_panel(
        drawing,
        calibration[calibration["cohort"].eq("ELSA")],
        305,
        300,
        "B",
        "ELSA calibration of M2",
        ELSA,
    )
    build_decision_panel(
        drawing,
        decision[decision["cohort"].eq("HRS")],
        45,
        52,
        "C",
        "HRS decision curve",
        HRS,
    )
    build_decision_panel(
        drawing,
        decision[decision["cohort"].eq("ELSA")],
        305,
        52,
        "D",
        "ELSA decision curve",
        ELSA,
    )
    save_vector(drawing, output_dir, "Figure_3")
    plotted_decision = decision[
        decision["threshold"].between(0.10, 0.50)
        & (~decision["model"].eq("treat_all") | decision["net_benefit"].ge(-0.05))
    ].copy()
    return calibration, plotted_decision


def write_readme(output_dir: Path) -> None:
    text_value = """# Editable Figure Sources

The PDF files are the preferred JAMA Network Open submission files. They preserve text, lines, markers, and plotted paths as vector objects. The SVG files are editable in Inkscape, Adobe Illustrator, Affinity Designer, or recent versions of Microsoft PowerPoint after conversion to shapes.

`Figure_Source_Data.xlsx` and the CSV files under `data` contain only the aggregate values displayed in the figures. They do not contain individual-level CHARLS, HRS, or ELSA data.

Regenerate all vector figures by running:

`python 40_jama_vector_figures.py`

Figure 1 uses flow counts recorded in the script and exported to `data/Figure_1_flow.csv`. Figure 2 reads the prespecified sensitivity and ELSA paired-difference aggregate CSV files. Figure 3 reads the HRS and ELSA aggregate calibration and decision-curve CSV files. Figure titles and full legends are intentionally kept in the manuscript rather than embedded in the artwork.
"""
    (output_dir / "README.md").write_text(text_value, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    register_fonts()
    figure1 = build_figure1(args.output_dir)
    figure2 = build_figure2(args.input_dir, args.output_dir)
    calibration, decision = build_figure3(args.input_dir, args.output_dir)
    figure1.to_csv(data_dir / "Figure_1_flow.csv", index=False)
    figure2.to_csv(data_dir / "Figure_2_forest.csv", index=False)
    calibration.to_csv(data_dir / "Figure_3_calibration.csv", index=False)
    decision.to_csv(data_dir / "Figure_3_decision_curve.csv", index=False)
    write_readme(args.output_dir)
    print(f"Created vector figures and aggregate source tables in {args.output_dir}")


if __name__ == "__main__":
    main()
