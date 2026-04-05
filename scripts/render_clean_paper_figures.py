#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


CATEGORY_ORDER = [
    "circuit_evolution",
    "measurement_prediction",
    "operator_composition",
    "entanglement_classification",
]

CATEGORY_LABELS = {
    "circuit_evolution": "Circuit evolution",
    "measurement_prediction": "Measurement prediction",
    "operator_composition": "Operator composition",
    "entanglement_classification": "Entanglement",
}

MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o mini",
    "gpt-4.1-mini": "GPT-4.1 mini",
    "gpt-4.1": "GPT-4.1",
    "gpt-5.1": "GPT-5.1",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.4-mini": "GPT-5.4 mini",
    "gpt-5.4": "GPT-5.4",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-opus-4-1-20250805": "Claude Opus 4.1",
}

PALETTE = {
    "bg": "#fcfcfb",
    "panel": "#f4f1eb",
    "card": "#fbfaf7",
    "text": "#18212b",
    "muted": "#5f6b76",
    "grid": "#d9d3ca",
    "blue": "#2f6c8f",
    "blue_soft": "#d9e7ee",
    "coral": "#c96f4a",
    "coral_soft": "#f2ddd1",
    "teal": "#5f8d80",
    "teal_soft": "#dce9e3",
    "navy": "#33425a",
    "sand": "#e8ddcf",
}


def fonts():
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font_bold = "/System/Library/Fonts/Helvetica.ttc"
    return {
        "title": ImageFont.truetype(font_bold, 34),
        "panel": ImageFont.truetype(font_bold, 24),
        "label": ImageFont.truetype(font_path, 20),
        "small": ImageFont.truetype(font_path, 17),
        "num": ImageFont.truetype(font_bold, 18),
        "badge": ImageFont.truetype(font_bold, 14),
    }


def draw_badge(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fill: str, font: ImageFont.FreeTypeFont) -> None:
    tw, th = fit_text(draw, text, font)
    pad_x, pad_y = 12, 7
    draw.rounded_rectangle(
        (x, y, x + tw + pad_x * 2, y + th + pad_y * 2),
        radius=16,
        fill=fill,
        outline=None,
    )
    draw.text((x + pad_x, y + pad_y - 1), text, fill="white", font=font)


def draw_panel_box(draw: ImageDraw.ImageDraw, bounds: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(bounds, radius=28, fill=PALETTE["card"], outline=PALETTE["grid"], width=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render clean paper figures from benchmark result JSON files.")
    parser.add_argument("--results-2q", required=True)
    parser.add_argument("--results-4q", required=True)
    parser.add_argument("--results-8q")
    parser.add_argument("--takeaways-out", required=True)
    parser.add_argument("--breakdown-2q-out", required=True)
    parser.add_argument("--breakdown-4q-out", required=True)
    parser.add_argument("--breakdown-8q-out")
    return parser.parse_args()


def load_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return [row for row in rows if row.get("api_error") is None and row.get("d_phys") is not None]


def mean(values: Iterable[float]) -> float:
    seq = list(values)
    return sum(seq) / len(seq) if seq else 0.0


def group_mean(rows: list[dict], key: str) -> list[tuple[str, float]]:
    keys = sorted({str(row[key]) for row in rows}, key=str)
    return [(name, mean(float(row["d_phys"]) for row in rows if str(row[key]) == name)) for name in keys]


def group_mean_int(rows: list[dict], key: str) -> list[tuple[int, float]]:
    keys = sorted({int(row[key]) for row in rows})
    return [(name, mean(float(row["d_phys"]) for row in rows if int(row[key]) == name)) for name in keys]


def model_category_matrix(rows: list[dict]) -> tuple[list[str], list[str], list[list[float]]]:
    models = sorted(
        {str(row["model"]) for row in rows},
        key=lambda model: mean(float(row["d_phys"]) for row in rows if str(row["model"]) == model),
    )
    cats = list(CATEGORY_ORDER)
    matrix = []
    for model in models:
        matrix.append(
            [
                mean(
                    float(row["d_phys"])
                    for row in rows
                    if str(row["model"]) == model and str(row["category"]) == category
                )
                for category in cats
            ]
        )
    return models, cats, matrix


def category_means(rows: list[dict]) -> list[tuple[str, float]]:
    ordered = [
        (category, mean(float(row["d_phys"]) for row in rows if str(row["category"]) == category))
        for category in CATEGORY_ORDER
    ]
    return sorted(ordered, key=lambda item: item[1], reverse=True)


def fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def render_takeaways(rows: list[dict], path: str) -> None:
    W, H = 1540, 1200
    ft = fonts()
    bg = PALETTE["bg"]
    text = PALETTE["text"]
    muted = PALETTE["muted"]
    blue = PALETTE["blue"]
    rose = PALETTE["coral"]
    teal = PALETTE["teal"]
    navy = PALETTE["navy"]
    orange = PALETTE["coral"]

    models = sorted(group_mean(rows, "model"), key=lambda item: item[1])
    depths = group_mean_int(rows, "depth")
    categories = category_means(rows)

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    draw.text((70, 42), "Direct takeaways from the pooled 2-qubit case study", fill=text, font=ft["title"])
    draw.text(
        (70, 90),
        "These panels summarize the three strongest benchmark-level findings: capability matters, prompted depth barely moves the mean, and exact quantum dynamics remain the hardest task family.",
        fill=muted,
        font=ft["small"],
    )

    panels = [
        (60, 150, 1480, 455),
        (60, 495, 1480, 805),
        (60, 845, 1480, 1145),
    ]
    for bounds in panels:
        draw_panel_box(draw, bounds)

    # Panel A: model ranking
    x0, y0, x1, y1 = panels[0]
    draw_badge(draw, x0 + 28, y0 + 22, "A", blue, ft["badge"])
    draw.text((x0 + 82, y0 + 20), "Stronger models reduce physics-grounded error", fill=text, font=ft["panel"])
    draw.text((x0 + 82, y0 + 56), "Rows are sorted best-to-worst, so the ranking can be read immediately.", fill=muted, font=ft["small"])
    plot_left, plot_right = x0 + 330, x1 - 95
    plot_top, plot_bottom = y0 + 116, y1 - 36
    max_x = max(v for _, v in models) + 0.08
    rows_y = (plot_bottom - plot_top) / len(models)
    for tick in [0.0, 0.2, 0.4, 0.6]:
        px = plot_left + (plot_right - plot_left) * (tick / max_x)
        draw.line((px, plot_top, px, plot_bottom), fill=PALETTE["grid"], width=2)
        label = f"{tick:.1f}"
        tw, _ = fit_text(draw, label, ft["small"])
        draw.text((px - tw / 2, plot_bottom + 8), label, fill=muted, font=ft["small"])
    for idx, (model, value) in enumerate(models):
        cy = plot_top + rows_y * (idx + 0.5)
        label = MODEL_LABELS.get(model, model)
        _, th = fit_text(draw, label, ft["label"])
        draw.text((x0 + 28, cy - th / 2), label, fill=text, font=ft["label"])
        px = plot_left + (plot_right - plot_left) * (value / max_x)
        draw.line((plot_left, cy, px, cy), fill=PALETTE["sand"], width=8)
        draw.ellipse((px - 11, cy - 11, px + 11, cy + 11), fill=blue, outline=PALETTE["card"], width=3)
        draw.text((px + 18, cy - 10), f"{value:.3f}", fill=text, font=ft["num"])

    # Panel B: depth
    x0, y0, x1, y1 = panels[1]
    draw_badge(draw, x0 + 28, y0 + 22, "B", rose, ft["badge"])
    draw.text((x0 + 82, y0 + 20), "Prompted depth changes the mean only slightly", fill=text, font=ft["panel"])
    draw.text((x0 + 82, y0 + 56), "The line is nearly flat; any benefit from asking for more steps is marginal.", fill=muted, font=ft["small"])
    plot_left, plot_right = x0 + 180, x1 - 120
    plot_top, plot_bottom = y0 + 120, y1 - 60
    xs = [plot_left + (plot_right - plot_left) * i / (len(depths) - 1) for i in range(len(depths))]
    min_y = min(v for _, v in depths) - 0.02
    max_y = max(v for _, v in depths) + 0.02
    for tick in [0.38, 0.40, 0.42, 0.44]:
        py = plot_bottom - (plot_bottom - plot_top) * ((tick - min_y) / (max_y - min_y))
        draw.line((plot_left, py, plot_right, py), fill=PALETTE["grid"], width=2)
        label = f"{tick:.2f}"
        tw, th = fit_text(draw, label, ft["small"])
        draw.text((x0 + 28, py - th / 2), label, fill=muted, font=ft["small"])
    points = []
    for idx, (depth, value) in enumerate(depths):
        px = xs[idx]
        py = plot_bottom - (plot_bottom - plot_top) * ((value - min_y) / (max_y - min_y))
        points.append((px, py))
    for idx in range(len(points) - 1):
        draw.line((points[idx][0], points[idx][1], points[idx + 1][0], points[idx + 1][1]), fill="#8ea1b1", width=5)
    for (depth, value), (px, py) in zip(depths, points):
        draw.ellipse((px - 13, py - 13, px + 13, py + 13), fill=rose, outline=PALETTE["card"], width=3)
        label = f"Depth {depth}"
        tw, _ = fit_text(draw, label, ft["label"])
        draw.text((px - tw / 2, plot_bottom + 14), label, fill=text, font=ft["label"])
        draw.text((px - 20, py - 40), f"{value:.3f}", fill=text, font=ft["num"])

    # Panel C: categories
    x0, y0, x1, y1 = panels[2]
    draw_badge(draw, x0 + 28, y0 + 22, "C", teal, ft["badge"])
    draw.text((x0 + 82, y0 + 20), "Dynamics-heavy tasks remain hardest", fill=text, font=ft["panel"])
    draw.text((x0 + 82, y0 + 56), "This is the core benchmark result: state evolution and measurement dominate the residual error budget.", fill=muted, font=ft["small"])
    plot_left, plot_right = x0 + 380, x1 - 95
    plot_top, plot_bottom = y0 + 112, y1 - 34
    max_x = max(v for _, v in categories) + 0.08
    colors = {
        "circuit_evolution": rose,
        "measurement_prediction": orange,
        "operator_composition": navy,
        "entanglement_classification": teal,
    }
    gap = (plot_bottom - plot_top) / len(categories)
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8]:
        px = plot_left + (plot_right - plot_left) * (tick / max_x)
        draw.line((px, plot_top, px, plot_bottom), fill=PALETTE["grid"], width=2)
        label = f"{tick:.1f}"
        tw, _ = fit_text(draw, label, ft["small"])
        draw.text((px - tw / 2, plot_bottom + 6), label, fill=muted, font=ft["small"])
    for idx, (category, value) in enumerate(categories):
        cy = plot_top + gap * (idx + 0.5)
        label = CATEGORY_LABELS[category]
        _, th = fit_text(draw, label, ft["label"])
        draw.text((x0 + 28, cy - th / 2), label, fill=text, font=ft["label"])
        px = plot_left + (plot_right - plot_left) * (value / max_x)
        draw.rounded_rectangle((plot_left, cy - 13, px, cy + 13), radius=11, fill=colors[category])
        draw.text((px + 16, cy - 10), f"{value:.3f}", fill=text, font=ft["num"])

    ensure_parent(path)
    img.save(path)


def render_breakdown(rows: list[dict], path: str, title: str, subtitle: str) -> None:
    W, H = 1600, 1320
    bg = PALETTE["bg"]
    text = PALETTE["text"]
    muted = PALETTE["muted"]
    rose = PALETTE["coral"]
    orange = "#d18a5a"
    navy = PALETTE["navy"]
    teal = PALETTE["teal"]
    ft = fonts()

    cat_means = category_means(rows)
    models, cats, matrix = model_category_matrix(rows)
    colors = {
        "circuit_evolution": rose,
        "measurement_prediction": orange,
        "operator_composition": navy,
        "entanglement_classification": teal,
    }

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    draw.text((70, 42), title, fill=text, font=ft["title"])
    draw.text((70, 90), subtitle, fill=muted, font=ft["small"])

    panels = [(60, 160, 1540, 540), (60, 590, 1540, 1250)]
    for bounds in panels:
        draw_panel_box(draw, bounds)

    # Category difficulty panel
    x0, y0, x1, y1 = panels[0]
    draw_badge(draw, x0 + 28, y0 + 22, "A", PALETTE["blue"], ft["badge"])
    draw.text((x0 + 82, y0 + 20), "Category difficulty", fill=text, font=ft["panel"])
    draw.text((x0 + 82, y0 + 56), "The horizontal ordering exposes which benchmark families dominate the error budget.", fill=muted, font=ft["small"])
    plot_left, plot_right = x0 + 420, x1 - 100
    plot_top, plot_bottom = y0 + 120, y1 - 42
    max_x = max(v for _, v in cat_means) + 0.08
    gap = (plot_bottom - plot_top) / len(cat_means)
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8]:
        px = plot_left + (plot_right - plot_left) * (tick / max_x)
        draw.line((px, plot_top, px, plot_bottom), fill=PALETTE["grid"], width=2)
        label = f"{tick:.1f}"
        tw, _ = fit_text(draw, label, ft["small"])
        draw.text((px - tw / 2, plot_bottom + 10), label, fill=muted, font=ft["small"])
    for idx, (category, value) in enumerate(cat_means):
        cy = plot_top + gap * (idx + 0.5)
        label = CATEGORY_LABELS[category]
        _, th = fit_text(draw, label, ft["label"])
        draw.text((x0 + 28, cy - th / 2), label, fill=text, font=ft["label"])
        px = plot_left + (plot_right - plot_left) * (value / max_x)
        draw.rounded_rectangle((plot_left, cy - 13, px, cy + 13), radius=11, fill=colors[category])
        draw.text((px + 18, cy - 10), f"{value:.3f}", fill=text, font=ft["num"])

    # Heatmap panel
    x0, y0, x1, y1 = panels[1]
    draw_badge(draw, x0 + 28, y0 + 22, "B", PALETTE["teal"], ft["badge"])
    draw.text((x0 + 82, y0 + 20), "Model-by-category error map", fill=text, font=ft["panel"])
    draw.text((x0 + 82, y0 + 56), "Rows are ordered best-to-worst overall. Darker cells mark the failure pockets that remain after averaging over prompts and depths.", fill=muted, font=ft["small"])
    left_margin = x0 + 390
    top_margin = y0 + 146
    cell_w = 255
    cell_h = 58
    for c_idx, category in enumerate(cats):
        label = CATEGORY_LABELS[category]
        tw, th = fit_text(draw, label, ft["small"])
        draw.text((left_margin + c_idx * cell_w + (cell_w - tw) / 2, top_margin - 56), label, fill=text, font=ft["small"])
    for r_idx, model in enumerate(models):
        label = MODEL_LABELS.get(model, model)
        _, th = fit_text(draw, label, ft["small"])
        cy = top_margin + r_idx * cell_h + cell_h / 2
        draw.text((x0 + 28, cy - th / 2), label, fill=text, font=ft["small"])
        row = matrix[r_idx]
        for c_idx, value in enumerate(row):
            x = left_margin + c_idx * cell_w
            y = top_margin + r_idx * cell_h
            t = min(max(value / 0.95, 0.0), 1.0)
            low = (243, 238, 229)
            high = (57, 80, 110)
            fill = tuple(int(low[i] + (high[i] - low[i]) * t) for i in range(3))
            draw.rounded_rectangle((x, y, x + cell_w - 16, y + cell_h - 12), radius=12, fill=fill, outline=PALETTE["card"], width=2)
            val_text = f"{value:.2f}"
            tw, th = fit_text(draw, val_text, ft["num"])
            txt_color = "white" if value >= 0.45 else text
            draw.text((x + (cell_w - 16 - tw) / 2, y + (cell_h - 12 - th) / 2), val_text, fill=txt_color, font=ft["num"])

    ensure_parent(path)
    img.save(path)


def main() -> None:
    args = parse_args()
    rows_2q = load_rows(args.results_2q)
    rows_4q = load_rows(args.results_4q)
    render_takeaways(rows_2q, args.takeaways_out)
    render_breakdown(
        rows_2q,
        args.breakdown_2q_out,
        "Pooled 2-qubit benchmark-family breakdown",
        "The pooled 2-qubit case study shows that dynamics-heavy categories dominate residual error and that model weaknesses are localized rather than uniform.",
    )
    render_breakdown(
        rows_4q,
        args.breakdown_4q_out,
        "Pooled 4-qubit benchmark-family breakdown",
        "The matched 4-qubit rerun preserves the same qualitative failure map, but with higher category means and broader degradation across models.",
    )
    if args.results_8q and args.breakdown_8q_out:
        rows_8q = load_rows(args.results_8q)
        render_breakdown(
            rows_8q,
            args.breakdown_8q_out,
            "Pooled 8-qubit benchmark-family breakdown",
            "The matched 8-qubit rerun further amplifies the same dynamics-heavy failure pattern as the state space grows.",
        )


if __name__ == "__main__":
    main()
