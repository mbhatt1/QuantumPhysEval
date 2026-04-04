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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render clean paper figures from benchmark result JSON files.")
    parser.add_argument("--results-2q", required=True)
    parser.add_argument("--results-4q", required=True)
    parser.add_argument("--takeaways-out", required=True)
    parser.add_argument("--breakdown-2q-out", required=True)
    parser.add_argument("--breakdown-4q-out", required=True)
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
    W, H = 1500, 1180
    bg = "white"
    text = "#1f2937"
    muted = "#6b7280"
    panel = "#f8fafc"
    grid = "#e5e7eb"
    blue = "#4f8fba"
    rose = "#c75c5c"
    teal = "#81b29a"
    navy = "#3d405b"
    orange = "#e07a5f"

    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font_bold = "/System/Library/Fonts/Helvetica.ttc"
    ft_title = ImageFont.truetype(font_bold, 32)
    ft_panel = ImageFont.truetype(font_bold, 24)
    ft_label = ImageFont.truetype(font_path, 21)
    ft_small = ImageFont.truetype(font_path, 18)
    ft_num = ImageFont.truetype(font_bold, 20)

    models = sorted(group_mean(rows, "model"), key=lambda item: item[1])
    depths = group_mean_int(rows, "depth")
    categories = category_means(rows)

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    draw.text((70, 45), "Direct takeaways from the pooled 2-qubit case study", fill=text, font=ft_title)
    draw.text(
        (70, 90),
        "These panels summarize the three strongest benchmark-level findings: capability matters, prompted depth barely moves the mean, and exact quantum dynamics remain the hardest task family.",
        fill=muted,
        font=ft_small,
    )

    panels = [
        (60, 150, 1440, 460),
        (60, 500, 1440, 790),
        (60, 830, 1440, 1120),
    ]
    for bounds in panels:
        draw.rounded_rectangle(bounds, radius=24, fill=panel, outline=grid, width=2)

    # Panel A: model ranking
    x0, y0, x1, y1 = panels[0]
    draw.text((x0 + 28, y0 + 20), "A. Stronger models reduce physics-grounded error", fill=text, font=ft_panel)
    draw.text((x0 + 28, y0 + 56), "Lower values are better. The best pooled score belongs to Claude Opus 4.1, and the weakest is GPT-4o mini.", fill=muted, font=ft_small)
    plot_left, plot_right = x0 + 280, x1 - 80
    plot_top, plot_bottom = y0 + 112, y1 - 30
    max_x = max(v for _, v in models) + 0.08
    rows_y = (plot_bottom - plot_top) / len(models)
    for tick in [0.0, 0.2, 0.4, 0.6]:
        px = plot_left + (plot_right - plot_left) * (tick / max_x)
        draw.line((px, plot_top, px, plot_bottom), fill=grid, width=2)
        label = f"{tick:.1f}"
        tw, _ = fit_text(draw, label, ft_small)
        draw.text((px - tw / 2, plot_bottom + 6), label, fill=muted, font=ft_small)
    for idx, (model, value) in enumerate(models):
        cy = plot_top + rows_y * (idx + 0.5)
        label = MODEL_LABELS.get(model, model)
        _, th = fit_text(draw, label, ft_label)
        draw.text((x0 + 28, cy - th / 2), label, fill=text, font=ft_label)
        px = plot_left + (plot_right - plot_left) * (value / max_x)
        draw.line((plot_left, cy, px, cy), fill="#d5dbe3", width=7)
        draw.ellipse((px - 11, cy - 11, px + 11, cy + 11), fill=blue, outline="white", width=2)
        draw.text((px + 18, cy - 11), f"{value:.3f}", fill=text, font=ft_num)

    # Panel B: depth
    x0, y0, x1, y1 = panels[1]
    draw.text((x0 + 28, y0 + 20), "B. Prompted depth changes the mean only slightly", fill=text, font=ft_panel)
    draw.text((x0 + 28, y0 + 56), "Depth 1, 2, and 4 remain tightly clustered in the pooled benchmark.", fill=muted, font=ft_small)
    plot_left, plot_right = x0 + 160, x1 - 110
    plot_top, plot_bottom = y0 + 120, y1 - 60
    xs = [plot_left + (plot_right - plot_left) * i / (len(depths) - 1) for i in range(len(depths))]
    min_y = min(v for _, v in depths) - 0.02
    max_y = max(v for _, v in depths) + 0.02
    for tick in [0.38, 0.40, 0.42, 0.44]:
        py = plot_bottom - (plot_bottom - plot_top) * ((tick - min_y) / (max_y - min_y))
        draw.line((plot_left, py, plot_right, py), fill=grid, width=2)
        label = f"{tick:.2f}"
        tw, th = fit_text(draw, label, ft_small)
        draw.text((x0 + 28, py - th / 2), label, fill=muted, font=ft_small)
    points = []
    for idx, (depth, value) in enumerate(depths):
        px = xs[idx]
        py = plot_bottom - (plot_bottom - plot_top) * ((value - min_y) / (max_y - min_y))
        points.append((px, py))
    for idx in range(len(points) - 1):
        draw.line((points[idx][0], points[idx][1], points[idx + 1][0], points[idx + 1][1]), fill="#a3adb9", width=4)
    for (depth, value), (px, py) in zip(depths, points):
        draw.ellipse((px - 13, py - 13, px + 13, py + 13), fill=rose, outline="white", width=2)
        label = f"Depth {depth}"
        tw, _ = fit_text(draw, label, ft_label)
        draw.text((px - tw / 2, plot_bottom + 14), label, fill=text, font=ft_label)
        draw.text((px - 20, py - 40), f"{value:.3f}", fill=text, font=ft_num)

    # Panel C: categories
    x0, y0, x1, y1 = panels[2]
    draw.text((x0 + 28, y0 + 20), "C. Dynamics-heavy tasks remain hardest", fill=text, font=ft_panel)
    draw.text((x0 + 28, y0 + 56), "Circuit evolution and measurement prediction dominate the residual error budget.", fill=muted, font=ft_small)
    plot_left, plot_right = x0 + 350, x1 - 90
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
        draw.line((px, plot_top, px, plot_bottom), fill=grid, width=2)
        label = f"{tick:.1f}"
        tw, _ = fit_text(draw, label, ft_small)
        draw.text((px - tw / 2, plot_bottom + 6), label, fill=muted, font=ft_small)
    for idx, (category, value) in enumerate(categories):
        cy = plot_top + gap * (idx + 0.5)
        label = CATEGORY_LABELS[category]
        _, th = fit_text(draw, label, ft_label)
        draw.text((x0 + 28, cy - th / 2), label, fill=text, font=ft_label)
        px = plot_left + (plot_right - plot_left) * (value / max_x)
        draw.rounded_rectangle((plot_left, cy - 13, px, cy + 13), radius=11, fill=colors[category])
        draw.text((px + 16, cy - 11), f"{value:.3f}", fill=text, font=ft_num)

    ensure_parent(path)
    img.save(path)


def render_breakdown(rows: list[dict], path: str, title: str, subtitle: str) -> None:
    W, H = 1560, 1280
    bg = "white"
    text = "#1f2937"
    muted = "#6b7280"
    panel = "#f8fafc"
    grid = "#e5e7eb"
    rose = "#b56576"
    orange = "#e07a5f"
    navy = "#3d405b"
    teal = "#81b29a"

    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font_bold = "/System/Library/Fonts/Helvetica.ttc"
    ft_title = ImageFont.truetype(font_bold, 32)
    ft_panel = ImageFont.truetype(font_bold, 24)
    ft_label = ImageFont.truetype(font_path, 20)
    ft_small = ImageFont.truetype(font_path, 18)
    ft_num = ImageFont.truetype(font_bold, 18)

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
    draw.text((70, 45), title, fill=text, font=ft_title)
    draw.text((70, 90), subtitle, fill=muted, font=ft_small)

    panels = [(60, 160, 1500, 540), (60, 590, 1500, 1220)]
    for bounds in panels:
        draw.rounded_rectangle(bounds, radius=24, fill=panel, outline=grid, width=2)

    # Category difficulty panel
    x0, y0, x1, y1 = panels[0]
    draw.text((x0 + 28, y0 + 20), "A. Category difficulty", fill=text, font=ft_panel)
    draw.text((x0 + 28, y0 + 56), "Higher values indicate larger mean normalized physics error.", fill=muted, font=ft_small)
    plot_left, plot_right = x0 + 390, x1 - 100
    plot_top, plot_bottom = y0 + 120, y1 - 42
    max_x = max(v for _, v in cat_means) + 0.08
    gap = (plot_bottom - plot_top) / len(cat_means)
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8]:
        px = plot_left + (plot_right - plot_left) * (tick / max_x)
        draw.line((px, plot_top, px, plot_bottom), fill=grid, width=2)
        label = f"{tick:.1f}"
        tw, _ = fit_text(draw, label, ft_small)
        draw.text((px - tw / 2, plot_bottom + 10), label, fill=muted, font=ft_small)
    for idx, (category, value) in enumerate(cat_means):
        cy = plot_top + gap * (idx + 0.5)
        label = CATEGORY_LABELS[category]
        _, th = fit_text(draw, label, ft_label)
        draw.text((x0 + 28, cy - th / 2), label, fill=text, font=ft_label)
        px = plot_left + (plot_right - plot_left) * (value / max_x)
        draw.rounded_rectangle((plot_left, cy - 13, px, cy + 13), radius=11, fill=colors[category])
        draw.text((px + 18, cy - 10), f"{value:.3f}", fill=text, font=ft_num)

    # Heatmap panel
    x0, y0, x1, y1 = panels[1]
    draw.text((x0 + 28, y0 + 20), "B. Model-by-category error map", fill=text, font=ft_panel)
    draw.text((x0 + 28, y0 + 56), "Darker cells indicate higher average error. Rows are sorted from strongest to weakest overall.", fill=muted, font=ft_small)
    left_margin = x0 + 350
    top_margin = y0 + 132
    cell_w = 255
    cell_h = 52
    for c_idx, category in enumerate(cats):
        label = CATEGORY_LABELS[category]
        tw, th = fit_text(draw, label, ft_small)
        draw.text((left_margin + c_idx * cell_w + (cell_w - tw) / 2, top_margin - 52), label, fill=text, font=ft_small)
    for r_idx, model in enumerate(models):
        label = MODEL_LABELS.get(model, model)
        _, th = fit_text(draw, label, ft_small)
        cy = top_margin + r_idx * cell_h + cell_h / 2
        draw.text((x0 + 28, cy - th / 2), label, fill=text, font=ft_small)
        row = matrix[r_idx]
        for c_idx, value in enumerate(row):
            x = left_margin + c_idx * cell_w
            y = top_margin + r_idx * cell_h
            shade = int(245 - min(value / 0.85, 1.0) * 140)
            fill = (shade, 233, 214) if value < 0.25 else (max(26, shade - 40), max(76, shade), max(107, shade + 8))
            draw.rounded_rectangle((x, y, x + cell_w - 16, y + cell_h - 12), radius=10, fill=fill, outline="white", width=2)
            val_text = f"{value:.2f}"
            tw, th = fit_text(draw, val_text, ft_num)
            txt_color = "white" if value >= 0.45 else text
            draw.text((x + (cell_w - 16 - tw) / 2, y + (cell_h - 12 - th) / 2), val_text, fill=txt_color, font=ft_num)

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


if __name__ == "__main__":
    main()
