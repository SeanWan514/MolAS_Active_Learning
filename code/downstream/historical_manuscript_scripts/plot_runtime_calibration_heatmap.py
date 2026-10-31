#!/usr/bin/env python3
"""Render queried-cell runtime calibration as a dependency-free SVG heatmap."""

from __future__ import annotations

import csv
import html
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(
    "/Users/wanweilin/Documents/PORT-AS/MolAS_AL_restart/AL_MolAS/"
    "mean_ucb_cost_5fold_paper/outputs/runs"
)
OUT = Path(
    "/Users/wanweilin/Downloads/SRS_2026_AL_MolAS_Docking/"
    "NeurIPS2026/figures"
)
MAX_PERCENTAGE = 20

METHODS = {
    "plain_mean_ucb": ["Mean-UCB"],
    "direct_cost_uc": ["Direct", "cost-utility"],
    "relaxed_cost_power_05": ["Relaxed cost", "(alpha = 0.5)"],
    "log_cost": ["Log cost"],
    "runtime_uncertainty": ["Runtime", "uncertainty"],
    "cost_tiers_5": ["Five cost tiers"],
}
ALGORITHMS = [
    ("surfdock", "SurfDock"),
    ("unimol", "Uni-Mol"),
    ("diffdock", "DiffDock"),
    ("diffdock_l", "DiffDock-L"),
    ("gnina", "GNINA"),
    ("autodock", "AutoDock Vina"),
    ("smina", "Smina"),
]


def geometric_mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value) and value > 0]
    return math.exp(sum(math.log(value) for value in clean) / len(clean)) if clean else math.nan


def read_queries() -> dict[tuple[str, str], list[tuple[float, float]]]:
    records: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for method in METHODS:
        pattern = (
            f"fold0/seed0/queried_cells/cumulative/"
            f"queried_{MAX_PERCENTAGE:03d}pct.csv"
        )
        for path in sorted((ROOT / method).glob(pattern)):
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    algorithm = row["algorithm"].strip().lower()
                    try:
                        predicted = float(row["predicted_runtime_mean"])
                        realized = float(row["runtime"])
                    except (TypeError, ValueError):
                        continue
                    if predicted > 0 and realized > 0:
                        records[(method, algorithm)].append((predicted, realized))
        print(f"Loaded {method}", flush=True)
    return records


def summarize(records):
    summaries = {}
    rows = []
    for method in METHODS:
        for algorithm, _ in ALGORITHMS:
            pairs = records[(method, algorithm)]
            predicted = [pair[0] for pair in pairs]
            realized = [pair[1] for pair in pairs]
            p_geo = geometric_mean(predicted)
            r_geo = geometric_mean(realized)
            bias = (
                statistics.median(
                    math.log2(prediction / observation)
                    for prediction, observation in pairs
                )
                if pairs
                else math.nan
            )
            summaries[(method, algorithm)] = (p_geo, r_geo, bias, len(pairs))
            rows.append(
                {
                    "method": method,
                    "algorithm": algorithm,
                    "queried_cells": len(pairs),
                    "predicted_geomean_seconds": f"{p_geo:.8g}",
                    "realized_geomean_seconds": f"{r_geo:.8g}",
                    "median_log2_predicted_over_realized": f"{bias:.8g}",
                }
            )
    return summaries, rows


def interpolate(a: tuple[int, int, int], b: tuple[int, int, int], t: float):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def color_for(value: float, limit: float) -> str:
    blue = (49, 105, 168)
    neutral = (248, 248, 246)
    red = (196, 71, 72)
    fraction = min(1.0, abs(value) / limit)
    rgb = interpolate(neutral, red if value >= 0 else blue, fraction)
    return "#%02x%02x%02x" % rgb


def svg_text(x, y, content, size=22, anchor="middle", weight="normal", fill="#20242a"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{html.escape(content)}</text>'
    )


def render_svg(summaries) -> str:
    width, height = 1800, 900
    left, top = 255, 145
    cell_w, cell_h = 225, 86
    values = [entry[2] for entry in summaries.values() if math.isfinite(entry[2])]
    ordered = sorted(abs(value) for value in values)
    limit = max(2.0, ordered[round(0.95 * (len(ordered) - 1))])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 45, "Predicted versus realized runtime on actively queried cells", 31, weight="bold"),
        svg_text(width / 2, 78, "Queries acquired through 20%; representative matched trajectory (fold 0, seed 0)", 19, fill="#4b5563"),
    ]

    for column, lines in enumerate(METHODS.values()):
        x = left + column * cell_w + cell_w / 2
        for line_index, line in enumerate(lines):
            parts.append(svg_text(x, 112 + line_index * 22, line, 18, weight="bold"))

    for row, (algorithm, algorithm_label) in enumerate(ALGORITHMS):
        y = top + row * cell_h
        parts.append(svg_text(left - 18, y + cell_h / 2 + 7, algorithm_label, 19, anchor="end", weight="bold"))
        for column, method in enumerate(METHODS):
            x = left + column * cell_w
            predicted, realized, bias, count = summaries[(method, algorithm)]
            if not math.isfinite(bias):
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                    f'fill="#eeeeec" stroke="white" stroke-width="3"/>'
                )
                parts.append(svg_text(x + cell_w / 2, y + 49, "Not queried", 16, fill="#737373"))
                continue
            fill = color_for(bias, limit)
            text_color = "white" if abs(bias) > 0.58 * limit else "#17191d"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{fill}" stroke="white" stroke-width="3"/>'
            )
            parts.append(svg_text(x + cell_w / 2, y + 36, f"{predicted:.1f} / {realized:.1f} s", 18, fill=text_color, weight="bold"))
            parts.append(svg_text(x + cell_w / 2, y + 63, f"n = {count:,}", 15, fill=text_color))

    legend_y = top + len(ALGORITHMS) * cell_h + 52
    legend_x = left + 265
    legend_w = 820
    steps = 100
    for step in range(steps):
        value = -limit + (2 * limit * step / (steps - 1))
        x = legend_x + legend_w * step / steps
        parts.append(
            f'<rect x="{x:.2f}" y="{legend_y}" width="{legend_w / steps + 1:.2f}" '
            f'height="18" fill="{color_for(value, limit)}"/>'
        )
    parts.extend(
        [
            svg_text(legend_x - 8, legend_y + 15, f"-{limit:.1f}", 14, anchor="end"),
            svg_text(legend_x + legend_w / 2, legend_y + 15, "0", 14),
            svg_text(legend_x + legend_w + 8, legend_y + 15, f"+{limit:.1f}", 14, anchor="start"),
            svg_text(legend_x + legend_w / 2, legend_y + 48, "Median log2(predicted / realized runtime)", 17),
            svg_text(width / 2, height - 26, "Blue: underprediction   |   Red: overprediction   |   Cell text: predicted / realized geometric-mean seconds", 16, fill="#374151"),
            "</svg>",
        ]
    )
    return "\n".join(parts)


def main():
    summaries, rows = summarize(read_queries())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runtime_calibration_heatmap.svg").write_text(render_svg(summaries))
    with (OUT / "runtime_calibration_heatmap_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote SVG and data to {OUT}")


if __name__ == "__main__":
    main()
