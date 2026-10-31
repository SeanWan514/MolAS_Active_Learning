#!/usr/bin/env python3
"""Per-algorithm predicted/realized runtime density for log-cost Mean-UCB."""

import csv
import html
import math
import statistics
from collections import defaultdict
from pathlib import Path

SOURCE = Path(
    "/Users/wanweilin/Documents/PORT-AS/MolAS_AL_restart/AL_MolAS/"
    "mean_ucb_cost_5fold_paper/outputs/runs/log_cost/fold0/seed0/"
    "queried_cells/cumulative/queried_040pct.csv"
)
OUT = Path("/Users/wanweilin/Downloads/SRS_2026_AL_MolAS_Docking/NeurIPS2026/figures")
ALGORITHMS = [
    ("surfdock", "SurfDock"), ("smina", "Smina"), ("unimol", "Uni-Mol"),
    ("gnina", "GNINA"), ("diffdock_l", "DiffDock-L"),
    ("diffdock", "DiffDock"), ("autodock", "AutoDock Vina"),
]


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else 0.0


def text(x, y, value, size=18, anchor="middle", weight="normal"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="Arial,Helvetica,sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="#20242a">{html.escape(value)}</text>')


def color(t):
    stops = [(68, 1, 84), (49, 104, 142), (34, 168, 132), (253, 231, 37)]
    t = min(1.0, max(0.0, t)) * (len(stops) - 1)
    i = min(len(stops) - 2, int(t)); f = t - i
    rgb = tuple(round(stops[i][k] + f * (stops[i + 1][k] - stops[i][k])) for k in range(3))
    return "#%02x%02x%02x" % rgb


data = defaultdict(list)
with SOURCE.open(newline="") as handle:
    for row in csv.DictReader(handle):
        try:
            predicted = float(row["predicted_runtime_mean"])
            realized = float(row["runtime"])
        except (TypeError, ValueError):
            continue
        if predicted > 0 and realized > 0:
            data[row["algorithm"].lower()].append((predicted, realized))

width, height = 1800, 930
panel_w, panel_h = 400, 320
lefts = [75, 505, 935, 1365]
tops = [125, 525]
log_min, log_max, bins = math.log10(0.2), math.log10(5000), 38
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
       '<rect width="100%" height="100%" fill="white"/>',
       text(width / 2, 42, "Predicted vs. realized docking runtime by algorithm", 30, weight="bold"),
       text(width / 2, 72, "Log-cost Mean-UCB; queried cells through 40% (fold 0, seed 0)", 18)]

for p, (algorithm, label) in enumerate(ALGORITHMS):
    row, col = divmod(p, 4); x0, y0 = lefts[col], tops[row]
    pairs = data[algorithm]; xs = [x for x, _ in pairs]; ys = [y for _, y in pairs]
    if not pairs:
        svg += [text(x0 + panel_w / 2, y0 - 38, label, 21, weight="bold"),
                text(x0 + panel_w / 2, y0 + panel_h / 2, "Not queried", 18)]
        continue
    rho = spearman(xs, ys); mae = statistics.mean(abs(x - y) for x, y in pairs)
    svg += [text(x0 + panel_w / 2, y0 - 38, label, 21, weight="bold"),
            text(x0 + panel_w / 2, y0 - 13, f"rho = {rho:.3f}, MAE = {mae:.1f} s", 16)]
    counts = [[0] * bins for _ in range(bins)]
    for x, y in pairs:
        bx = int((math.log10(x) - log_min) / (log_max - log_min) * bins)
        by = int((math.log10(y) - log_min) / (log_max - log_min) * bins)
        if 0 <= bx < bins and 0 <= by < bins: counts[by][bx] += 1
    maximum = max(max(line) for line in counts) or 1
    cw, ch = panel_w / bins, panel_h / bins
    for by in range(bins):
        for bx in range(bins):
            count = counts[by][bx]
            if count:
                intensity = math.log1p(count) / math.log1p(maximum)
                svg.append(f'<rect x="{x0 + bx*cw:.2f}" y="{y0 + panel_h-(by+1)*ch:.2f}" '
                           f'width="{cw+0.4:.2f}" height="{ch+0.4:.2f}" fill="{color(intensity)}"/>')
    svg.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#333" stroke-width="1.5"/>')
    svg.append(f'<line x1="{x0}" y1="{y0+panel_h}" x2="{x0+panel_w}" y2="{y0}" stroke="#e11d48" stroke-width="2" stroke-dasharray="8,6"/>')
    for tick, tick_label in [(0, "0.2"), (0.159, "1"), (0.386, "10"), (0.614, "100"), (0.841, "1000")]:
        tx = x0 + tick * panel_w; ty = y0 + panel_h - tick * panel_h
        svg += [f'<line x1="{tx}" y1="{y0+panel_h}" x2="{tx}" y2="{y0+panel_h+6}" stroke="#333"/>',
                text(tx, y0 + panel_h + 23, tick_label, 13),
                f'<line x1="{x0-6}" y1="{ty}" x2="{x0}" y2="{ty}" stroke="#333"/>']
        if col == 0: svg.append(text(x0 - 10, ty + 5, tick_label, 13, anchor="end"))
    if row == 1: svg.append(text(x0 + panel_w/2, y0 + panel_h + 49, "Predicted runtime (s)", 16))
    if col == 0:
        svg.append(f'<text transform="translate({x0-53},{y0+panel_h/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="16">Realized runtime (s)</text>')

svg.append('</svg>')
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "log_cost_predicted_vs_realized_by_algorithm.svg").write_text("\n".join(svg))
print("Wrote", OUT / "log_cost_predicted_vs_realized_by_algorithm.svg")
