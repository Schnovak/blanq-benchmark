#!/usr/bin/env python3
"""
Generates the comparison chart shown in README.md and docs/index.html.
Reads every results/<tool>/scores.json and the manifest, and writes a
single PNG at docs/leaderboard.png.

Fool-proof layout: F1 is the ONLY metric shown on the chart. Precision,
recall, mean IoU and speed live in the scores.json files for anyone who
wants the fine detail. This prevents "precision looks most important
because it's leftmost" misreads.

Two panels:
  Left  — overall F1, one bar per detector (sorted best-to-worst)
  Right — F1 per category, grouped by detector

Style follows the "release-day" chart convention: light background,
subtle gridlines, values labelled on bars, subject tool coloured,
baselines in gray. Larger gaps between bar groups so labels never
touch.
"""
import csv, json, os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "dataset", "manifest.csv")
RESULTS  = os.path.join(ROOT, "results")
OUT_PNG  = os.path.join(ROOT, "docs", "leaderboard.png")

# Ordering matters: BlanQ last so it draws on top and picks the accent color.
TOOLS_IN_ORDER = ["gemini_flash_lite", "pymupdf_naive", "blanq"]
TOOL_LABELS = {
    "blanq":             "BlanQ",
    "pymupdf_naive":     "Underscore + line heuristic",
    "gemini_flash_lite": "Gemini 2.5 Flash Lite",
}
TOOL_COLORS = {
    "blanq":             "#2563EB",   # blue, the subject
    "pymupdf_naive":     "#9CA3AF",   # gray baseline
    "gemini_flash_lite": "#D97706",   # amber for the LLM
}

CATEGORIES = ["education", "government", "banking_insurance", "medical", "hr", "legal"]
CATEGORY_LABELS = {
    "education": "Education",
    "government": "Government",
    "banking_insurance": "Banking",
    "medical": "Medical",
    "hr": "HR",
    "legal": "Legal",
}


def cat_of_page():
    with open(MANIFEST) as f:
        return {r["id"]: r["category"] for r in csv.DictReader(f)}


def per_category_f1(tool_name, cat_map):
    with open(os.path.join(RESULTS, tool_name, "scores.json")) as f:
        s = json.load(f)
    agg = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    for page_id, page in s["per_page"].items():
        cat = cat_map.get(page_id)
        if not cat:
            continue
        t = page.get("line_alignment") or page["by_threshold"]["0.5"]
        agg[cat][0] += t["tp"]; agg[cat][1] += t["fp"]; agg[cat][2] += t["fn"]  # noqa
    out = {}
    for cat, (tp, fp, fn) in agg.items():
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        out[cat] = 2 * p * r / (p + r) if (p + r) else 0
    return out


def headline_metrics(tool_name):
    with open(os.path.join(RESULTS, tool_name, "scores.json")) as f:
        s = json.load(f)
    a = s["aggregate"]
    t = a.get("line_alignment") or a["iou@0.5"]
    return {
        "Precision": t["precision"],
        "Recall":    t["recall"],
        "F1":        t["f1"],
        "Mean IoU":  a.get("mean_iou", 0),
    }


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(colors="#4B5563", length=0)
    ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.6)
    ax.set_axisbelow(True)


def draw_grouped_bars(ax, categories, values_by_tool, title):
    n_tools = len(TOOLS_IN_ORDER)
    n_cats  = len(categories)
    bar_w = 0.62 / n_tools
    x = np.arange(n_cats) * 1.15
    for i, tool in enumerate(TOOLS_IN_ORDER):
        vals = [values_by_tool[tool].get(cat, 0) for cat in categories]
        offset = (i - (n_tools - 1) / 2) * bar_w
        ax.bar(x + offset, vals, bar_w, label=TOOL_LABELS[tool],
               color=TOOL_COLORS[tool], edgecolor="none")
    # No numeric labels on the bars: the y-axis is precise enough and
    # avoids collisions on tied values. Absolute F1 numbers live on the
    # left panel and in scores.json.
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS.get(c, c) for c in categories],
                       fontsize=10, color="#111827")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=9, color="#6B7280")
    ax.set_title(title, fontsize=12, color="#111827", loc="left", pad=14, weight="bold")
    style_axes(ax)


def main():
    cat_map = cat_of_page()

    headline = {t: headline_metrics(t) for t in TOOLS_IN_ORDER}
    per_cat  = {t: per_category_f1(t, cat_map) for t in TOOLS_IN_ORDER}

    # Sort detectors by overall F1 (worst first so bars ascend to the winner)
    tools_ranked = sorted(TOOLS_IN_ORDER, key=lambda t: headline[t]["F1"])

    plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14.5, 5.6),
        gridspec_kw={"width_ratios": [1, 1.9], "wspace": 0.22})
    fig.patch.set_facecolor("#FFFFFF")

    # LEFT: overall F1 only, one bar per detector, sorted best-to-worst.
    # F1 is the only headline number; precision/recall/speed live in
    # scores.json so nothing can misread a leftmost bar as "most important".
    labels = [TOOL_LABELS[t] for t in tools_ranked]
    colors = [TOOL_COLORS[t] for t in tools_ranked]
    values = [headline[t]["F1"] for t in tools_ranked]
    y = np.arange(len(tools_ranked))
    bars = ax_left.barh(y, values, height=0.62, color=colors, edgecolor="none")
    for bar, v, t in zip(bars, values, tools_ranked):
        weight = "bold" if t == "blanq" else "normal"
        ax_left.text(v + 0.015, bar.get_y() + bar.get_height() / 2,
                     f"{v:.2f}", va="center", ha="left",
                     fontsize=12, color="#111827", weight=weight)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(labels, fontsize=10.5, color="#111827")
    ax_left.set_xlim(0, 1.1)
    ax_left.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_left.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=8.5, color="#6B7280")
    ax_left.set_title("Overall F1  (140 pages, 4,386 blanks)",
                      fontsize=12, color="#111827", loc="left", pad=14, weight="bold")
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.spines["left"].set_visible(False)
    ax_left.spines["bottom"].set_color("#D1D5DB")
    ax_left.xaxis.grid(True, color="#E5E7EB", linewidth=0.6)
    ax_left.tick_params(colors="#4B5563", length=0)
    ax_left.set_axisbelow(True)

    # RIGHT: F1 per category, grouped bars in fixed left-to-right tool order
    draw_grouped_bars(ax_right, CATEGORIES, per_cat, "F1 by category")

    # Single shared legend up top, ordered same as the right chart
    from matplotlib.patches import Patch
    handles = [Patch(color=TOOL_COLORS[t], label=TOOL_LABELS[t]) for t in TOOLS_IN_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=len(TOOLS_IN_ORDER),
               frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("PDF blank detection, BlanQ v0.1 vs. baselines and a frontier LLM",
                 fontsize=15, color="#111827", x=0.02, ha="left", y=1.10,
                 weight="bold")
    fig.text(0.02, 1.05,
             "F1 score across 140 pages. A detection matches a blank when its bottom edge is within 6pt of the ground-truth bottom AND covers ≥95% of the horizontal span.",
             fontsize=10, color="#6B7280", ha="left")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
