#!/usr/bin/env python3
"""
Generates the comparison chart shown in README.md and docs/index.html.
Reads every results/<tool>/scores.json and the manifest, and writes a
single PNG at docs/leaderboard.png.

Two panels:
  Left  — headline metrics (precision, recall, F1, mean IoU) grouped by tool
  Right — F1 per category, grouped by tool

Style follows the "release-day" chart convention: light background, no
gridlines except a subtle horizontal one, values labelled on bars, the
subject tool colored, baselines in gray.
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
TOOLS_IN_ORDER = ["pymupdf_naive", "pymupdf_widgets", "blanq"]
TOOL_LABELS = {
    "blanq":            "BlanQ",
    "pymupdf_widgets":  "AcroForm-only baseline",
    "pymupdf_naive":    "Underscore + line heuristic",
}
TOOL_COLORS = {
    "blanq":            "#2563EB",   # blue, the subject
    "pymupdf_widgets":  "#6B7280",   # neutral gray
    "pymupdf_naive":    "#B0B6BF",   # lighter gray
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
        t = page["by_threshold"]["0.5"]
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
    t = a["iou@0.5"]
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
    bar_w = 0.78 / n_tools
    x = np.arange(n_cats)
    for i, tool in enumerate(TOOLS_IN_ORDER):
        vals = [values_by_tool[tool].get(cat, 0) for cat in categories]
        offset = (i - (n_tools - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=TOOL_LABELS[tool],
                      color=TOOL_COLORS[tool], edgecolor="none")
        for bar, v in zip(bars, vals):
            if v == 0:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.03,
                        "0.00", ha="center", va="bottom",
                        fontsize=7.5, color="#9CA3AF", style="italic")
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="#111827")
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS.get(c, c) for c in categories],
                       fontsize=9, color="#111827")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_title(title, fontsize=11, color="#111827", loc="left", pad=14)
    style_axes(ax)


def main():
    cat_map = cat_of_page()

    headline = {t: headline_metrics(t) for t in TOOLS_IN_ORDER}
    per_cat  = {t: per_category_f1(t, cat_map) for t in TOOLS_IN_ORDER}

    plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(13.5, 5.2),
        gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.18})
    fig.patch.set_facecolor("#FFFFFF")

    # LEFT: headline metrics grouped by tool
    metric_names = list(next(iter(headline.values())).keys())
    n_tools = len(TOOLS_IN_ORDER)
    bar_w = 0.78 / n_tools
    x = np.arange(len(metric_names))
    for i, tool in enumerate(TOOLS_IN_ORDER):
        vals = [headline[tool][m] for m in metric_names]
        offset = (i - (n_tools - 1) / 2) * bar_w
        bars = ax_left.bar(x + offset, vals, bar_w,
                           label=TOOL_LABELS[tool],
                           color=TOOL_COLORS[tool], edgecolor="none")
        for bar, v in zip(bars, vals):
            ax_left.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                         f"{v:.2f}", ha="center", va="bottom",
                         fontsize=8, color="#111827")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(metric_names, fontsize=9.5, color="#111827")
    ax_left.set_ylim(0, 1.15)
    ax_left.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_left.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax_left.set_title("Overall — 140 pages, 4,386 ground-truth blanks",
                      fontsize=11, color="#111827", loc="left", pad=14)
    style_axes(ax_left)

    # RIGHT: F1 per category
    draw_grouped_bars(ax_right, CATEGORIES, per_cat, "F1 by category (IoU ≥ 0.5)")

    # Single shared legend up top
    handles, labels = ax_left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(TOOLS_IN_ORDER),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("PDF blank detection, BlanQ v0.1 vs. baselines",
                 fontsize=14, color="#111827", x=0.02, ha="left", y=1.09,
                 weight="bold")
    fig.text(0.02, 1.045,
             "Higher is better. IoU threshold 0.5. BlanQ is the only detector that works on both digital forms and hand-drawn ones (Education).",
             fontsize=9.5, color="#6B7280", ha="left")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
