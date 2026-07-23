#!/usr/bin/env python3
"""
Aggregates every results/<tool>/scores.json into docs/leaderboard.json and
renders docs/index.html (the GitHub Pages leaderboard) from
docs/index_template.html.

Run this after adding/updating any tool's scores:
  python3 scripts/build_leaderboard.py
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
DOCS_DIR = os.path.join(ROOT, "docs")
MANIFEST_PATH = os.path.join(ROOT, "dataset", "manifest.csv")
TEMPLATE_PATH = os.path.join(DOCS_DIR, "index_template.html")


def dataset_stats():
    n_categories = 0
    n_pages = 0
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            rows = list(csv.DictReader(f))
        n_pages = len(rows)
        n_categories = len({r["category"] for r in rows if r.get("category")})
    return n_pages, n_categories


def load_tool_summary(tool_dir):
    scores_path = os.path.join(tool_dir, "scores.json")
    if not os.path.exists(scores_path):
        return None
    with open(scores_path) as f:
        scores = json.load(f)
    agg = scores["aggregate"]
    return {
        "tool": scores.get("tool", os.path.basename(tool_dir)),
        "f1_50": agg["iou@0.5"]["f1"],
        "precision_50": agg["iou@0.5"]["precision"],
        "recall_50": agg["iou@0.5"]["recall"],
        "f1_75": agg["iou@0.75"]["f1"],
        "f1_90": agg["iou@0.9"]["f1"],
        "mean_iou": agg["mean_iou"],
        "pct_detected_50": agg["pct_blanks_detected_iou50"],
        "median_line_error_px": agg["median_line_error_px"],
        "multiline_row_accuracy": agg["multiline_row_accuracy"],
        "mean_ms_per_page": agg["mean_detection_ms_per_page"],
        "failure_rate": agg["failure_rate"],
        "n_gt_blanks": agg["n_gt_blanks"],
    }


def main():
    n_pages, n_categories = dataset_stats()

    tools = []
    if os.path.isdir(RESULTS_DIR):
        for name in sorted(os.listdir(RESULTS_DIR)):
            tool_dir = os.path.join(RESULTS_DIR, name)
            if os.path.isdir(tool_dir):
                summary = load_tool_summary(tool_dir)
                if summary:
                    tools.append(summary)

    n_gt_blanks = tools[0]["n_gt_blanks"] if tools else 0

    data = {
        "dataset": {"n_pages": n_pages, "n_gt_blanks": n_gt_blanks, "n_categories": n_categories},
        "tools": tools,
    }

    os.makedirs(DOCS_DIR, exist_ok=True)
    leaderboard_json_path = os.path.join(DOCS_DIR, "leaderboard.json")
    with open(leaderboard_json_path, "w") as f:
        json.dump(data, f, indent=2)

    with open(TEMPLATE_PATH) as f:
        template = f.read()
    html = template.replace("__LEADERBOARD_DATA__", json.dumps(data))
    with open(os.path.join(DOCS_DIR, "index.html"), "w") as f:
        f.write(html)

    print(f"{len(tools)} tool(s) in leaderboard: {[t['tool'] for t in tools]}")
    print(f"Wrote {leaderboard_json_path}")
    print(f"Wrote {os.path.join(DOCS_DIR, 'index.html')}")


if __name__ == "__main__":
    main()
