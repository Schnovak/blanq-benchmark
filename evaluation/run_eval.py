#!/usr/bin/env python3
"""
Score one detector's output against ground truth.

Detections file format (what a tool/competitor submits):
{
  "tool": "my-detector-name",
  "pages": {
    "<page_id>": {
      "detections": [
        {"x": 12.0, "y": 34.0, "width": 100.0, "height": 16.0,
         "type": "single_line", "confidence": 0.97, "rows": null},
        ...
      ],
      "detection_time_ms": 83.4,
      "failed": false
    },
    ...
  },
  "system_info": {"cpu": "...", "ram_gb": 16, "gpu": "none"}
}

`type`, `confidence`, and `rows` are optional per-detection (matching only
needs x/y/width/height); they unlock the type-aware metrics (multiline row
accuracy, line-position error) when present.

Usage:
  python3 evaluation/run_eval.py --detections results/blanq/detections.json \
      --ground-truth ground_truth/ --out results/blanq/scores.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import score_page, aggregate  # noqa: E402


def load_ground_truth(gt_dir):
    pages = {}
    for fname in sorted(os.listdir(gt_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(gt_dir, fname)) as f:
                gt = json.load(f)
            pages[gt["id"]] = gt
    return pages


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", required=True, help="Path to a detections.json submission")
    ap.add_argument("--ground-truth", default="ground_truth", help="Path to ground_truth/ directory")
    ap.add_argument("--out", required=True, help="Where to write the resulting scores.json")
    args = ap.parse_args()

    with open(args.detections) as f:
        submission = json.load(f)

    gt_pages = load_ground_truth(args.ground_truth)
    tool_name = submission.get("tool", os.path.basename(os.path.dirname(args.detections)))

    page_scores = []
    missing_pages = []
    for page_id, gt in gt_pages.items():
        det_page = submission.get("pages", {}).get(page_id)
        if det_page is None:
            missing_pages.append(page_id)
            det_page = {"detections": [], "failed": True}
        page_scores.append(score_page(gt, det_page))

    result = {
        "tool": tool_name,
        "per_page": {
            s.page_id: {
                "n_gt": s.n_gt, "n_det": s.n_det,
                "by_threshold": s.by_threshold,
                "detection_time_ms": s.detection_time_ms,
                "failed": s.failed,
            } for s in page_scores
        },
        "aggregate": aggregate(page_scores),
        "missing_pages": missing_pages,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    agg = result["aggregate"]
    print(f"Tool: {tool_name}")
    print(f"Pages scored: {agg['n_pages']}  |  GT blanks: {agg['n_gt_blanks']}")
    if missing_pages:
        print(f"WARNING: {len(missing_pages)} page(s) had no submission and were scored as full misses: {missing_pages}")
    for thr in (0.5, 0.75, 0.9):
        m = agg[f"iou@{thr}"]
        print(f"  IoU>={thr:<4} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(f"  Mean IoU (matched): {agg['mean_iou']}")
    print(f"  % blanks detected (IoU>=0.5): {agg['pct_blanks_detected_iou50']}%")
    print(f"  Median line-position error: {agg['median_line_error_px']} px")
    print(f"  Multiline row accuracy: {agg['multiline_row_accuracy']}")
    print(f"  Mean detection time: {agg['mean_detection_ms_per_page']} ms/page")
    print(f"  Failure rate: {agg['failure_rate']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
