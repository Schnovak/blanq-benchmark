#!/usr/bin/env python3
"""
Simulates a strong detector by perturbing ground truth slightly. This exists
so the eval + leaderboard pipeline has a believable "ceiling" result before
any real tool is wired in. Replace with a real integration -- see README.md
in this folder.
"""
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = os.path.join(ROOT, "ground_truth")
OUT_PATH = os.path.join(ROOT, "results", "mock_near_perfect", "detections.json")

random.seed(42)

JITTER_PX = 1.5          # small localization noise
DROP_PROB = 0.02         # rarely misses a blank entirely
ROW_MISCOUNT_PROB = 0.06 # occasionally off by one row on multi-line fields
TIME_MS_RANGE = (620, 910)


def perturb_box(b):
    return {
        "x": round(b["x"] + random.uniform(-JITTER_PX, JITTER_PX), 2),
        "y": round(b["y"] + random.uniform(-JITTER_PX, JITTER_PX), 2),
        "width": round(b["width"] + random.uniform(-JITTER_PX, JITTER_PX), 2),
        "height": round(b["height"] + random.uniform(-JITTER_PX, JITTER_PX), 2),
        "type": b["type"],
        "confidence": round(random.uniform(0.86, 0.99), 3),
        "rows": _maybe_miscount_rows(b.get("rows")),
    }


def _maybe_miscount_rows(rows):
    if rows is None:
        return None
    if random.random() < ROW_MISCOUNT_PROB:
        return max(1, rows + random.choice([-1, 1]))
    return rows


def main():
    pages = {}
    for fname in sorted(os.listdir(GT_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(GT_DIR, fname)) as f:
            gt = json.load(f)
        detections = [perturb_box(b) for b in gt["blanks"] if random.random() > DROP_PROB]
        pages[gt["id"]] = {
            "detections": detections,
            "detection_time_ms": round(random.uniform(*TIME_MS_RANGE), 1),
            "failed": False,
        }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "tool": "mock_near_perfect",
            "pages": pages,
            "system_info": {"note": "synthetic demo detector, not a real tool"},
        }, f, indent=2)
    print(f"Wrote {OUT_PATH} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
