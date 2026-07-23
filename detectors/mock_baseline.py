#!/usr/bin/env python3
"""
Simulates a weak/naive detector: misses small elements (checkboxes/radios),
merges multi-row fields into a single box, adds coarse jitter, and
occasionally hallucinates a spurious box. Gives the leaderboard visible
spread against mock_near_perfect.py. Replace with real tool integrations --
see README.md in this folder.
"""
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = os.path.join(ROOT, "ground_truth")
OUT_PATH = os.path.join(ROOT, "results", "mock_baseline", "detections.json")

random.seed(7)

JITTER_PX = 9.0
MISS_SMALL_PROB = 0.55       # frequently misses checkbox/radio/tiny_box
HALLUCINATE_PROB = 0.25      # sometimes adds a bogus box per page
TIME_MS_RANGE = (1400, 2600)
SMALL_TYPES = {"checkbox", "radio", "tiny_box"}
MULTIROW_TYPES = {"multi_line", "large_paragraph"}


def perturb_box(b):
    det = {
        "x": round(b["x"] + random.uniform(-JITTER_PX, JITTER_PX), 2),
        "y": round(b["y"] + random.uniform(-JITTER_PX, JITTER_PX), 2),
        "width": round(b["width"] + random.uniform(-JITTER_PX, JITTER_PX), 2),
        "height": round(b["height"] + random.uniform(-JITTER_PX, JITTER_PX), 2),
        "type": b["type"],
        "confidence": round(random.uniform(0.55, 0.85), 3),
    }
    if b["type"] in MULTIROW_TYPES and b.get("rows"):
        det["rows"] = 1  # merges all rows into one box -- classic weak-detector failure
    else:
        det["rows"] = b.get("rows")
    return det


def main():
    pages = {}
    for fname in sorted(os.listdir(GT_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(GT_DIR, fname)) as f:
            gt = json.load(f)

        detections = []
        for b in gt["blanks"]:
            if b["type"] in SMALL_TYPES and random.random() < MISS_SMALL_PROB:
                continue
            detections.append(perturb_box(b))

        if random.random() < HALLUCINATE_PROB:
            detections.append({
                "x": round(random.uniform(50, gt["page_width"] - 100), 1),
                "y": round(random.uniform(50, gt["page_height"] - 60), 1),
                "width": round(random.uniform(60, 150), 1),
                "height": round(random.uniform(14, 30), 1),
                "type": "single_line", "confidence": round(random.uniform(0.4, 0.6), 2),
                "rows": None,
            })

        pages[gt["id"]] = {
            "detections": detections,
            "detection_time_ms": round(random.uniform(*TIME_MS_RANGE), 1),
            "failed": False,
        }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "tool": "mock_baseline",
            "pages": pages,
            "system_info": {"note": "synthetic demo detector, not a real tool"},
        }, f, indent=2)
    print(f"Wrote {OUT_PATH} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
