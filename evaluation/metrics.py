"""
Core scoring functions for the blanq-benchmark blank-detection leaderboard.

A "page score" is computed by matching a detector's output boxes against the
ground-truth boxes for that page (greedy, highest-IoU-first, one-to-one), then
counting true positives / false positives / false negatives at each of three
IoU thresholds (0.5 / 0.75 / 0.9), per Phase 6 of the benchmark spec.

Everything here is pure/stateless so it's easy to unit test and to reuse from
run_eval.py, notebooks, or a future CI check.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

IOU_THRESHOLDS = (0.5, 0.75, 0.9)


def iou(a, b):
    """Intersection-over-union of two boxes given as dicts with x, y, width, height."""
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["width"], a["y"] + a["height"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, a["width"]) * max(0.0, a["height"])
    area_b = max(0.0, b["width"]) * max(0.0, b["height"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(gt_blanks, detections):
    """One-to-one greedy matching by descending IoU. Returns a list of
    (gt_blank, detection, iou) for every pair whose IoU > 0, plus the
    unmatched gt indices and unmatched detection indices (IoU == 0 with
    everything). Thresholding into TP/FP/FN happens later per-threshold so we
    only need to run matching once per page."""
    candidates = []
    for gi, g in enumerate(gt_blanks):
        for di, d in enumerate(detections):
            v = iou(g, d)
            if v > 0:
                candidates.append((v, gi, di))
    candidates.sort(key=lambda t: t[0], reverse=True)

    matched_gt, matched_det = set(), set()
    pairs = []
    for v, gi, di in candidates:
        if gi in matched_gt or di in matched_det:
            continue
        matched_gt.add(gi)
        matched_det.add(di)
        pairs.append((gi, di, v))

    unmatched_gt = [i for i in range(len(gt_blanks)) if i not in matched_gt]
    unmatched_det = [i for i in range(len(detections)) if i not in matched_det]
    return pairs, unmatched_gt, unmatched_det


@dataclass
class PageScore:
    page_id: str
    n_gt: int
    n_det: int
    by_threshold: dict = field(default_factory=dict)  # threshold -> {tp,fp,fn,precision,recall,f1}
    matched_ious: list = field(default_factory=list)
    line_errors_px: list = field(default_factory=list)   # per matched line-like blank
    multiline_rows: list = field(default_factory=list)   # (gt_rows, det_rows) for matched multi-row blanks
    detection_time_ms: float | None = None
    failed: bool = False


LINE_TYPES = {"single_line", "date", "name", "underline"}
MULTI_ROW_TYPES = {"multi_line", "large_paragraph", "table_cell"}


def score_page(gt_page, det_page, thresholds=IOU_THRESHOLDS):
    """gt_page: {'id', 'page_width', 'page_height', 'blanks': [...]}
    det_page: {'detections': [...], 'detection_time_ms': float, 'failed': bool}"""
    gt_blanks = gt_page.get("blanks", [])
    dets = [] if det_page.get("failed") else det_page.get("detections", [])

    pairs, unmatched_gt, unmatched_det = greedy_match(gt_blanks, dets)

    score = PageScore(
        page_id=gt_page["id"], n_gt=len(gt_blanks), n_det=len(dets),
        detection_time_ms=det_page.get("detection_time_ms"),
        failed=bool(det_page.get("failed", False)),
    )
    score.matched_ious = [v for _, _, v in pairs]

    for thr in thresholds:
        tp = sum(1 for _, _, v in pairs if v >= thr)
        # anything not a TP at this threshold is either an unmatched det (FP) or a
        # matched-but-below-threshold pair, which counts as both a missed gt (FN)
        # and a spurious det (FP).
        below_thr_pairs = sum(1 for _, _, v in pairs if v < thr)
        fp = len(unmatched_det) + below_thr_pairs
        fn = len(unmatched_gt) + below_thr_pairs
        precision = tp / (tp + fp) if (tp + fp) else (1.0 if not dets and not gt_blanks else 0.0)
        recall = tp / (tp + fn) if (tp + fn) else (1.0 if not dets and not gt_blanks else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        score.by_threshold[thr] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
                                    "recall": recall, "f1": f1}

    # Baseline / line-position pixel error: for matched line-like blanks, compare
    # the bottom edge (where the ruled line / baseline sits in our convention).
    for gi, di, v in pairs:
        g, d = gt_blanks[gi], dets[di]
        if g["type"] in LINE_TYPES:
            gt_line_y = g["y"] + g["height"]
            det_line_y = d["y"] + d["height"]
            score.line_errors_px.append(abs(gt_line_y - det_line_y))
        if g["type"] in MULTI_ROW_TYPES and g.get("rows") is not None and d.get("rows") is not None:
            score.multiline_rows.append((g["rows"], d["rows"]))

    return score


def aggregate(page_scores, thresholds=IOU_THRESHOLDS):
    """Roll a list of PageScore into the dataset-level numbers shown on the leaderboard."""
    out = {"n_pages": len(page_scores), "n_gt_blanks": sum(s.n_gt for s in page_scores)}

    for thr in thresholds:
        tp = sum(s.by_threshold[thr]["tp"] for s in page_scores)
        fp = sum(s.by_threshold[thr]["fp"] for s in page_scores)
        fn = sum(s.by_threshold[thr]["fn"] for s in page_scores)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[f"iou@{thr}"] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4),
                              "recall": round(recall, 4), "f1": round(f1, 4)}

    all_ious = [v for s in page_scores for v in s.matched_ious]
    out["mean_iou"] = round(statistics.fmean(all_ious), 4) if all_ious else 0.0

    all_errs = [e for s in page_scores for e in s.line_errors_px]
    out["median_line_error_px"] = round(statistics.median(all_errs), 2) if all_errs else None
    out["mean_line_error_px"] = round(statistics.fmean(all_errs), 2) if all_errs else None

    row_pairs = [p for s in page_scores for p in s.multiline_rows]
    if row_pairs:
        diffs = [abs(a - b) for a, b in row_pairs]
        exact = sum(1 for a, b in row_pairs if a == b)
        out["multiline_row_accuracy"] = round(exact / len(row_pairs), 4)
        out["multiline_row_mean_abs_error"] = round(statistics.fmean(diffs), 3)
    else:
        out["multiline_row_accuracy"] = None
        out["multiline_row_mean_abs_error"] = None

    times = [s.detection_time_ms for s in page_scores if s.detection_time_ms is not None]
    out["mean_detection_ms_per_page"] = round(statistics.fmean(times), 2) if times else None

    n_failed = sum(1 for s in page_scores if s.failed)
    out["failure_rate"] = round(n_failed / len(page_scores), 4) if page_scores else 0.0
    out["n_failed_pages"] = n_failed

    detected_of_total = sum(s.by_threshold[0.5]["tp"] for s in page_scores)
    out["pct_blanks_detected_iou50"] = (
        round(100 * detected_of_total / out["n_gt_blanks"], 2) if out["n_gt_blanks"] else 0.0
    )

    return out
