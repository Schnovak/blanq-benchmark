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

# ── Line-alignment match rule ────────────────────────────────────────────────
# A blank on a form is a horizontal line at a specific y position with a
# specific length. Box height is a convention (some tools draw the stroke
# only, some include the writing area above), so it should NOT count.
# A detection matches a ground-truth blank when its bottom edge is on the
# same line and it covers roughly the same horizontal extent. This is the
# primary matching rule for the benchmark; IoU is kept as a secondary
# tightness score.

LINE_BOTTOM_TOL_PT   = 6.0   # bottom-edge distance in PDF points
LINE_HORIZ_MIN_RATIO = 0.95  # intersection / min(det_len, gt_len)


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


def line_alignment_score(gt, det):
    """
    Returns (matched, bottom_error, horiz_overlap_ratio) where `matched` is
    True iff the detection has its bottom on the same line as the ground-
    truth blank AND covers at least LINE_HORIZ_MIN_RATIO of the shorter box.
    """
    gt_bottom  = gt["y"]  + gt["height"]
    det_bottom = det["y"] + det["height"]
    bottom_err = abs(gt_bottom - det_bottom)

    ix0 = max(gt["x"], det["x"])
    ix1 = min(gt["x"] + gt["width"], det["x"] + det["width"])
    inter_x = max(0.0, ix1 - ix0)
    shorter = max(1e-9, min(gt["width"], det["width"]))
    overlap = inter_x / shorter

    matched = bottom_err <= LINE_BOTTOM_TOL_PT and overlap >= LINE_HORIZ_MIN_RATIO
    return matched, bottom_err, overlap


def greedy_match_line_alignment(gt_blanks, detections):
    """Greedy one-to-one matcher for the line-alignment rule. Ranks candidates
    by (small bottom_error, large horiz overlap) so the tightest pairs get
    matched first."""
    candidates = []
    for gi, g in enumerate(gt_blanks):
        for di, d in enumerate(detections):
            matched, bot_err, overlap = line_alignment_score(g, d)
            if matched:
                # rank key: prefer small bottom_err, then high overlap
                candidates.append((bot_err, -overlap, gi, di, overlap))
    candidates.sort()

    matched_gt, matched_det = set(), set()
    pairs = []
    for _, _, gi, di, overlap in candidates:
        if gi in matched_gt or di in matched_det:
            continue
        matched_gt.add(gi)
        matched_det.add(di)
        gt_bottom = gt_blanks[gi]["y"] + gt_blanks[gi]["height"]
        det_bottom = detections[di]["y"] + detections[di]["height"]
        pairs.append((gi, di, abs(gt_bottom - det_bottom), overlap))

    unmatched_gt  = [i for i in range(len(gt_blanks))  if i not in matched_gt]
    unmatched_det = [i for i in range(len(detections)) if i not in matched_det]
    return pairs, unmatched_gt, unmatched_det


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
    # Line-alignment scoring (primary metric)
    line_align: dict = field(default_factory=dict)       # {tp, fp, fn, precision, recall, f1, mean_bottom_err, mean_overlap}
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

    # Line-alignment scoring (primary metric): a match means "same line, same
    # length" and ignores box height / top-edge convention.
    la_pairs, la_unmatched_gt, la_unmatched_det = greedy_match_line_alignment(gt_blanks, dets)
    la_tp = len(la_pairs)
    la_fp = len(la_unmatched_det)
    la_fn = len(la_unmatched_gt)
    la_p  = la_tp / (la_tp + la_fp) if (la_tp + la_fp) else (1.0 if not dets and not gt_blanks else 0.0)
    la_r  = la_tp / (la_tp + la_fn) if (la_tp + la_fn) else (1.0 if not dets and not gt_blanks else 0.0)
    la_f1 = 2 * la_p * la_r / (la_p + la_r) if (la_p + la_r) else 0.0
    la_bot_errs = [be for _, _, be, _ in la_pairs]
    la_overlaps = [ov for _, _, _, ov in la_pairs]
    score.line_align = {
        "tp": la_tp, "fp": la_fp, "fn": la_fn,
        "precision": la_p, "recall": la_r, "f1": la_f1,
        "mean_bottom_error_pt": statistics.fmean(la_bot_errs) if la_bot_errs else None,
        "mean_horiz_overlap":   statistics.fmean(la_overlaps) if la_overlaps else None,
    }

    return score


def aggregate(page_scores, thresholds=IOU_THRESHOLDS):
    """Roll a list of PageScore into the dataset-level numbers shown on the leaderboard."""
    out = {"n_pages": len(page_scores), "n_gt_blanks": sum(s.n_gt for s in page_scores)}

    # Primary metric: line-alignment (bottom-edge + horizontal-length match).
    la_tp = sum(s.line_align.get("tp", 0) for s in page_scores)
    la_fp = sum(s.line_align.get("fp", 0) for s in page_scores)
    la_fn = sum(s.line_align.get("fn", 0) for s in page_scores)
    la_p  = la_tp / (la_tp + la_fp) if (la_tp + la_fp) else 0.0
    la_r  = la_tp / (la_tp + la_fn) if (la_tp + la_fn) else 0.0
    la_f1 = 2 * la_p * la_r / (la_p + la_r) if (la_p + la_r) else 0.0
    all_bot_errs = [e for s in page_scores for e in ([s.line_align.get("mean_bottom_error_pt")] if s.line_align.get("mean_bottom_error_pt") is not None else [])]
    all_overlaps = [o for s in page_scores for o in ([s.line_align.get("mean_horiz_overlap")]   if s.line_align.get("mean_horiz_overlap")   is not None else [])]
    out["line_alignment"] = {
        "tp": la_tp, "fp": la_fp, "fn": la_fn,
        "precision": round(la_p, 4), "recall": round(la_r, 4), "f1": round(la_f1, 4),
        "mean_bottom_error_pt": round(statistics.fmean(all_bot_errs), 2) if all_bot_errs else None,
        "mean_horiz_overlap":   round(statistics.fmean(all_overlaps), 3) if all_overlaps else None,
        "bottom_tol_pt":        LINE_BOTTOM_TOL_PT,
        "horiz_min_overlap":    LINE_HORIZ_MIN_RATIO,
    }

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
