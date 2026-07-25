#!/usr/bin/env python3
"""
Baseline detector that uses only visible cues: underscore sequences in
the text stream and horizontal drawn lines in the vector graphics.

This is the "what a Python script could do in 20 lines" answer. It does
not know about form widgets, does not look at pixels, and does not know
what a signature line looks like. Useful as a stand-in for the many
tools that claim "PDF form detection" but really just find underscores.

Rules:
  underscores  → a run of 3+ underscore characters, with the bounding box
                 of the first and last character stitched together
  drawn lines  → any horizontal path shorter than 3pt tall and longer
                 than 30pt wide, with reasonable width limits
"""
import csv, fitz, json, os, re, time

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "dataset", "manifest.csv")
GT_DIR   = os.path.join(ROOT, "ground_truth")
OUT_PATH = os.path.join(ROOT, "results", "pymupdf_naive", "detections.json")

UNDERSCORE_RUN = re.compile(r"_{3,}")


def find_underscore_boxes(page):
    """
    Find every run of 3+ underscores in the text stream and return an
    axis-aligned box roughly covering that run. Uses page.get_text("dict")
    so we can read span-level bbox and character positions.
    """
    out = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if "___" not in text:
                    continue
                span_bbox = span["bbox"]
                span_w = span_bbox[2] - span_bbox[0]
                if not text or span_w <= 0:
                    continue
                per_char = span_w / len(text)
                for m in UNDERSCORE_RUN.finditer(text):
                    x0 = span_bbox[0] + m.start() * per_char
                    x1 = span_bbox[0] + m.end()   * per_char
                    y0, y1 = span_bbox[1], span_bbox[3]
                    w, h = x1 - x0, y1 - y0
                    if w < 15:
                        continue
                    out.append({
                        "x": round(x0, 2), "y": round(y0, 2),
                        "width": round(w, 2), "height": round(h, 2),
                        "type": "single_line", "confidence": 0.75, "rows": None,
                    })
    return out


def find_drawn_line_boxes(page):
    """Horizontal drawn lines with fill-in-line proportions."""
    out = []
    for path in page.get_drawings():
        r = path["rect"]
        w, h = r.width, r.height
        if h > 3 or w < 30 or w > 500:
            continue
        # Skip framing lines that are the full page width
        page_w = page.rect.width
        if w > 0.85 * page_w:
            continue
        out.append({
            "x": round(r.x0, 2), "y": round(r.y0 - 12, 2),
            "width": round(w, 2), "height": round(14, 2),
            "type": "single_line", "confidence": 0.65, "rows": None,
        })
    return out


def dedupe(boxes, iou_min=0.5):
    """Keep the higher-confidence box among overlapping duplicates."""
    kept = []
    for b in sorted(boxes, key=lambda x: -x["confidence"]):
        keep = True
        for k in kept:
            ix1, iy1 = max(b["x"], k["x"]), max(b["y"], k["y"])
            ix2 = min(b["x"] + b["width"], k["x"] + k["width"])
            iy2 = min(b["y"] + b["height"], k["y"] + k["height"])
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter == 0:
                continue
            ua = b["width"] * b["height"] + k["width"] * k["height"] - inter
            if ua and inter / ua >= iou_min:
                keep = False
                break
        if keep:
            kept.append(b)
    return kept


def detect_page(pdf_path):
    t0 = time.time()
    doc = fitz.open(pdf_path)
    page = doc[0]
    boxes = find_underscore_boxes(page) + find_drawn_line_boxes(page)
    doc.close()
    return dedupe(boxes), int((time.time() - t0) * 1000)


def main():
    with open(MANIFEST) as f:
        manifest = {r["id"]: r for r in csv.DictReader(f)}
    gt_ids = {f[:-5] for f in os.listdir(GT_DIR) if f.endswith(".json")}
    pages = {}
    for rid in gt_ids:
        row = manifest.get(rid)
        if not row:
            continue
        pdf = os.path.join(ROOT, "dataset", row["file"])
        try:
            dets, ms = detect_page(pdf)
            pages[rid] = {"detections": dets, "detection_time_ms": ms, "failed": False}
        except Exception as e:
            pages[rid] = {"detections": [], "detection_time_ms": 0, "failed": True, "error": str(e)}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "tool": "pymupdf_naive",
            "pages": pages,
            "system_info": {"note": "underscore runs + drawn horizontal lines, no ML"},
        }, f, indent=2)
    print(f"Wrote {OUT_PATH} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
