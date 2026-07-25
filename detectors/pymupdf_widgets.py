#!/usr/bin/env python3
"""
Baseline detector that trusts the source PDF entirely. Scans each PDF for
its embedded AcroForm widgets (Text and ComboBox) and calls those the
blanks.

This is the "if you did nothing clever" answer. It scores well when the
form designer already annotated every field, and it scores zero when the
form was scanned, drawn by hand, or otherwise has no widget metadata.
Useful as a floor for what BlanQ has to beat.
"""
import csv, fitz, json, os, time

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "dataset", "manifest.csv")
GT_DIR   = os.path.join(ROOT, "ground_truth")
OUT_PATH = os.path.join(ROOT, "results", "pymupdf_widgets", "detections.json")

WIDGET_TYPES = ("Text", "ComboBox")


def detect_page(pdf_path):
    t0 = time.time()
    doc = fitz.open(pdf_path)
    page = doc[0]
    out = []
    for w in (page.widgets() or []):
        if w.field_type_string not in WIDGET_TYPES:
            continue
        r = w.rect
        out.append({
            "x": round(r.x0, 2), "y": round(r.y0, 2),
            "width": round(r.width, 2), "height": round(r.height, 2),
            "type": "multi_line" if r.height > 22 else "single_line",
            "confidence": 1.0,
            "rows": max(1, round(r.height / 15)) if r.height > 22 else None,
        })
    doc.close()
    return out, int((time.time() - t0) * 1000)


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
            "tool": "pymupdf_widgets",
            "pages": pages,
            "system_info": {"note": "scans embedded AcroForm Text/ComboBox widgets"},
        }, f, indent=2)
    print(f"Wrote {OUT_PATH} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
