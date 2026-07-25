#!/usr/bin/env python3
"""Show sample proposed additions per category so we can eyeball them."""
import fitz, os, json, csv, sys
from collections import defaultdict

REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "dataset", "manifest.csv")
GT_DIR   = os.path.join(REPO, "ground_truth")
DET_FILE = os.path.join(REPO, "results", "blanq", "detections.json")

LABEL_KEYWORDS = {
    "signature": ("signature", "sign here", "signed", "signer"),
    "date":      ("date:", "mm/dd", "day of", "date of", "date signed"),
    "name":      ("name:", "first name", "last name", "middle name",
                  "full name", "print name", "your name",
                  "petitioner", "debtor", "applicant"),
    "contact":   ("phone", "telephone", "tel:", "fax:", "email", "e-mail"),
    "address":   ("address", "street", "city", "state", "zip"),
    "id_number": ("ssn", "social security", "ein", "case number", "case no",
                  "account number", "id number", "tin"),
    "amount":    ("amount", "total", "subtotal", "balance"),
}


def iou_area(a, b):
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2 = min(a["x"] + a["width"], b["x"] + b["width"])
    iy2 = min(a["y"] + a["height"], b["y"] + b["height"])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def context_words(page, box, pad=80):
    r = fitz.Rect(box["x"] - pad, box["y"] - pad,
                  box["x"] + box["width"] + pad,
                  box["y"] + box["height"] + pad)
    return page.get_text(clip=r).lower()


def label_for(context):
    for lab, kws in LABEL_KEYWORDS.items():
        if any(kw in context for kw in kws):
            return lab
    return None


def looks_like_blank(box):
    w, h = box["width"], box["height"]
    if h > 22:
        return w >= 30
    if w < 40:
        return False
    return (w / h) >= 2.5


def main():
    with open(MANIFEST) as f:
        manifest = {r["id"]: r for r in csv.DictReader(f)}
    with open(DET_FILE) as f:
        det_pages = json.load(f)["pages"]

    samples = defaultdict(list)  # category -> [(rid, det, label, ctx)]
    for rid, row in manifest.items():
        gt_path = os.path.join(GT_DIR, f"{rid}.json")
        if not os.path.exists(gt_path): continue
        with open(gt_path) as f:
            gt = json.load(f)
        if gt.get("source") != "acroform": continue
        det_page = det_pages.get(rid, {"detections": []})["detections"]
        if not det_page: continue
        existing = gt["blanks"]
        added_local = []
        pdf_path = os.path.join(REPO, "dataset", row["file"])
        doc = fitz.open(pdf_path)
        page = doc[0]
        for d in det_page:
            if not looks_like_blank(d): continue
            if any(iou_area(d, e) > 0 for e in existing): continue
            if any(iou_area(d, a) > 0 for a in added_local): continue
            ctx = context_words(page, d)
            lab = label_for(ctx)
            if lab is None: continue
            added_local.append(d)
            if len(samples[row["category"]]) < 8:
                samples[row["category"]].append((rid, d, lab, ctx))
        doc.close()

    for cat in sorted(samples):
        print(f"\n════════ {cat} sample additions ════════")
        for rid, d, lab, ctx in samples[cat]:
            print(f"[{rid}] label={lab} box=({d['x']:.0f},{d['y']:.0f},{d['width']:.0f}×{d['height']:.0f})")
            words = ctx.split()
            trimmed = " ".join(words)[:140]
            print(f"  ctx: {trimmed!r}")


if __name__ == "__main__":
    main()
