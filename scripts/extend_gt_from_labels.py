#!/usr/bin/env python3
"""
Extend ground truth with BlanQ detections that:
  1. LOOK like fill-in blanks — elongated horizontal boxes, not checkboxes
     (min width 40pt AND aspect ratio >= 2.5:1, OR multi-line with height > 22)
  2. have a clear text label nearby (signature/name/date/address/id/amount/contact)
  3. do NOT overlap any existing GT box (guards against granularity double-count)
  4. do NOT overlap another about-to-be-added detection (dedup)

Only touches pages where GT source is 'acroform' — manual GT is authoritative.
Never touches education (already manual, near-perfect).

Runs dry by default. Pass --apply to actually write.
"""
import fitz, os, json, csv, sys
from collections import Counter

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


def iou(a, b):
    ax1, ay1 = a["x"], a["y"]; ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx1, by1 = b["x"], b["y"]; bx2, by2 = bx1 + b["width"], by1 + b["height"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def any_overlap(box, others):
    """Any positive area overlap with any of `others`."""
    return any(iou(box, o) > 0 for o in others)


def context_words(page, box, pad=80):
    r = fitz.Rect(box["x"] - pad, box["y"] - pad,
                  box["x"] + box["width"] + pad,
                  box["y"] + box["height"] + pad)
    return page.get_text(clip=r).lower()


def label(context):
    for lab, kws in LABEL_KEYWORDS.items():
        if any(kw in context for kw in kws):
            return lab
    return None


def looks_like_blank(box):
    """Elongated horizontal box, not a checkbox / tick / dot."""
    w, h = box["width"], box["height"]
    if h > 22:            # multi-line: any reasonable width is fine
        return w >= 30
    if w < 40:            # too short to be a fill-in line
        return False
    return (w / h) >= 2.5  # aspect ratio filter — excludes squares/checkboxes


def main():
    apply = "--apply" in sys.argv
    with open(MANIFEST) as f:
        manifest = {r["id"]: r for r in csv.DictReader(f)}
    with open(DET_FILE) as f:
        det_pages = json.load(f)["pages"]

    per_cat = Counter()
    per_cat_pages = Counter()
    per_label = Counter()
    write_plan = []  # list of (rid, new_blanks_list, updated_gt_dict)

    for rid, row in manifest.items():
        gt_path = os.path.join(GT_DIR, f"{rid}.json")
        if not os.path.exists(gt_path):
            continue
        with open(gt_path) as f:
            gt = json.load(f)
        # only extend AcroForm-derived pages
        if gt.get("source") != "acroform":
            continue

        det_page = det_pages.get(rid, {"detections": []})["detections"]
        if not det_page:
            continue

        existing = list(gt["blanks"])
        added = []
        pdf_path = os.path.join(REPO, "dataset", row["file"])
        doc = fitz.open(pdf_path)
        page = doc[0]

        for d in det_page:
            # Rule 1 — must look like a fill-in blank, not a checkbox
            if not looks_like_blank(d):
                continue
            # Rule 3 — skip if overlaps ANY existing GT
            if any_overlap(d, existing):
                continue
            # Rule 4 — skip if overlaps a pending addition
            if any_overlap(d, added):
                continue
            # Rule 2 — must have a clear label
            ctx = context_words(page, d)
            lab = label(ctx)
            if lab is None:
                continue
            added.append({
                "x": d["x"], "y": d["y"], "width": d["width"], "height": d["height"],
                "type": d.get("type", "single_line"),
                "confidence": 1.0,
                "rows": d.get("rows"),
                "label_hint": lab,
                "source_extended": "blanq-labeled",
            })
            per_label[lab] += 1

        doc.close()

        if added:
            cat = row["category"]
            per_cat[cat] += len(added)
            per_cat_pages[cat] += 1
            merged = existing + added
            for i, b in enumerate(merged, 1):
                b["id"] = f"b{i:03d}"
            new_gt = dict(gt)
            new_gt["blanks"] = merged
            new_gt["source"] = "acroform+labeled_blanq"
            write_plan.append((rid, added, new_gt))

    total_add = sum(per_cat.values())
    print(f"Would add {total_add} labeled blanks across {sum(per_cat_pages.values())} pages")
    print("\nPer category (new blanks / pages touched):")
    for cat in sorted(per_cat):
        print(f"  {cat:22s} {per_cat[cat]:5d} on {per_cat_pages[cat]:4d} pages")
    print("\nPer label:")
    for lab, n in per_label.most_common():
        print(f"  {lab:12s} {n:5d}")

    if apply:
        for rid, added, new_gt in write_plan:
            with open(os.path.join(GT_DIR, f"{rid}.json"), "w") as f:
                json.dump(new_gt, f, indent=2)
        print(f"\nWrote {len(write_plan)} updated ground truth files.")
    else:
        print("\nDry run — pass --apply to write changes.")


if __name__ == "__main__":
    main()
