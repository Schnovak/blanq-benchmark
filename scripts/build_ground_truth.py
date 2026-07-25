#!/usr/bin/env python3
"""
Derive ground_truth/<id>.json from AcroForm widget positions in each
dataset PDF.

Ground truth sources, in priority order:
  1. Manual approval (existing ground_truth/<id>.json)   — kept as-is.
  2. Manual flag (existing review_later/<id>.json)       — kept as-is.
  3. AcroForm widgets on page 1 of the PDF               — auto-derived here.
  4. Neither                                             — printed as "no source".

Only Text and ComboBox widgets count as blanks. Checkboxes and buttons
do not — BlanQ's detector targets fill-in-the-blank fields.
"""
import fitz, os, json, csv

REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "dataset", "manifest.csv")
GT_DIR   = os.path.join(REPO, "ground_truth")
RL_DIR   = os.path.join(REPO, "review_later")

BLANK_WIDGET_TYPES = ("Text", "ComboBox")


def widget_to_blank(w, idx):
    r = w.rect
    return {
        "id": f"b{idx:03d}",
        "x": round(r.x0, 2),
        "y": round(r.y0, 2),
        "width": round(r.width, 2),
        "height": round(r.height, 2),
        "type": "multi_line" if r.height > 22 else "single_line",
        "confidence": 1.0,
        "rows": max(1, round(r.height / 15)) if r.height > 22 else None,
    }


def derive(pdf_path, rid):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pw, ph = page.rect.width, page.rect.height
    widgets = [w for w in (page.widgets() or [])
               if w.field_type_string in BLANK_WIDGET_TYPES]
    doc.close()
    if not widgets:
        return None
    blanks = [widget_to_blank(w, i + 1) for i, w in enumerate(widgets)]
    return {
        "id": rid,
        "page_width": round(pw, 2),
        "page_height": round(ph, 2),
        "blanks": blanks,
        "source": "acroform",
    }


def main():
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))

    approved = {f[:-5] for f in os.listdir(GT_DIR) if f.endswith(".json")}
    flagged  = {f[:-5] for f in os.listdir(RL_DIR) if f.endswith(".json")}

    stats = {"manual_gt": 0, "flagged_skipped": 0,
             "auto_derived": 0, "no_source": 0}
    no_source = []

    for r in rows:
        rid = r["id"]
        if rid in approved:
            stats["manual_gt"] += 1
            continue
        if rid in flagged:
            stats["flagged_skipped"] += 1
            continue
        pdf_path = os.path.join(REPO, "dataset", r["file"])
        gt = derive(pdf_path, rid)
        if gt is None:
            stats["no_source"] += 1
            no_source.append(rid)
            continue
        with open(os.path.join(GT_DIR, f"{rid}.json"), "w") as f:
            json.dump(gt, f, indent=2)
        stats["auto_derived"] += 1

    print(json.dumps(stats, indent=2))
    if no_source:
        print(f"\n{len(no_source)} pages with no ground truth source:")
        for s in no_source:
            print(f"  {s}")


if __name__ == "__main__":
    main()
