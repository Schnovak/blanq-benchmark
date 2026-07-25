#!/usr/bin/env python3
"""
For each page: identify BlanQ false positives (detections that didn't
match any ground-truth blank) and false negatives (ground-truth blanks
BlanQ missed). Extract the text within 80pt of each mismatch so we can
see what BlanQ is actually finding / missing.

Prints a per-category sample, no writes.
"""
import fitz, os, json, csv, sys
from collections import defaultdict, Counter

REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "dataset", "manifest.csv")
GT_DIR   = os.path.join(REPO, "ground_truth")
DET_FILE = os.path.join(REPO, "results", "blanq", "detections.json")
IOU_T    = 0.5


def iou(a, b):
    ax1, ay1 = a["x"], a["y"]; ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx1, by1 = b["x"], b["y"]; bx2, by2 = bx1 + b["width"], by1 + b["height"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0: return 0.0
    ua = a["width"]*a["height"] + b["width"]*b["height"] - inter
    return inter / ua if ua > 0 else 0.0


def match(gt, det, t=IOU_T):
    matched_gt, matched_det = set(), set()
    for j, d in enumerate(det):
        best, best_iou = -1, t
        for i, g in enumerate(gt):
            if i in matched_gt: continue
            s = iou(d, g)
            if s >= best_iou: best_iou, best = s, i
        if best >= 0:
            matched_gt.add(best); matched_det.add(j)
    fp_idx = [j for j in range(len(det)) if j not in matched_det]
    fn_idx = [i for i in range(len(gt))  if i not in matched_gt]
    return fp_idx, fn_idx


def context_text(page, box, pad=80):
    r = fitz.Rect(box["x"] - pad, box["y"] - pad,
                  box["x"] + box["width"] + pad,
                  box["y"] + box["height"] + pad)
    words = page.get_text("words", clip=r)  # list of (x0,y0,x1,y1,word,...)
    words.sort(key=lambda w: (round(w[1] / 10), w[0]))
    return " ".join(w[4] for w in words)[:200]


def classify(context):
    c = context.lower()
    labels = []
    for kw in ("signature", "sign here", "signed", "signer"):
        if kw in c: labels.append("signature"); break
    for kw in ("date:", "mm/dd", "day of", "date of", "date signed"):
        if kw in c: labels.append("date"); break
    for kw in ("name:", "first name", "last name", "middle name",
               "full name", "print name", "your name", "debtor", "petitioner"):
        if kw in c: labels.append("name"); break
    for kw in ("phone", "telephone", "tel:", "fax:", "email", "e-mail"):
        if kw in c: labels.append("contact"); break
    for kw in ("address", "street", "city", "state", "zip"):
        if kw in c: labels.append("address"); break
    for kw in ("ssn", "social security", "ein", "case number", "case no",
               "account number", "id number", "tin"):
        if kw in c: labels.append("id_number"); break
    for kw in ("amount", "$", "total", "subtotal", "balance"):
        if kw in c: labels.append("amount"); break
    return labels or ["unlabeled"]


def main():
    show = sys.argv[1] if len(sys.argv) > 1 else None  # optional category filter
    with open(MANIFEST) as f:
        manifest = {r["id"]: r for r in csv.DictReader(f)}
    with open(DET_FILE) as f:
        dets = json.load(f)["pages"]

    fp_labels = defaultdict(Counter)  # category → label counter
    fn_labels = defaultdict(Counter)
    fp_samples = defaultdict(list)
    fn_samples = defaultdict(list)

    for rid, row in manifest.items():
        gt_path = os.path.join(GT_DIR, f"{rid}.json")
        if not os.path.exists(gt_path): continue
        with open(gt_path) as f:
            gt = json.load(f)
        det_page = dets.get(rid, {"detections": []})["detections"]
        fp_idx, fn_idx = match(gt["blanks"], det_page)
        if not fp_idx and not fn_idx: continue

        cat = row["category"]
        gt_src = gt.get("source", "manual")
        pdf_path = os.path.join(REPO, "dataset", row["file"])
        doc = fitz.open(pdf_path)
        page = doc[0]

        for j in fp_idx:
            ctx = context_text(page, det_page[j])
            for lab in classify(ctx):
                fp_labels[cat][lab] += 1
            if len(fp_samples[cat]) < 6:
                fp_samples[cat].append((rid, gt_src, det_page[j], ctx))

        for i in fn_idx:
            ctx = context_text(page, gt["blanks"][i])
            for lab in classify(ctx):
                fn_labels[cat][lab] += 1
            if len(fn_samples[cat]) < 6:
                fn_samples[cat].append((rid, gt_src, gt["blanks"][i], ctx))

        doc.close()

    cats = [show] if show else sorted(set(fp_labels) | set(fn_labels))
    for cat in cats:
        print(f"\n════════ {cat} ════════")
        fps = sum(fp_labels[cat].values())
        fns = sum(fn_labels[cat].values())
        print(f"FALSE POSITIVES (BlanQ detected, no matching GT): {sum(fp_labels[cat].values())} label-hits")
        for lab, n in fp_labels[cat].most_common():
            print(f"   {lab:12s} {n:5d} ({100*n/fps:.0f}%)" if fps else "")
        print(f"\nFALSE NEGATIVES (BlanQ missed a GT blank): {sum(fn_labels[cat].values())} label-hits")
        for lab, n in fn_labels[cat].most_common():
            print(f"   {lab:12s} {n:5d} ({100*n/fns:.0f}%)" if fns else "")

        print("\n— Sample FALSE POSITIVES —")
        for rid, src, box, ctx in fp_samples[cat][:4]:
            print(f"   [{rid} src={src}] box=({box['x']:.0f},{box['y']:.0f},{box['width']:.0f}×{box['height']:.0f})")
            print(f"      ctx: {ctx[:160]!r}")
        print("\n— Sample FALSE NEGATIVES —")
        for rid, src, box, ctx in fn_samples[cat][:4]:
            print(f"   [{rid} src={src}] box=({box['x']:.0f},{box['y']:.0f},{box['width']:.0f}×{box['height']:.0f})")
            print(f"      ctx: {ctx[:160]!r}")


if __name__ == "__main__":
    main()
