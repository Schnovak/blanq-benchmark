#!/usr/bin/env python3
"""
Run BlanQ's detect API on every page that has ground truth. Writes
results/blanq/detections.json. Skips pages already present in that file
unless --refresh is passed.
"""
import os, sys, csv, json, time, urllib.request, fitz

REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "dataset", "manifest.csv")
GT_DIR   = os.path.join(REPO, "ground_truth")
OUT      = os.path.join(REPO, "results", "blanq", "detections.json")
API_URL  = os.environ.get("BLANQ_API_URL", "http://172.25.0.5:8000/process-pdf")


def call(pdf_path):
    with open(pdf_path, "rb") as f:
        data = f.read()
    boundary = b"----BlanqBound"
    body = (b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="file"; filename="p.pdf"\r\n'
            b"Content-Type: application/pdf\r\n\r\n"
            + data + b"\r\n--" + boundary + b"--\r\n")
    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
        method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=90) as r:
        result = json.loads(r.read())
    return result, int((time.time() - t0) * 1000)


def blanks_to_dets(page_data, pdf_path):
    doc = fitz.open(pdf_path)
    pw, ph = doc[0].rect.width, doc[0].rect.height
    doc.close()
    sx = page_data["canvasW"] / pw
    sy = page_data["canvasH"] / ph
    out = []
    for b in page_data.get("blanks", []):
        rows = len(b.get("mergedHeights", [1]))
        out.append({
            "x": round(b["x"] / sx, 2),
            "y": round(b["y"] / sy, 2),
            "width": round(b["width"] / sx, 2),
            "height": round(b["height"] / sy, 2),
            "type": "multi_line" if rows > 1 else "single_line",
            "confidence": round(b.get("confidence", 1.0), 4),
            "rows": rows if rows > 1 else None,
        })
    return out


def main():
    refresh = "--refresh" in sys.argv
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))
    gt_ids = {f[:-5] for f in os.listdir(GT_DIR) if f.endswith(".json")}

    data = {"tool": "blanq", "pages": {}, "system_info": {"source": "blanq-ai-detect"}}
    if os.path.exists(OUT):
        with open(OUT) as f:
            data = json.load(f)

    todo = [r for r in rows if r["id"] in gt_ids and (refresh or r["id"] not in data["pages"])]
    print(f"{len(todo)} pages to process (of {len(gt_ids)} with ground truth)")

    for i, r in enumerate(todo, 1):
        rid = r["id"]
        pdf = os.path.join(REPO, "dataset", r["file"])
        try:
            result, ms = call(pdf)
            if not result.get("ok") or not result.get("pages"):
                data["pages"][rid] = {"detections": [], "detection_time_ms": ms, "failed": True}
                print(f"  [{i}/{len(todo)}] {rid} FAIL {ms}ms")
                continue
            dets = blanks_to_dets(result["pages"][0], pdf)
            data["pages"][rid] = {"detections": dets, "detection_time_ms": ms, "failed": False}
            print(f"  [{i}/{len(todo)}] {rid} OK {len(dets)} blanks {ms}ms")
        except Exception as e:
            data["pages"][rid] = {"detections": [], "detection_time_ms": 0, "failed": True, "error": str(e)}
            print(f"  [{i}/{len(todo)}] {rid} ERR {e}")

        if i % 10 == 0:
            with open(OUT, "w") as f:
                json.dump(data, f, indent=2)

    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
