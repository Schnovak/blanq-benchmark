#!/usr/bin/env python3
"""
LLM competitor: Google Gemini 2.5 Flash Lite.

Renders each dataset page to a PNG, sends it to Gemini with a prompt
asking for the pixel-space bounding boxes of every fill-in-the-blank
region on the page, and converts the response back to PDF points.

Needs GEMINI_API_KEY in the environment. Costs roughly $0.03 per full
benchmark run (140 pages, ~1.5k input tokens each).
"""
import base64, csv, fitz, io, json, os, re, sys, time, urllib.request

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "dataset", "manifest.csv")
GT_DIR   = os.path.join(ROOT, "ground_truth")
OUT_PATH = os.path.join(ROOT, "results", "gemini_flash_lite", "detections.json")

MODEL      = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
API_KEY    = os.environ.get("GEMINI_API_KEY")
RENDER_DPI = 150       # image resolution sent to the model
BATCH_SAVE = 10        # save partial output every N pages

if not API_KEY:
    sys.exit("Set GEMINI_API_KEY in the environment.")

ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

PROMPT = """Detect every FILLABLE BLANK on this form page. A "blank" is an empty region where a person will write text with a pen or type an answer.

Do NOT include: printed labels ("Name:", "1.", "Complete:"), instructional text, headings, checkboxes, or decorative underlines beneath headings.
Do INCLUDE: horizontal drawn lines waiting for writing, empty rectangular boxes, empty space after labels like "Name:" or a numbered question, signature lines.

Example: on a line reading "Name: ______________", the blank is the underscored area, NOT the word "Name:".

Return ONLY a JSON array. Each element is exactly:
  {"box_2d": [ymin, xmin, ymax, xmax]}
using Gemini's native normalized 0-1000 coordinates from top-left. If there are no blanks, return [].
"""


def render_page_png(pdf_path, dpi=RENDER_DPI):
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    png = pix.tobytes("png")
    pw_pts, ph_pts = page.rect.width, page.rect.height
    doc.close()
    return png, pix.width, pix.height, pw_pts, ph_pts


def call_gemini(png_bytes, image_w, image_h, timeout=90):
    body = {
        "contents": [{
            "parts": [
                {"text": PROMPT},
                {"inline_data": {"mime_type": "image/png",
                                 "data": base64.b64encode(png_bytes).decode()}},
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0,
            "maxOutputTokens": 8192,
        }
    }
    req = urllib.request.Request(
        ENDPOINT, method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(body).encode())
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    ms = int((time.time() - t0) * 1000)
    parts = resp.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        boxes = json.loads(text) if text else []
    except json.JSONDecodeError:
        # Salvage any complete {"box_2d": [...]} entries — response was truncated
        boxes = []
        for m in re.finditer(r'\{"box_2d":\s*\[([^\]]+)\]', text):
            try:
                nums = [int(x) for x in m.group(1).split(",")]
                if len(nums) == 4:
                    boxes.append({"box_2d": nums})
            except ValueError:
                continue
    return boxes, ms


def to_pdf_coords(box, image_w, image_h, pw_pts, ph_pts):
    """
    Gemini returns bounding boxes in two shapes depending on the prompt:
      1. {"x", "y", "width", "height"} in pixels (what we asked for)
      2. {"box_2d": [y0, x0, y1, x1]} normalized to 0-1000 (its default)
    Accept either.
    """
    if "box_2d" in box:
        y0, x0, y1, x1 = box["box_2d"]
        x_px  = (x0 / 1000) * image_w
        y_px  = (y0 / 1000) * image_h
        w_px  = ((x1 - x0) / 1000) * image_w
        h_px  = ((y1 - y0) / 1000) * image_h
    else:
        x_px, y_px = float(box["x"]), float(box["y"])
        w_px, h_px = float(box["width"]), float(box["height"])
    sx = pw_pts / image_w
    sy = ph_pts / image_h
    w_pt, h_pt = w_px * sx, h_px * sy
    return {
        "x": round(x_px * sx, 2),
        "y": round(y_px * sy, 2),
        "width": round(w_pt, 2),
        "height": round(h_pt, 2),
        "type": "multi_line" if h_pt > 22 else "single_line",
        "confidence": round(float(box.get("confidence", 0.9)), 3),
        "rows": max(1, round(h_pt / 15)) if h_pt > 22 else None,
    }


def main():
    with open(MANIFEST) as f:
        manifest = {r["id"]: r for r in csv.DictReader(f)}
    gt_ids = {f[:-5] for f in os.listdir(GT_DIR) if f.endswith(".json")}

    data = {"tool": "gemini_flash_lite",
            "pages": {},
            "system_info": {"model": MODEL, "render_dpi": RENDER_DPI}}
    if os.path.exists(OUT_PATH) and "--refresh" not in sys.argv:
        with open(OUT_PATH) as f:
            data = json.load(f)

    todo = sorted(rid for rid in gt_ids if rid not in data["pages"])
    print(f"{len(todo)} pages to process (of {len(gt_ids)} with ground truth)")
    if not todo:
        return

    for i, rid in enumerate(todo, 1):
        row = manifest.get(rid)
        if not row:
            continue
        pdf = os.path.join(ROOT, "dataset", row["file"])
        try:
            png, iw, ih, pw, ph = render_page_png(pdf)
            raw_boxes, ms = call_gemini(png, iw, ih)
            valid = [b for b in raw_boxes
                     if isinstance(b, dict) and (
                         "box_2d" in b or
                         all(k in b for k in ("x", "y", "width", "height")))]
            dets = [to_pdf_coords(b, iw, ih, pw, ph) for b in valid]
            data["pages"][rid] = {"detections": dets, "detection_time_ms": ms, "failed": False}
            print(f"  [{i}/{len(todo)}] {rid} {len(dets)} blanks {ms}ms")
        except Exception as e:
            data["pages"][rid] = {"detections": [], "detection_time_ms": 0,
                                  "failed": True, "error": str(e)}
            print(f"  [{i}/{len(todo)}] {rid} ERR {e}")
        if i % BATCH_SAVE == 0:
            os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
            with open(OUT_PATH, "w") as f:
                json.dump(data, f, indent=2)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
