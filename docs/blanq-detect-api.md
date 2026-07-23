# BlanQ Detect API

Internal HTTP service running in `blanq-ai-detect` container at `172.25.0.5:8000`.
Not exposed to the host directly — call from within the Docker network or via the
nginx proxy at `/api/detect` on blanq.izum.ch / blanqdev.izum.ch.

## Endpoint

```
POST http://172.25.0.5:8000/process-pdf
Content-Type: multipart/form-data
field: file  (the PDF, any number of pages)
```

## Response

```json
{
  "ok": true,
  "pageCount": 1,
  "pages": [
    {
      "page": 1,
      "canvasW": 1415,
      "canvasH": 2000,
      "blankCount": 65,
      "blanks": [ ... ],
      "badgeJpeg": "<base64-encoded JPEG string>"
    }
  ]
}
```

### `blanks[]` — detected regions

Each blank is in **canvas pixel coordinates** (`canvasW × canvasH`), origin top-left.

| Field           | Type    | Description |
|----------------|---------|-------------|
| `x`            | float   | Left edge in canvas pixels |
| `y`            | float   | Top edge in canvas pixels |
| `width`        | float   | Width in canvas pixels |
| `height`       | float   | Height in canvas pixels |
| `confidence`   | float   | Model confidence 0–1 |
| `type`         | string  | Always `"TextBox"` for now |
| `n`            | int     | Sequential blank index (1-based) |
| `mergedHeights`| float[] | Height of each merged row. `len()` = number of lines in a multiline blank |
| `answer`       | string  | Pre-filled answer if any (empty string for blanks) |
| `page`         | int     | 1-based page number |
| `canvasW`      | int     | Canvas width (same as parent) |
| `canvasH`      | int     | Canvas height (same as parent) |

### `badgeJpeg` — pre-rendered overlay image

Base64-encoded JPEG. The model's internal visualisation with **circles** marking
detected regions. Good for quick human review of detection coverage.

Use as: `<img src="data:image/jpeg;base64,{badgeJpeg}">`

---

## Converting canvas → PDF points

The API renders PDFs at a fixed canvas size. To convert detections to PDF point
coordinates (the benchmark's ground truth format, 72pt/inch, origin top-left):

```python
import fitz

doc = fitz.open("page.pdf")
rect = doc[0].rect
pw, ph = rect.width, rect.height   # PDF points (e.g. 595.3 × 841.9 for A4)
doc.close()

canvas_w = page_data["canvasW"]   # e.g. 1415
canvas_h = page_data["canvasH"]   # e.g. 2000

sx = canvas_w / pw   # pixels per point  (~2.376 for A4)
sy = canvas_h / ph

for blank in page_data["blanks"]:
    x_pt     = blank["x"]      / sx
    y_pt     = blank["y"]      / sy
    width_pt = blank["width"]  / sx
    height_pt= blank["height"] / sy
    rows     = len(blank["mergedHeights"])  # 1 = single_line, >1 = multi_line
```

---

## Rendering a rectangle overlay with PyMuPDF

Draws green bounding boxes on the rendered page — cleaner than the badge circles
for human review of field-level detection accuracy:

```python
import fitz, base64

def render_with_overlay(pdf_path, page_data, scale=1.5):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pw, ph = page.rect.width, page.rect.height
    sx = page_data["canvasW"] / pw
    sy = page_data["canvasH"] / ph

    for blank in page_data["blanks"]:
        x  = blank["x"]      / sx
        y  = blank["y"]      / sy
        x2 = x + blank["width"]  / sx
        y2 = y + blank["height"] / sy
        rect = fitz.Rect(x, y, x2, y2)
        page.draw_rect(rect,
                       color=(0.0, 0.65, 0.25),   # green border
                       fill=(0.0, 0.65, 0.25),
                       fill_opacity=0.25,
                       width=1.5)

    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    doc.close()
    return base64.b64encode(pix.tobytes("png")).decode()
```

---

## Quick Python example (from host)

```python
import urllib.request, json

def detect(pdf_path):
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    boundary = b"----BlanqBoundary"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="page.pdf"\r\n'
        b"Content-Type: application/pdf\r\n\r\n"
        + pdf_bytes + b"\r\n"
        b"--" + boundary + b"--\r\n"
    )
    req = urllib.request.Request(
        "http://172.25.0.5:8000/process-pdf",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

result = detect("myform.pdf")
page = result["pages"][0]
print(f"{page['blankCount']} blanks detected")

# Badge JPEG (circles, AI-style visualisation)
with open("badge.jpg", "wb") as f:
    import base64
    f.write(base64.b64decode(page["badgeJpeg"]))
```
