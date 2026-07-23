#!/usr/bin/env python3
"""
Phase 2 of the benchmark spec: "different document types" -- the same content,
degraded the way real-world capture degrades it (scan skew, phone photo,
coffee stains, fax quality, low/high resolution, JPEG artifacts...).

This script takes the born-digital v0.1 seed pages (from
generate_seed_dataset.py) and produces one raster-image PDF per condition,
recomputing ground truth exactly for every pixel-level transform applied
(most importantly: rotation, where each blank's axis-aligned bbox is
recomputed from the rotated corners, not just copied).

Must be run AFTER generate_seed_dataset.py (it reads dataset/manifest.csv +
ground_truth/*.json for the base pages it degrades).

Run:
  python3 scripts/generate_condition_variants.py
"""
import csv
import io
import json
import math
import os
import random

import fitz  # pymupdf
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "dataset")
GT_DIR = os.path.join(ROOT, "ground_truth")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")

BASE_RENDER_DPI = 150.0
MANIFEST_COLUMNS = [
    "id", "file", "category", "country", "language", "source", "pages",
    "capture", "dpi", "rotation_deg", "noise", "difficulty", "license",
]

random.seed(2026)
np.random.seed(2026)

# Each condition: (noise_label, difficulty, capture, rotation_deg_or_0, apply_fn)
# apply_fn(img: PIL.Image, rng) -> PIL.Image  (geometry-preserving edits only;
# rotation is handled centrally so ground truth stays exact)


def _scan_texture(img, grain=6, blur=0.6):
    img = img.convert("RGB")
    arr = np.asarray(img).astype(np.int16)
    noise = np.random.normal(0, grain, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    return out


def cond_bad_lighting(img, rng):
    img = img.convert("RGB")
    w, h = img.size
    grad = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(grad)
    # darken diagonally from one corner
    for x in range(0, w, 4):
        frac = x / w
        d.rectangle([x, 0, x + 4, h], fill=int(90 * frac))
    grad = grad.filter(ImageFilter.GaussianBlur(40))
    dark = Image.new("RGB", img.size, (0, 0, 0))
    return Image.composite(dark, img, grad.point(lambda v: 255 - v))


def cond_coffee_stain(img, rng):
    img = img.convert("RGB")
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    cx, cy = rng.randint(int(w * 0.2), int(w * 0.8)), rng.randint(int(h * 0.2), int(h * 0.8))
    r = rng.randint(int(min(w, h) * 0.07), int(min(w, h) * 0.14))
    for i in range(6):
        rr = r - i * (r // 7)
        alpha = 40 + i * 12
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                  outline=(101, 67, 33, alpha), width=max(1, r // 18))
    d.ellipse([cx - r + r // 6, cy - r + r // 6, cx + r - r // 6, cy + r - r // 6],
               fill=(139, 94, 43, 30))
    overlay = overlay.filter(ImageFilter.GaussianBlur(2))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def cond_shadow(img, rng):
    img = img.convert("RGB")
    w, h = img.size
    grad = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(grad)
    band_w = int(w * 0.22)
    for x in range(band_w):
        v = int(255 * (x / band_w) ** 1.5)
        d.line([(x, 0), (x, h)], fill=v)
    grad = grad.filter(ImageFilter.GaussianBlur(15))
    dark = Image.new("RGB", img.size, (20, 20, 25))
    return Image.composite(img, dark, grad)


def cond_folded(img, rng):
    img = img.convert("RGB")
    w, h = img.size
    fold_y = h // 2 + rng.randint(-20, 20)
    band = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(band)
    d.line([(0, fold_y), (w, fold_y)], fill=140, width=3)
    band = band.filter(ImageFilter.GaussianBlur(6))
    dark = Image.new("RGB", img.size, (60, 60, 60))
    creased = Image.composite(dark, img, band.point(lambda v: 255 - v))
    return _scan_texture(creased, grain=3, blur=0.3)


def cond_wrinkled(img, rng):
    img = img.convert("RGB")
    arr = np.asarray(img).astype(np.int16)
    tex_raw = np.random.normal(0, 14, arr.shape[:2])
    tex_u8 = np.clip(tex_raw + 128, 0, 255).astype(np.uint8)
    tex_blurred = np.asarray(Image.fromarray(tex_u8, mode="L").filter(ImageFilter.GaussianBlur(3)))
    tex = tex_blurred.astype(np.int16)[..., None] - 128
    arr = np.clip(arr + tex, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def cond_jpeg_artifacts(img, rng):
    img = _scan_texture(img.convert("RGB"), grain=4, blur=0.4)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=12)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def cond_low_res(img, rng):
    w, h = img.size
    factor = 150.0 / 55.0  # simulate ~55 dpi capture
    small = img.convert("RGB").resize((max(1, int(w / factor)), max(1, int(h / factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def cond_high_res(img, rng):
    # no distortion -- this variant's realism comes from rendering at a higher
    # source DPI (handled by the caller passing a bigger render_dpi), not a filter
    return _scan_texture(img.convert("RGB"), grain=2, blur=0.2)


def cond_phone_photo(img, rng):
    img = img.convert("RGB")
    w, h = img.size
    vign = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(vign)
    d.ellipse([-w * 0.25, -h * 0.25, w * 1.25, h * 1.25], fill=255)
    vign = vign.filter(ImageFilter.GaussianBlur(80))
    dark = Image.new("RGB", img.size, (10, 10, 10))
    lit = Image.composite(img, dark, vign)
    lit = ImageEnhance.Color(lit).enhance(0.85)
    lit = ImageEnhance.Contrast(lit).enhance(1.08)
    return _scan_texture(lit, grain=5, blur=0.5)


def cond_fax_quality(img, rng):
    gray = img.convert("L")
    bw = gray.point(lambda v: 255 if v > 150 else 0, mode="1").convert("L")
    return bw.convert("RGB")


def cond_rotate_only(img, rng):
    return _scan_texture(img.convert("RGB"), grain=5, blur=0.5)


# id_suffix, condition_fn, rotation_deg, capture, noise_label, difficulty, render_dpi_multiplier, save_fmt
CONDITIONS = [
    ("rot1", cond_rotate_only, 1.3, "printed_scanned", "none", "medium", 1.0, "JPEG"),
    ("rot3", cond_rotate_only, -3.2, "printed_scanned", "none", "hard", 1.0, "JPEG"),
    ("rot7", cond_rotate_only, 6.8, "printed_scanned", "none", "nightmare", 1.0, "JPEG"),
    ("badlight", cond_bad_lighting, 0.0, "printed_scanned", "bad_lighting", "medium", 1.0, "JPEG"),
    ("coffee", cond_coffee_stain, 0.0, "printed_scanned", "coffee_stain", "hard", 1.0, "JPEG"),
    ("shadow", cond_shadow, 0.0, "phone_photo", "shadow", "medium", 1.0, "JPEG"),
    ("folded", cond_folded, 0.0, "printed_scanned", "folded", "hard", 1.0, "JPEG"),
    ("wrinkled", cond_wrinkled, 0.0, "printed_scanned", "wrinkled", "hard", 1.0, "JPEG"),
    ("jpeg", cond_jpeg_artifacts, 0.0, "phone_photo", "jpeg_artifacts", "medium", 1.0, "JPEG"),
    ("faxq", cond_fax_quality, 1.0, "fax", "fax_quality", "nightmare", 0.6, "PNG"),
    ("lowres", cond_low_res, 0.0, "phone_photo", "low_res", "hard", 1.0, "JPEG"),
    ("highres", cond_high_res, 0.0, "printed_scanned", "high_res", "easy", 1.7, "JPEG"),
    ("phonephoto", cond_phone_photo, -2.4, "phone_photo", "none", "medium", 1.0, "JPEG"),
]


def load_base_rows():
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["source"] == "synthetic-seed"]


def rotate_point(dx, dy, theta_rad):
    """Matches PIL's Image.rotate(angle, expand=True) convention -- calibrated
    empirically, see scripts/README or CONTRIBUTING for the derivation."""
    dxp = dx * math.cos(theta_rad) + dy * math.sin(theta_rad)
    dyp = -dx * math.sin(theta_rad) + dy * math.cos(theta_rad)
    return dxp, dyp


def transform_ground_truth(gt, angle_deg, old_px_size, new_px_size, px_to_pt):
    if angle_deg == 0:
        return gt["blanks"]
    theta = math.radians(angle_deg)
    ow, oh = old_px_size
    nw, nh = new_px_size
    ocx, ocy = ow / 2, oh / 2
    ncx, ncy = nw / 2, nh / 2

    new_blanks = []
    for b in gt["blanks"]:
        x_px, y_px = b["x"] / px_to_pt, b["y"] / px_to_pt
        w_px, h_px = b["width"] / px_to_pt, b["height"] / px_to_pt
        corners = [(x_px, y_px), (x_px + w_px, y_px), (x_px, y_px + h_px), (x_px + w_px, y_px + h_px)]
        new_corners = []
        for cx, cy in corners:
            dxp, dyp = rotate_point(cx - ocx, cy - ocy, theta)
            new_corners.append((ncx + dxp, ncy + dyp))
        xs = [c[0] for c in new_corners]
        ys = [c[1] for c in new_corners]
        nb = dict(b)
        nb["x"] = round(min(xs) * px_to_pt, 1)
        nb["y"] = round(min(ys) * px_to_pt, 1)
        nb["width"] = round((max(xs) - min(xs)) * px_to_pt, 1)
        nb["height"] = round((max(ys) - min(ys)) * px_to_pt, 1)
        new_blanks.append(nb)
    return new_blanks


def render_base_page(pdf_path, dpi):
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img, page.rect.width, page.rect.height


def save_variant_pdf(img, page_w_pt, page_h_pt, out_path, fmt="JPEG"):
    """fmt='JPEG' keeps file size sane -- these images carry synthetic grain/noise
    that compresses very poorly as lossless PNG (multi-MB per page). fmt='PNG' is
    used only for the 1-bit fax-quality variant, where PNG is both smaller and
    avoids reintroducing gray fuzz into a deliberately bilevel image."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=88, optimize=True)
    else:
        img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    doc = fitz.open()
    page = doc.new_page(width=page_w_pt, height=page_h_pt)
    page.insert_image(fitz.Rect(0, 0, page_w_pt, page_h_pt), stream=buf.read())
    doc.save(out_path)
    doc.close()


def main():
    base_rows = load_base_rows()
    rng = random.Random(2026)
    variant_rows = []
    total_new_pages = 0

    for i, cond in enumerate(CONDITIONS):
        suffix, fn, angle, capture, noise, difficulty, dpi_mult, save_fmt = cond
        base_row = base_rows[i % len(base_rows)]
        base_id = base_row["id"]
        base_pdf_path = os.path.join(DATASET_DIR, base_row["file"])
        with open(os.path.join(GT_DIR, f"{base_id}.json")) as f:
            base_gt = json.load(f)

        render_dpi = BASE_RENDER_DPI * dpi_mult
        img, page_w_pt, page_h_pt = render_base_page(base_pdf_path, render_dpi)
        old_px_size = img.size

        if angle != 0:
            img = fn(img, rng)
            img = img.rotate(angle, expand=True, fillcolor="white", resample=Image.BICUBIC)
        else:
            img = fn(img, rng)

        new_px_size = img.size
        px_to_pt = 72.0 / render_dpi
        new_blanks = transform_ground_truth(base_gt, angle, old_px_size, new_px_size, px_to_pt)
        new_page_w_pt = new_px_size[0] * px_to_pt
        new_page_h_pt = new_px_size[1] * px_to_pt

        new_id = f"{base_id}-{suffix}"
        rel_path = os.path.join(base_row["category"], f"{new_id}.pdf")
        out_path = os.path.join(DATASET_DIR, rel_path)
        save_variant_pdf(img, new_page_w_pt, new_page_h_pt, out_path, fmt=save_fmt)

        with open(os.path.join(GT_DIR, f"{new_id}.json"), "w") as f:
            json.dump({"id": new_id, "page_width": round(new_page_w_pt, 1),
                       "page_height": round(new_page_h_pt, 1), "blanks": new_blanks}, f, indent=2)

        variant_rows.append({
            "id": new_id, "file": rel_path, "category": base_row["category"],
            "country": base_row["country"], "language": base_row["language"],
            "source": "synthetic-seed-variant", "pages": 1, "capture": capture,
            "dpi": int(render_dpi), "rotation_deg": angle, "noise": noise,
            "difficulty": difficulty, "license": "synthetic",
        })
        total_new_pages += 1
        print(f"  {new_id:35s} <- {base_id:25s} cond={suffix:11s} rot={angle:+.1f}  blanks={len(new_blanks)}")

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in base_rows + variant_rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_COLUMNS})

    print(f"\nAdded {total_new_pages} condition-variant pages. Manifest now has "
          f"{len(base_rows) + len(variant_rows)} total rows.")


if __name__ == "__main__":
    main()
