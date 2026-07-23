# Contributing to blanq-benchmark

The whole point of this project is that every number in the leaderboard is
reproducible from a clean checkout. That means contributions need to follow
the same schema the seed dataset uses — see `schema/` for the formal JSON
Schemas referenced below.

## Adding a page to the dataset

1. **Source it responsibly.** Only add pages you have the right to
   redistribute (public-domain government forms, your own scanned documents
   with PII removed, open educational resources, synthetic pages you
   generate, etc). Record the real license in the manifest row — don't mark
   something CC-BY if you're not sure it is.
2. **Drop the PDF** under `dataset/<category>/`, one of: `government`,
   `education`, `medical`, `banking_insurance`, `hr`, `legal`,
   `random_internet`. One page per PDF file is easiest for ground truth to
   stay unambiguous — if you have a multi-page source document, split it and
   give each page its own `id`.
3. **Add a row to `dataset/manifest.csv`** with a unique `id` (pattern:
   `<category-prefix>-<short-slug>-<3-digit-number>`, e.g.
   `gov-us-w4-002`). All columns are described in
   `schema/metadata_schema.json`. Be honest about `capture`, `dpi`,
   `rotation_deg`, `noise`, and `difficulty` — these are what let the
   leaderboard report accuracy *by condition*, not just in aggregate.
4. **Annotate ground truth exactly.** Create
   `ground_truth/<id>.json` following `schema/ground_truth_schema.json`.
   Coordinates are PDF points (1/72 inch), origin top-left. Every blank a
   human is expected to write in gets one entry — not approximately, exactly.
   If you're annotating by hand, open the PDF in a viewer that reports
   cursor position in points (or write a small script — see
   `scripts/generate_seed_dataset.py` for a fully-scripted example where
   ground truth is emitted alongside the PDF).
5. **Sanity-check your ground truth** by overlaying it on the PDF before
   committing:
   ```python
   import fitz  # pymupdf
   doc = fitz.open(f"dataset/<category>/<id>.pdf")
   page = doc[0]
   import json
   gt = json.load(open(f"ground_truth/<id>.json"))
   for b in gt["blanks"]:
       page.draw_rect(fitz.Rect(b["x"], b["y"], b["x"]+b["width"], b["y"]+b["height"]), color=(1,0,0))
   page.get_pixmap(matrix=fitz.Matrix(2,2)).save("/tmp/check.png")
   ```
   Every box should sit exactly on the field it's meant to mark.
6. Run `python3 scripts/build_leaderboard.py` — it doesn't touch your new
   page directly, but confirms the pipeline still runs cleanly.

## Submitting a detector's results

See [`detectors/README.md`](detectors/README.md) for the full submission
format. Short version: write an adapter that runs the tool over everything in
`dataset/` and writes `results/<tool_name>/detections.json` in the schema
documented in `evaluation/run_eval.py`, then:

```bash
python3 evaluation/run_eval.py --detections results/<tool_name>/detections.json --out results/<tool_name>/scores.json
python3 scripts/build_leaderboard.py
```

If you don't have programmatic access to a tool (e.g. it's a web UI only),
still worth including — run it manually on each page, note the boxes it drew
by hand into a `detections.json`, and flag in your PR description that the
results were captured manually so reviewers know why there's no adapter
script.

## Ground rules

- No PII. If a real-world form has been filled in, redact or use a blank
  template instead.
- Don't inflate a category's page count with near-duplicate pages (e.g. the
  same tax form fetched from five mirrors) — diversity of *layout*, not raw
  count, is what makes the benchmark meaningful.
- If you disagree with an existing ground-truth annotation, open an issue
  with the page id and what you think is wrong, rather than silently
  overwriting it — ground truth changes affect every tool's historical
  score, so they should be discussed.
