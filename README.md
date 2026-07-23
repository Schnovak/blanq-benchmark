# blanq-benchmark

**The reference benchmark for PDF fillable-field detection.**

Not "a benchmark." *The* benchmark — public, reproducible, and built so anyone
can re-run every number in it. If you're evaluating tools that find blank
regions in PDFs (checkboxes, signature lines, multi-line answer boxes, table
cells...), this repo is meant to be the place you point to instead of a
marketing page.

> **Answer one question:** which tool detects fillable regions in arbitrary
> PDFs most accurately, fastest, and most reliably?
>
> Not OCR. Not "AI answering the form." Only blank detection.

## Status: v0.1 — seed

This repo is intentionally small right now: **21 synthetic pages (8 clean
born-digital forms across 6 categories + 13 Phase-2 condition variants —
rotation at 1°/3°/7°, bad lighting, coffee stains, shadow, folds, wrinkles,
JPEG artifacts, fax quality, low/high res, phone photo), 261 ground-truth
blanks, 2 example detectors**, wired end to end so the whole pipeline
(dataset → ground truth → detector → scoring → leaderboard) already runs and
produces real numbers — including the harder Phase 2 conditions, where
rotated ground-truth boxes are recomputed exactly (axis-aligned bbox of the
rotated corners), not just copied. See the live leaderboard in
[`docs/index.html`](docs/index.html) (open it directly, or serve `docs/` with
GitHub Pages).

Growing this to the target scale (500–1000 real, sourced/scanned pages across
7 categories) is the roadmap below — see [CONTRIBUTING.md](CONTRIBUTING.md)
for how to add pages.

## Why "pages," not "PDFs"

500 different 1-page PDFs are much less useful than 500 pages pulled from a
smaller number of real, messy, multi-page documents. Diversity of *layout and
condition* is what makes a detector's score mean something.

## The two leaderboards

1. **Blank Detection** — "find every place where a human is expected to
   write." This is what's implemented today.
2. **AI Fill** — "given the detected blanks and some context, place the
   correct answer in the correct field." Proves the end-to-end workflow, not
   just the computer vision. Scaffolded in `docs/index.html` but not yet
   populated — see Phase 8 below.

Companies care more about the complete workflow than raw detection, but you
can't prove the workflow is good without first proving the detection is
good. Hence two leaderboards, not one.

## Repo layout

```
dataset/          PDFs, one row per page in dataset/manifest.csv
ground_truth/     One JSON per page: every blank, manually verified, exact bbox
schema/           JSON Schemas for manifest rows and ground-truth files
evaluation/       metrics.py (IoU/precision/recall/F1) + run_eval.py (CLI scorer)
detectors/        Adapters that turn a tool's output into detections.json
                   (includes two synthetic mock detectors so the pipeline has
                   real numbers before any real competitor is wired in)
results/          <tool>/detections.json + <tool>/scores.json per tool
docs/             The GitHub Pages leaderboard site (index.html + leaderboard.json)
scripts/          generate_seed_dataset.py, build_leaderboard.py
```

## Quickstart

```bash
pip install reportlab pymupdf numpy pillow

# 1. Regenerate the v0.1 seed dataset (synthetic, exact ground truth by construction)
python3 scripts/generate_seed_dataset.py

# 1b. Regenerate the Phase-2 condition variants (rotation, scan noise, phone photo, ...)
python3 scripts/generate_condition_variants.py

# 2. Run the two example detectors (stand-ins until real tools are wired in)
python3 detectors/mock_near_perfect.py
python3 detectors/mock_baseline.py

# 3. Score each one against ground truth
python3 evaluation/run_eval.py --detections results/mock_near_perfect/detections.json --out results/mock_near_perfect/scores.json
python3 evaluation/run_eval.py --detections results/mock_baseline/detections.json --out results/mock_baseline/scores.json

# 4. Rebuild the leaderboard site
python3 scripts/build_leaderboard.py
open docs/index.html   # or: python3 -m http.server -d docs
```

To add a real tool (Blanq, SimplePDF, Adobe Acrobat AI, Foxit AI, PDFgear,
your own detector, a research paper's open-source implementation...), write
an adapter under `detectors/` that runs the tool over `dataset/` and emits
`results/<tool>/detections.json` in the format documented in
`evaluation/run_eval.py`, then repeat steps 3–4.

---

## The full plan (roadmap beyond v0.1)

### Phase 1 — The dataset: 500–1000 *pages*, not PDFs

| Category | Share | Examples |
|---|---|---|
| Government forms | 25% | Tax forms, visa/passport applications, customs declarations, social security, healthcare, DMV, insurance claims — sourced from IRS, UK Gov, Canada, Australia, Switzerland, Germany, Austria, Croatia, EU |
| Schools & universities | 20% | Homework sheets, worksheets, exams, fill-in-the-blank, lab reports (math, chemistry, physics, languages) |
| Medical | 15% | Patient intake, medical history, dental, physiotherapy, mental health |
| Banking / insurance | 10% | Mortgage, loan, KYC, account opening, claims |
| HR | 10% | Employment applications, onboarding, vacation requests, timesheets |
| Legal | 10% | Contracts, agreements, signing forms, witness forms |
| Random internet PDFs | 10% | `filetype:pdf application form`, `filetype:pdf worksheet`, `filetype:pdf registration form` — deliberate chaos |

Sourcing candidates: government agencies, universities, open educational
resources, hospitals/clinics (public forms only), insurance companies, banks,
HR template sites, legal template sites, public sample PDFs on GitHub, your
own scanned documents (PII removed), friends' school/university worksheets
(with permission), and synthetic forms generated to cover edge cases the real
sourcing can't reach.

### Phase 2 — Document conditions

Every difficulty a real-world tool has to survive: born-digital (perfect
PDFs), printed-then-scanned, phone photo, crooked scans (1°/3°/7° rotation),
bad lighting, coffee stains, shadows, folded paper, wrinkled paper, JPEG
artifacts, fax quality, very low resolution, very high resolution.

### Phase 3 — Metadata (per page)

`id, pages, country, language, category, scan/digital, dpi, rotation, noise,
difficulty` — difficulty ∈ {easy, medium, hard, nightmare}. Schema:
[`schema/metadata_schema.json`](schema/metadata_schema.json) /
`dataset/manifest.csv`.

### Phase 4 — Ground truth

The most important part. Every blank on every page, annotated exactly (not
approximately): `id, x, y, width, height, type, confidence, rows`. Types:
checkbox, radio, signature, single_line, multi_line, table_cell, date, name,
large_paragraph, tiny_box, circle, underline. Schema:
[`schema/ground_truth_schema.json`](schema/ground_truth_schema.json).

This is the gold standard everything else is measured against.

### Phase 5 — Competitors

Run the same dataset through every tool you can reach: Blanq, SimplePDF,
Adobe Acrobat AI, Foxit AI, PDFgear, any research papers or open-source
detectors. Even where raw detections aren't exposed, compare what's
reasonably comparable.

### Phase 6 — Metrics

Recall, precision, F1, IoU (thresholds 0.5 / 0.75 / 0.9), detection time
(ms/page), memory (RAM/CPU/GPU), failure rate (pages where the algorithm
completely breaks), multiline accuracy (rows detected vs. actual), baseline
error (detected line vs. actual printed line, average pixel error). All
implemented in [`evaluation/metrics.py`](evaluation/metrics.py).

### Phase 7 — Interesting statistics

The marketing gold that falls out of a rigorous benchmark, e.g. "Blanq
detected 99.2% of 18,327 blanks, averaging 0.83s/page at 96% mean IoU;
Competitor X missed 34% of multiline regions; Competitor Y merged adjacent
boxes; Competitor Z hallucinated boxes." These only carry weight if the
dataset and code producing them are public and reproducible — which is the
entire point of this repo.

### Phase 8 — Publish everything

`dataset/`, `ground_truth/`, `evaluation/`, `results/`, and a `docs/`
leaderboard, all on GitHub, all reproducible from a clean checkout. The AI
Fill leaderboard (given detected blanks + context, place the correct answer)
gets built out here once Blank Detection has real competitor coverage.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a page, annotate ground
truth, or submit a detector's results.

## License

Code (`evaluation/`, `detectors/`, `scripts/`) is MIT-licensed — see
[LICENSE](LICENSE). Every page in `dataset/` carries its own `license` field
in `dataset/manifest.csv` (the v0.1 seed pages are synthetic, generated by
this repo's own scripts, and marked `license: synthetic` — free to use); only
add sourced pages whose license permits redistribution, and record that
license per-row.
