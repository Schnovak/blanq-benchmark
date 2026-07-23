# Detectors

This folder holds adapters that turn a real tool's output into the benchmark's
`detections.json` format (see `evaluation/run_eval.py` docstring for the schema).

To submit a real competitor or your own tool: write a small script here that
runs the tool over every PDF in `dataset/`, converts its output boxes into
PDF-point coordinates (top-left origin, matching `schema/ground_truth_schema.json`),
and writes `results/<tool_name>/detections.json`. Then run:

```
python3 evaluation/run_eval.py --detections results/<tool_name>/detections.json --out results/<tool_name>/scores.json
python3 scripts/build_leaderboard.py
```

## Included examples (`mock_*.py`)

These don't call any real detector -- they read `ground_truth/` directly and
perturb it, so the full pipeline (detect -> score -> leaderboard) has real
numbers to show from the very first commit, before any real competitor has
been wired up:

- `mock_near_perfect.py` -- small jitter + occasional dropped/split box, to
  simulate a strong detector. Useful as a sanity-check ceiling.
- `mock_baseline.py` -- coarser boxes, misses small checkboxes, merges
  adjacent multi-line rows, and occasionally hallucinates a box. Simulates a
  weak/naive detector so the leaderboard has visible spread.

Replace both with real tool integrations as soon as you have API/CLI access
to Blanq, SimplePDF, Adobe Acrobat AI, Foxit AI, PDFgear, etc. (Phase 5 of
the benchmark spec in the root README).
