#!/usr/bin/env python3
"""
Generates the v0.1 SEED dataset for blanq-benchmark.

These are synthetic, born-digital pages (not real scraped forms). They exist so the
full pipeline (dataset -> ground truth -> detector -> eval -> leaderboard) runs
end-to-end from day one, with ground truth that is provably exact (we draw the
blanks, so we know their coordinates to the point). Real sourced/scanned pages
should be added under dataset/<category>/ following the same manifest + ground
truth schema -- see CONTRIBUTING.md.

Coordinate convention (must match schema/ground_truth_schema.json):
  origin top-left, x right, y DOWN, units = PDF points (1/72 inch).
reportlab's canvas is bottom-left origin with y UP, so every draw call below
converts through `to_rl_y()`.

Run:
  python3 scripts/generate_seed_dataset.py
"""
import csv
import json
import os

from reportlab.lib.pagesizes import LETTER, A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "dataset")
GT_DIR = os.path.join(ROOT, "ground_truth")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")

MANIFEST_COLUMNS = [
    "id", "file", "category", "country", "language", "source", "pages",
    "capture", "dpi", "rotation_deg", "noise", "difficulty", "license",
]


class PageBuilder:
    """Draws a single-page PDF and accumulates exact ground-truth blank records."""

    def __init__(self, page_id, pagesize=LETTER):
        self.page_id = page_id
        self.width, self.height = pagesize
        self.path = None  # set by save()
        self.c = None
        self.blanks = []
        self._n = 0
        self._pagesize = pagesize

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def _next_id(self):
        self._n += 1
        return f"b{self._n:03d}"

    def _rl_y(self, y_top, h=0):
        """Convert a top-origin y (and optional box height) to reportlab's bottom-origin y
        for the BOTTOM edge of a box, so drawRect(x, rl_y, w, h) lands correctly."""
        return self.height - y_top - h

    # ---- visual chrome (non-blank content) ----
    def init_canvas(self, path):
        self.path = path
        # invariant=1 -> reportlab emits a fixed creation date/ID instead of "now",
        # so re-running this script produces byte-identical PDFs (reproducibility).
        self.c = canvas.Canvas(path, pagesize=self._pagesize, invariant=1)

    def title(self, text, x=0.6 * inch, y=0.55 * inch, size=15):
        self.c.setFont("Helvetica-Bold", size)
        self.c.drawString(x, self.height - y, text)

    def section(self, text, x, y_top, size=10.5):
        self.c.setFont("Helvetica-Bold", size)
        self.c.drawString(x, self.height - y_top - size, text)

    def label(self, text, x, y_top, size=8.5):
        self.c.setFont("Helvetica", size)
        self.c.drawString(x, self.height - y_top - size, text)

    def paragraph(self, text, x, y_top, size=9, width=None):
        self.c.setFont("Helvetica", size)
        self.c.drawString(x, self.height - y_top - size, text)

    def hline(self, x, y_top, w):
        self.c.setLineWidth(0.75)
        self.c.line(x, self.height - y_top, x + w, self.height - y_top)

    def box_outline(self, x, y_top, w, h):
        self.c.setLineWidth(0.75)
        self.c.rect(x, self._rl_y(y_top, h), w, h, stroke=1, fill=0)

    # ---- blank primitives (draw + record ground truth) ----
    def add_line_blank(self, x, y_top, w, type_="single_line", label_text=None,
                        box_h=16, line_offset=14, rows=None):
        """A ruled line for handwriting, e.g. Name: __________"""
        if label_text:
            self.label(label_text, x, y_top - 9)
        line_y = y_top + line_offset
        self.hline(x, line_y, w)
        bid = self._next_id()
        self.blanks.append({
            "id": bid, "x": round(x, 1), "y": round(y_top, 1),
            "width": round(w, 1), "height": round(box_h, 1),
            "type": type_, "confidence": 1.0, "rows": rows,
        })
        return bid

    def add_box_blank(self, x, y_top, w, h, type_, label_text=None, rows=None,
                       ruled_rows=False):
        """A rectangular blank: checkbox / signature / tiny_box / table_cell /
        large_paragraph / multi_line."""
        if label_text:
            self.label(label_text, x, y_top - 10)
        self.box_outline(x, y_top, w, h)
        if ruled_rows and rows and rows > 1:
            row_h = h / rows
            for i in range(1, rows):
                self.hline(x, y_top + i * row_h, w)
        bid = self._next_id()
        self.blanks.append({
            "id": bid, "x": round(x, 1), "y": round(y_top, 1),
            "width": round(w, 1), "height": round(h, 1),
            "type": type_, "confidence": 1.0, "rows": rows,
        })
        return bid

    def add_checkbox(self, x, y_top, label_text, size=10):
        self.box_outline(x, y_top, size, size)
        self.label(label_text, x + size + 4, y_top - (size - 8) / 2)
        bid = self._next_id()
        self.blanks.append({
            "id": bid, "x": round(x, 1), "y": round(y_top, 1),
            "width": size, "height": size, "type": "checkbox",
            "confidence": 1.0, "rows": None,
        })
        return bid

    def add_radio(self, x, y_top, label_text, r=5):
        cx, cy_top = x + r, y_top + r
        self.c.circle(cx, self._rl_y(cy_top) + r, r, stroke=1, fill=0)
        self.label(label_text, x + 2 * r + 5, y_top - (2 * r - 8) / 2)
        bid = self._next_id()
        self.blanks.append({
            "id": bid, "x": round(x, 1), "y": round(y_top, 1),
            "width": round(2 * r, 1), "height": round(2 * r, 1),
            "type": "radio", "confidence": 1.0, "rows": None,
        })
        return bid

    def save(self):
        self.c.showPage()
        self.c.save()

    def ground_truth(self):
        return {
            "id": self.page_id,
            "page_width": self.width,
            "page_height": self.height,
            "blanks": self.blanks,
        }


def write_page(pb, category):
    pdf_rel = os.path.join(category, f"{pb.page_id}.pdf")
    pdf_abs = os.path.join(DATASET_DIR, pdf_rel)
    os.makedirs(os.path.dirname(pdf_abs), exist_ok=True)
    pb.init_canvas(pdf_abs)
    return pdf_rel


def build_gov_w4():
    pid = "gov-us-w4-001"
    pb = PageBuilder(pid, LETTER)
    rel = write_page(pb, "government")
    pb.title("Form W-4 (sample) - Employee's Withholding Certificate")
    pb.section("Step 1: Personal Information", 0.6 * inch, 1.0 * inch)
    pb.add_line_blank(0.6 * inch, 1.4 * inch, 2.6 * inch, "name", "First name and middle initial")
    pb.add_line_blank(3.4 * inch, 1.4 * inch, 3.0 * inch, "name", "Last name")
    pb.add_line_blank(0.6 * inch, 1.75 * inch, 4.0 * inch, "single_line", "Address")
    pb.add_line_blank(0.6 * inch, 2.25 * inch, 2.2 * inch, "single_line", "City or town")
    pb.add_line_blank(2.9 * inch, 2.25 * inch, 0.7 * inch, "single_line", "State")
    pb.add_line_blank(3.7 * inch, 2.25 * inch, 1.0 * inch, "single_line", "ZIP code")
    pb.add_line_blank(0.6 * inch, 2.75 * inch, 3.0 * inch, "single_line", "Social security number")

    pb.section("Filing Status (choose one)", 0.6 * inch, 3.2 * inch)
    pb.add_checkbox(0.6 * inch, 3.5 * inch, "Single or Married filing separately")
    pb.add_checkbox(0.6 * inch, 3.8 * inch, "Married filing jointly")
    pb.add_checkbox(0.6 * inch, 4.1 * inch, "Head of household")

    pb.section("Step 4: Other Adjustments (optional)", 0.6 * inch, 4.6 * inch)
    pb.add_line_blank(0.6 * inch, 4.9 * inch, 2.0 * inch, "single_line", "(a) Other income")
    pb.add_line_blank(0.6 * inch, 5.35 * inch, 2.0 * inch, "single_line", "(b) Deductions")

    pb.section("Step 5: Sign Here", 0.6 * inch, 6.0 * inch)
    pb.add_box_blank(0.6 * inch, 6.3 * inch, 3.2 * inch, 0.5 * inch, "signature", "Employee signature")
    pb.add_line_blank(4.1 * inch, 6.6 * inch, 1.5 * inch, "date", "Date")

    pb.save()
    return pb, rel, dict(country="US", language="en", source="synthetic-seed",
                          capture="born_digital", dpi="", rotation_deg=0,
                          noise="none", difficulty="easy", license="synthetic")


def build_gov_eu_visa():
    pid = "gov-eu-visa-001"
    pb = PageBuilder(pid, A4)
    rel = write_page(pb, "government")
    pb.title("Schengen Visa Application (sample)")
    pb.section("1. Applicant details", 0.6 * inch, 1.0 * inch)
    pb.add_line_blank(0.6 * inch, 1.4 * inch, 3.0 * inch, "name", "Surname")
    pb.add_line_blank(4.0 * inch, 1.4 * inch, 2.7 * inch, "name", "First name(s)")
    pb.add_line_blank(0.6 * inch, 1.75 * inch, 2.2 * inch, "date", "Date of birth")
    pb.add_line_blank(3.0 * inch, 1.75 * inch, 2.0 * inch, "single_line", "Place of birth")
    pb.add_line_blank(0.6 * inch, 2.25 * inch, 3.0 * inch, "single_line", "Nationality")
    pb.add_line_blank(4.0 * inch, 2.25 * inch, 2.7 * inch, "single_line", "Passport number")

    pb.section("2. Sex", 0.6 * inch, 2.75 * inch)
    pb.add_radio(0.6 * inch, 3.0 * inch, "Male")
    pb.add_radio(1.6 * inch, 3.0 * inch, "Female")

    pb.section("3. Purpose of travel", 0.6 * inch, 3.5 * inch)
    for i, lbl in enumerate(["Tourism", "Business", "Family visit", "Study"]):
        pb.add_checkbox(0.6 * inch + (i % 2) * 2.2 * inch, 3.8 * inch + (i // 2) * 0.35 * inch, lbl)

    pb.section("4. Declaration", 0.6 * inch, 4.9 * inch)
    pb.add_box_blank(0.6 * inch, 5.3 * inch, 6.2 * inch, 0.9 * inch, "large_paragraph",
                      "Additional remarks", rows=3, ruled_rows=True)

    pb.add_box_blank(0.6 * inch, 6.4 * inch, 2.6 * inch, 0.55 * inch, "signature", "Applicant signature")
    pb.add_line_blank(4.0 * inch, 6.75 * inch, 1.6 * inch, "date", "Date")

    pb.save()
    return pb, rel, dict(country="EU", language="en", source="synthetic-seed",
                          capture="born_digital", dpi="", rotation_deg=0,
                          noise="none", difficulty="easy", license="synthetic")


def build_edu_algebra():
    pid = "edu-algebra-worksheet-001"
    pb = PageBuilder(pid, LETTER)
    rel = write_page(pb, "education")
    pb.title("Algebra Worksheet 3.2 - Solving Linear Equations")
    pb.label("Name: ________________________   Date: ____________   Period: ____", 0.6 * inch, 0.85 * inch)
    pb.add_line_blank(1.0 * inch, 0.72 * inch, 1.9 * inch, "name")
    pb.add_line_blank(3.85 * inch, 0.72 * inch, 1.0 * inch, "date")
    pb.add_line_blank(5.75 * inch, 0.72 * inch, 0.6 * inch, "single_line")

    problems = ["1)  3x + 5 = 20", "2)  2(x - 4) = 10", "3)  x/3 + 7 = 12", "4)  5x - 2 = 3x + 8"]
    y = 1.3 * inch
    for prob in problems:
        pb.label(prob, 0.6 * inch, y - 9)
        pb.add_line_blank(3.4 * inch, y, 2.4 * inch, "single_line", "x =")
        y += 0.5 * inch

    pb.section("5) Word problem - show your work", 0.6 * inch, y + 0.1 * inch)
    pb.add_box_blank(0.6 * inch, y + 0.35 * inch, 6.9 * inch, 1.6 * inch, "multi_line",
                      rows=5, ruled_rows=True)

    pb.save()
    return pb, rel, dict(country="US", language="en", source="synthetic-seed",
                          capture="born_digital", dpi="", rotation_deg=0,
                          noise="none", difficulty="easy", license="synthetic")


def build_edu_chem_lab():
    pid = "edu-chem-lab-001"
    pb = PageBuilder(pid, LETTER)
    rel = write_page(pb, "education")
    pb.title("Chemistry Lab Report - Titration")
    pb.add_line_blank(0.6 * inch, 0.85 * inch, 2.2 * inch, "name", "Student name")
    pb.add_line_blank(3.2 * inch, 0.85 * inch, 1.3 * inch, "date", "Date")
    pb.add_line_blank(4.9 * inch, 0.85 * inch, 1.5 * inch, "single_line", "Lab partner")

    pb.section("Data Table", 0.6 * inch, 1.5 * inch)
    col_labels = ["Trial", "Volume (mL)", "Concentration (M)"]
    for i, lbl in enumerate(col_labels):
        pb.label(lbl, 0.6 * inch + i * 2.2 * inch, 1.72 * inch)
    for row in range(3):
        y = 1.95 * inch + row * 0.32 * inch
        for col in range(3):
            pb.add_box_blank(0.6 * inch + col * 2.2 * inch, y, 2.0 * inch, 0.28 * inch, "table_cell")

    pb.section("Observations", 0.6 * inch, 3.2 * inch)
    pb.add_box_blank(0.6 * inch, 3.45 * inch, 6.9 * inch, 1.3 * inch, "multi_line", rows=4, ruled_rows=True)

    pb.add_box_blank(0.6 * inch, 5.1 * inch, 2.6 * inch, 0.5 * inch, "signature", "Student signature")

    pb.save()
    return pb, rel, dict(country="US", language="en", source="synthetic-seed",
                          capture="born_digital", dpi="", rotation_deg=0,
                          noise="none", difficulty="medium", license="synthetic")


def build_medical_intake():
    pid = "med-intake-001"
    pb = PageBuilder(pid, LETTER)
    rel = write_page(pb, "medical")
    pb.title("New Patient Intake Form (sample)")
    pb.section("Patient Information", 0.6 * inch, 1.0 * inch)
    pb.add_line_blank(0.6 * inch, 1.4 * inch, 3.0 * inch, "name", "Full legal name")
    pb.add_line_blank(4.0 * inch, 1.4 * inch, 1.3 * inch, "date", "Date of birth")
    pb.add_line_blank(5.6 * inch, 1.4 * inch, 1.0 * inch, "single_line", "Sex")
    pb.add_line_blank(0.6 * inch, 1.75 * inch, 3.0 * inch, "single_line", "Phone number")
    pb.add_line_blank(4.0 * inch, 1.75 * inch, 2.6 * inch, "single_line", "Email")

    pb.section("Reason for visit", 0.6 * inch, 2.3 * inch)
    pb.add_box_blank(0.6 * inch, 2.55 * inch, 6.6 * inch, 0.8 * inch, "large_paragraph", rows=2, ruled_rows=True)

    pb.section("Do you have any of the following? (check all that apply)", 0.6 * inch, 3.6 * inch)
    symptoms = ["Fever", "Chest pain", "Shortness of breath", "Allergies", "Diabetes", "High blood pressure"]
    for i, s in enumerate(symptoms):
        pb.add_checkbox(0.6 * inch + (i % 3) * 2.3 * inch, 3.9 * inch + (i // 3) * 0.35 * inch, s)

    pb.section("Consent", 0.6 * inch, 4.9 * inch)
    pb.add_box_blank(0.6 * inch, 5.15 * inch, 2.8 * inch, 0.5 * inch, "signature", "Patient / guardian signature")
    pb.add_line_blank(3.8 * inch, 5.45 * inch, 1.4 * inch, "date", "Date")

    pb.save()
    return pb, rel, dict(country="US", language="en", source="synthetic-seed",
                          capture="born_digital", dpi="", rotation_deg=0,
                          noise="none", difficulty="easy", license="synthetic")


def build_bank_loan():
    pid = "bank-loan-app-001"
    pb = PageBuilder(pid, LETTER)
    rel = write_page(pb, "banking_insurance")
    pb.title("Personal Loan Application (sample)")
    pb.section("Applicant", 0.6 * inch, 1.0 * inch)
    pb.add_line_blank(0.6 * inch, 1.4 * inch, 3.0 * inch, "name", "Full name")
    pb.add_line_blank(4.0 * inch, 1.4 * inch, 2.6 * inch, "single_line", "SSN / national ID")
    pb.add_line_blank(0.6 * inch, 1.75 * inch, 2.0 * inch, "single_line", "Annual income")
    pb.add_line_blank(2.9 * inch, 1.75 * inch, 2.0 * inch, "single_line", "Employer")

    pb.section("Loan type", 0.6 * inch, 2.3 * inch)
    for i, lbl in enumerate(["Auto", "Mortgage", "Personal", "Business"]):
        pb.add_checkbox(0.6 * inch + (i % 2) * 2.4 * inch, 2.6 * inch + (i // 2) * 0.35 * inch, lbl)

    pb.section("Reason for loan", 0.6 * inch, 3.5 * inch)
    pb.add_box_blank(0.6 * inch, 3.75 * inch, 6.6 * inch, 1.0 * inch, "large_paragraph", rows=3, ruled_rows=True)

    pb.add_box_blank(0.6 * inch, 5.1 * inch, 2.8 * inch, 0.5 * inch, "signature", "Applicant signature")
    pb.add_line_blank(3.8 * inch, 5.4 * inch, 1.4 * inch, "date", "Date")

    pb.save()
    return pb, rel, dict(country="US", language="en", source="synthetic-seed",
                          capture="born_digital", dpi="", rotation_deg=0,
                          noise="none", difficulty="easy", license="synthetic")


def build_hr_onboarding():
    pid = "hr-onboarding-001"
    pb = PageBuilder(pid, LETTER)
    rel = write_page(pb, "hr")
    pb.title("New Hire Onboarding Form (sample)")
    pb.add_line_blank(0.6 * inch, 0.95 * inch, 3.0 * inch, "name", "Employee name")
    pb.add_line_blank(4.0 * inch, 0.95 * inch, 1.4 * inch, "date", "Start date")
    pb.add_line_blank(5.6 * inch, 0.95 * inch, 1.0 * inch, "single_line", "Dept")

    pb.section("Benefits elected", 0.6 * inch, 1.5 * inch)
    for i, lbl in enumerate(["Health insurance", "Dental", "Vision", "401(k)", "Life insurance", "Commuter benefits"]):
        pb.add_checkbox(0.6 * inch + (i % 3) * 2.3 * inch, 1.8 * inch + (i // 3) * 0.35 * inch, lbl)

    pb.section("Emergency contact", 0.6 * inch, 2.8 * inch)
    contact_cols = ["Name", "Relationship", "Phone"]
    for i, lbl in enumerate(contact_cols):
        pb.label(lbl, 0.6 * inch + i * 2.3 * inch, 3.02 * inch)
    for col in range(3):
        pb.add_box_blank(0.6 * inch + col * 2.3 * inch, 3.25 * inch, 2.1 * inch, 0.3 * inch, "table_cell")

    pb.section("Acknowledgement", 0.6 * inch, 3.9 * inch)
    pb.add_box_blank(0.6 * inch, 4.15 * inch, 2.8 * inch, 0.5 * inch, "signature", "Employee signature")
    pb.add_line_blank(3.8 * inch, 4.45 * inch, 1.4 * inch, "date", "Date")

    pb.save()
    return pb, rel, dict(country="US", language="en", source="synthetic-seed",
                          capture="born_digital", dpi="", rotation_deg=0,
                          noise="none", difficulty="easy", license="synthetic")


def build_legal_agreement():
    pid = "legal-agreement-001"
    pb = PageBuilder(pid, LETTER)
    rel = write_page(pb, "legal")
    pb.title("Simple Services Agreement (sample)")
    pb.paragraph("This Agreement is entered into between the parties named below.", 0.6 * inch, 1.0 * inch)

    pb.section("1. Scope of work", 0.6 * inch, 1.35 * inch)
    pb.add_box_blank(0.6 * inch, 1.6 * inch, 6.9 * inch, 1.1 * inch, "large_paragraph", rows=4, ruled_rows=True)

    pb.section("2. Signatures", 0.6 * inch, 3.0 * inch)
    pb.add_box_blank(0.6 * inch, 3.4 * inch, 2.8 * inch, 0.5 * inch, "signature", "Party A signature")
    pb.add_line_blank(0.6 * inch, 4.15 * inch, 2.8 * inch, "underline", "Printed name")
    pb.add_line_blank(0.6 * inch, 4.55 * inch, 1.4 * inch, "date", "Date")

    pb.add_box_blank(4.0 * inch, 3.4 * inch, 2.8 * inch, 0.5 * inch, "signature", "Party B signature")
    pb.add_line_blank(4.0 * inch, 4.15 * inch, 2.8 * inch, "underline", "Printed name")
    pb.add_line_blank(4.0 * inch, 4.55 * inch, 1.4 * inch, "date", "Date")

    pb.section("3. Witness", 0.6 * inch, 4.9 * inch)
    pb.add_box_blank(0.6 * inch, 5.3 * inch, 2.8 * inch, 0.5 * inch, "signature", "Witness signature")

    pb.save()
    return pb, rel, dict(country="US", language="en", source="synthetic-seed",
                          capture="born_digital", dpi="", rotation_deg=0,
                          noise="none", difficulty="easy", license="synthetic")


BUILDERS = [
    build_gov_w4, build_gov_eu_visa, build_edu_algebra, build_edu_chem_lab,
    build_medical_intake, build_bank_loan, build_hr_onboarding, build_legal_agreement,
]


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(GT_DIR, exist_ok=True)
    manifest_rows = []
    total_blanks = 0

    for builder in BUILDERS:
        pb, rel_path, meta = builder()
        gt = pb.ground_truth()
        with open(os.path.join(GT_DIR, f"{pb.page_id}.json"), "w") as f:
            json.dump(gt, f, indent=2)
        total_blanks += len(gt["blanks"])

        row = {"id": pb.page_id, "file": rel_path, "category": os.path.dirname(rel_path), "pages": 1}
        row.update(meta)
        manifest_rows.append(row)
        print(f"  {pb.page_id:30s} {len(gt['blanks']):3d} blanks  -> dataset/{rel_path}")

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_COLUMNS})

    print(f"\nWrote {len(manifest_rows)} seed pages, {total_blanks} ground-truth blanks total.")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
