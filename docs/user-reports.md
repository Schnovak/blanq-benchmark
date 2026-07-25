# User reports: PDF form auto-detect fails on scanned and teacher-authored documents

## Why this document exists

Every mainstream PDF editor (Adobe Acrobat, Foxit, Nitro, Drawboard, Smallpdf, etc.) uses AcroForm widget scanning to auto-detect fillable fields. That works well on professionally-designed forms where the designer already tagged the fields as widgets in the PDF metadata. It fails silently on anything else: scanned worksheets, teacher-authored homework, ESL exercises, hand-drawn lines, PDFs printed and rescanned, or anything exported from Word without form-preparation.

Since every editor uses the same underlying technique, the same class of complaint shows up across all of them. This document collects public reports that establish the gap is real and ongoing, not a theoretical concern.

## Adobe Acrobat "Prepare Form"

The message users routinely hit on scanned or teacher-authored PDFs when Acrobat's auto-detect runs:

> No new form field annotations were detected.

Community consensus across Adobe's own forums:

- The form-fields wizard is unable to infer intent from blank spaces on a page
- It works reliably only on very simple forms and generates unusable field names on harder ones
- Users always have to add missing fields manually, remove wrong ones, and adjust the rest
- Users explicitly ask for a "scanned form to fillable PDF as a service" on Adobe's developer forum, i.e. the exact capability BlanQ provides

Sources:
- [Adobe Community: "Prepare Form not creating all form fields"](https://community.adobe.com/questions-9/prepare-form-not-creating-all-form-fields-1258403)
- [Adobe Community: "No form field automatically added when creating forms"](https://community.adobe.com/t5/acrobat-discussions/no-form-field-automatically-added-when-creating-forms/m-p/14821369/highlight/true)
- [Adobe Community: "Prepare form not working"](https://community.adobe.com/t5/acrobat-discussions/prepare-form-not-working/m-p/13467081/highlight/true)
- [Adobe Community: user request for a "scanned form to fillable PDF as a service"](https://community.adobe.com/questions-18/scanned-form-to-fillable-pdf-as-a-service-591027)
- [Adobe Community: "Why is Prepare Form not filling the entire PDF"](https://community.adobe.com/t5/acrobat-discussions/why-is-the-prepare-form-function-not-filling-the-entire-pdf-form-with-fillable-boxes/m-p/11626599)

## Foxit PDF Editor

Foxit's own developer documentation admits the same limitation. Their form-recognition engine relies on fields having clear borders, and detection is described as unreliable when borders are absent. That is a direct vendor admission that the AcroForm-style approach fails on the same class of documents.

Source:
- [Foxit developer docs: "How to add form fields to scanned PDFs using automatic form recognition"](https://developers.foxit.com/pdf/how-to-add-form-fields-to-scanned-pdfs-using-automatic-form-recognition/)

## Teachers and educators

The pattern that repeats across teacher-focused forums: a teacher makes a worksheet in Word or scans one in, exports to PDF, expects students to type into it, discovers the auto-detect finds nothing.

Direct teacher quote from Adobe's community:

> Everytime I send one, they are not able to type on it.

The common workaround teachers end up adopting is to abandon PDF entirely and convert the worksheet into Google Slides, where each blank has to be manually covered with a positioned text box. This appears in university-published guidance:

> Fillable PDF files are not compatible with Google Classroom or Canvas, so they converted their workbooks to Google Slides where text boxes allow students to type answers.

Sources:
- [Adobe Community: "Teachers trying to make a fillable worksheet"](https://community.adobe.com/t5/acrobat-discussions/teachers-trying-to-make-a-filiable-worksheet/m-p/10995569)
- [Take Charge Today, University of Arizona: workbook conversion to Google Slides](https://takechargetoday.arizona.edu/node/16890)
- [Inclusive Schools Australia: "The overlooked challenges of a PDF worksheet"](https://inclusiveschools.com.au/the-overlooked-challenges-of-a-pdf-worksheet/)
- [Microsoft Q&A: sharing fillable PDFs in Teams](https://learn.microsoft.com/en-us/answers/questions/4390733/sharing-fillable-pdf-files-in-a-shared-workspace)
- [Curriculum Corner: making a PDF fillable in Google Classroom (via Google Slides conversion)](https://www.thecurriculumcorner.com/thecurriculumcorner123/making-a-pdf-fillable-in-google-classroom/)

## What this means for the benchmark

None of the tools above are broken. They all use the same underlying algorithm: scan the PDF for AcroForm widget annotations, present those as fillable fields. If the widgets do not exist (scanned, teacher-authored in Word, printed-then-rescanned, hand-drawn), the algorithm has nothing to report.

BlanQ takes a different approach: it detects blank regions visually on the rendered page, so it works whether or not the source PDF has widget metadata. On this benchmark's Education category (52 hand-reviewed ESL worksheets that do not have AcroForm widgets), the AcroForm baseline scores F1 = 0.00 and BlanQ scores F1 = 0.99. See [`../README.md`](../README.md) for the full comparison.

That is the gap this benchmark documents and BlanQ is built to fill.
