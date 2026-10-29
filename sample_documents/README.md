# Sample documents

`northwind_policies.pdf` is a small, **entirely fictional** six-page company policy document
("Northwind Analytics Ltd."). It contains no real, confidential or copyrighted material and exists
only for demos, automated tests and the evaluation in `evaluation_data/sample_questions.json`.

Regenerate it (deterministic output, requires `reportlab` from `requirements-dev.txt`):

```powershell
python scripts/generate_sample_pdf.py
```

Do not put private or confidential PDFs in this folder: it is tracked by Git.
