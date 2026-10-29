"""Generate the FICTIONAL sample policy PDF used for demos, tests and evaluation.

Usage:  python scripts/generate_sample_pdf.py [output_path]
Requires reportlab (listed in requirements-dev.txt). Output is deterministic (invariant=1).
"""

import sys
import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "sample_documents" / "northwind_policies.pdf"

COMPANY = "Northwind Analytics Ltd. (FICTIONAL - sample data for software testing only)"

PAGES: list[tuple[str, list[str]]] = [
    (
        "1. Vacation Policy",
        [
            "Full-time employees receive 22 paid vacation days per calendar year.",
            "Up to 5 unused vacation days may be carried over into the following year.",
            "Vacation requests longer than 10 consecutive days must be approved by a department head.",
        ],
    ),
    (
        "2. Sick and Parental Leave",
        [
            "Employees receive 10 paid sick days per year. A medical certificate is required "
            "after 3 consecutive sick days.",
            "Parental leave is 16 weeks at full pay and is available to either parent.",
        ],
    ),
    (
        "3. Password and Access Security",
        [
            "Passwords must be at least 14 characters long and must be rotated every 90 days.",
            "Multi-factor authentication is mandatory for all company systems.",
            "Shared accounts are prohibited.",
        ],
    ),
    (
        "4. Devices and Incident Reporting",
        [
            "All company laptops must use full-disk encryption.",
            "Lost or stolen devices must be reported to the security team within 24 hours.",
            "Suspected phishing emails must be forwarded to the security mailbox and then deleted.",
        ],
    ),
    (
        "5. Travel and Expenses",
        [
            "Meals are reimbursed up to 60 dollars per day while traveling.",
            "Hotel stays are reimbursed up to 180 dollars per night.",
            "Flights costing more than 500 dollars require prior manager approval.",
            "Expense receipts must be submitted within 30 days of the expense.",
        ],
    ),
    (
        "6. Remote Work",
        [
            "Employees may work remotely up to 3 days per week.",
            "Core collaboration hours are 10:00 to 15:00 local time.",
            "A one-time home office stipend of 400 dollars is available after the probation period.",
            "A VPN connection is required when accessing internal systems remotely.",
        ],
    ),
]


def build_pdf(path: Path = DEFAULT_OUTPUT) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4, invariant=1)
    c.setTitle("Northwind Analytics - Sample Company Policies (fictional)")
    width, height = A4
    for title, bullets in PAGES:
        y = height - 60
        c.setFont("Helvetica", 8)
        c.drawString(50, y, COMPANY)
        y -= 36
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, title)
        y -= 30
        c.setFont("Helvetica", 11)
        for bullet in bullets:
            for i, line in enumerate(textwrap.wrap(bullet, 85)):
                c.drawString(50 if i == 0 else 64, y, line)
                y -= 16
            y -= 8
        c.showPage()
    c.save()
    return path


if __name__ == "__main__":
    out = build_pdf(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT)
    print(f"Wrote {out} ({out.stat().st_size} bytes, {len(PAGES)} pages)")
