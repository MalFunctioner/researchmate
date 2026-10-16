"""Convert the Markdown report to a downloadable PDF."""
import re

import markdown
from fpdf import FPDF

_REPLACE = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
            "—": "-", "…": "...", "•": "-", " ": " "}


def _latin1(text: str) -> str:
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def report_to_pdf(report_md: str) -> bytes:
    html = markdown.markdown(_latin1(report_md), extensions=["extra"])
    # fpdf's HTML renderer doesn't support every tag; keep it simple.
    html = re.sub(r"</?(code|pre|table|thead|tbody|tr|td|th|hr)[^>]*>", "", html)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.write_html(html)
    return bytes(pdf.output())
