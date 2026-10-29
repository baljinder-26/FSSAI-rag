"""Read the PDF page by page: raw table rows, prose outside the table, full page text.

pdfplumber is used instead of Docling: the FSSAI PDF is a digital (not scanned)
document with ruled table lines, and pdfplumber keeps every row/column cell intact.
"""
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


@dataclass
class PageData:
    page: int                                   # 1-based PDF page number
    full_text: str                              # everything on the page (for validation)
    prose: str                                  # text outside the table
    rows: list[list[str | None]] = field(default_factory=list)  # raw table rows


def read_pdf(path: Path) -> list[PageData]:
    pages: list[PageData] = []
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            rows: list[list[str | None]] = []
            prose = page.extract_text() or ""
            if tables:
                rows = [r for t in tables for r in t.extract()]
                prose = page.outside_bbox(tables[0].bbox).extract_text() or ""
            pages.append(PageData(number, page.extract_text() or "", prose, rows))
    return pages
