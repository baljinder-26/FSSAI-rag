"""Turn raw table rows into validated TableRecord objects.

The PDF holds one long table (Regulation 2.1.1(2)) with three columns:
(1) name of metal contaminant, (2) article of food, (3) limit in ppm (mg/kg or mg/L).
The metal name is only printed on the first row of each group, so it is carried
forward. A row that is cut by a page break is joined back into one record.
"""
import re

from app.ingestion import PageData
from app.schemas import TableRecord

METAL_RE = re.compile(r"^\s*(\d+)\.\s*(.+)$", re.S)
VALUE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(.*)$", re.S)
DEFAULT_UNIT = "mg/kg or mg/L (ppm)"


def _clean(text: str | None) -> str:
    """Collapse line breaks/whitespace and drop stray gazette bracket marks."""
    text = (text or "").replace("”]", "").replace("]", "")
    return re.sub(r"\s+", " ", text).strip()


def _words(text: str, limit: int | None = None, tail: bool = False) -> str:
    """Lower-case alphanumeric words padded with spaces, so matches respect word boundaries."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    if limit:
        words = words[-limit:] if tail else words[:limit]
    return f" {' '.join(words)} "


def find_regulation(pages: list[PageData]) -> str:
    """Read the section number/heading (e.g. 2.1.1) from the page that introduces the table."""
    for p in pages:
        heading = re.search(r"(\d+\.\d+)\s*:\s*([A-Z][A-Z ]+)\n", p.full_text)
        section = re.search(r"\[?(\d+\.\d+\.\d+)\s*\n", p.full_text)
        if heading and section and "column (3)" in p.full_text:
            return f"Regulation {section.group(1)}(2) - {heading.group(2).strip().title()}"
    raise ValueError("Could not locate the table's regulation heading on any page.")


def _validate(page: PageData, food_cell: str, limit_cell: str, value_number: str) -> str:
    """Check the printed value sits on the page, next to the row's food text.

    Returns an empty string when OK, otherwise a note explaining why review is needed.
    """
    lines = [ln for ln in page.full_text.split("\n")]
    first = _words(food_cell.split("\n")[0], 4)
    last = _words(food_cell.split("\n")[-1], 4, tail=True)
    if not first.strip():
        return "empty food text"
    starts = [i for i, ln in enumerate(lines) if first in _words(ln)]
    if not starts:
        return "food text not found on page"
    spread = max(len(limit_cell.strip().split("\n")) - 1, 1)
    pattern = rf"(?<![\d.]){re.escape(value_number)}(?![\d])"
    for start in starts:  # the same food can appear under several metals on one page
        end = next((i for i in range(start, len(lines)) if last in _words(lines[i])), start)
        # A multi-line limit is vertically centred on the row, so look that far around it.
        if re.search(pattern, " ".join(lines[max(start - spread, 0): end + spread + 1])):
            return ""
    return f"value {value_number} not found beside the row text"


def parse_table(pages: list[PageData], document: str) -> list[TableRecord]:
    regulation = find_regulation(pages)
    records: list[TableRecord] = []
    parameter = ""
    parameter_note = ""
    table = ""

    for page in pages:
        for index, row in enumerate(page.rows):
            if len(row) != 3:
                continue
            metal_cell, food_cell, limit_cell = row
            if _clean(metal_cell).startswith(("Name of metal", "(1)")):
                continue  # header rows

            food_cell = food_cell or ""
            limit_cell = limit_cell or ""

            # A row split by a page break: first row of the page, no metal, no limit.
            if index == 0 and not _clean(limit_cell) and records and metal_cell == "":
                prev = records[-1]
                prev.food = f"{prev.food} {_clean(food_cell)}"
                prev.original_row += f" {_clean(food_cell)}"
                prev.page_end = page.page
                continue

            if _clean(metal_cell):
                m = METAL_RE.match(metal_cell.strip())
                if not m:
                    raise ValueError(f"Unrecognised metal cell on page {page.page}: {metal_cell!r}")
                name = _clean(m.group(2))
                note = re.search(r"\((.*?)\)", name)
                parameter_note = note.group(1) if note else ""
                parameter = re.sub(r"\s*\(.*?\)", "", name).strip()
                table = f"{regulation.split(' - ')[0]} table - {m.group(1)}. {parameter}"

            limit = _clean(limit_cell)
            vm = VALUE_RE.match(limit)
            if not vm:
                raise ValueError(f"No numeric limit on page {page.page}: {row!r}")
            number, condition = vm.group(1), vm.group(2).strip()
            food = _clean(food_cell)
            conditions = [c for c in (condition, parameter_note) if c]
            note = _validate(page, food_cell.strip(), limit_cell, number)

            records.append(TableRecord(
                id=f"R{len(records) + 1}",
                parameter=parameter,
                food=food,
                limit_value=float(number),
                limit_text=limit,
                unit="mg/L" if "expressed in mg/L" in food else DEFAULT_UNIT,
                condition="; ".join(conditions),
                regulation=regulation,
                document=document,
                page=page.page,
                table=table,
                original_row=f"{parameter} | {food} | {limit}",
                needs_review=bool(note),
                review_note=note,
            ))
    return records
