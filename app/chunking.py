"""Table-aware row chunks and section-aware prose chunks.

Every row chunk repeats the metal, food, unit, condition, regulation, table id and
page, so a row is never separated from the context that gives its number meaning.
Prose is split at clause boundaries (e.g. "1.1.1 :", "2.1 :") instead of by size.
"""
import re

from app.ingestion import PageData
from app.schemas import Chunk, TableRecord

CLAUSE_RE = re.compile(r"^(?:\d+\[)?(\d+\.\d+(?:\.\d+)?)\s*(?::|$)")
CHAPTER_RE = re.compile(r"^CHAPTER\s+\d+")


def record_to_chunk(r: TableRecord) -> Chunk:
    pages = f"page {r.page}" + (f"-{r.page_end}" if r.page_end else "")
    parts = [
        f"{r.regulation} | {r.table}",
        f"Metal contaminant: {r.parameter}",
        f"Article of food: {r.food}",
        f"Limit: {r.limit_text} ({r.limit_type}, unit: {r.unit})",
    ]
    if r.condition:
        parts.append(f"Condition: {r.condition}")
    parts.append(f"Source: {r.document}, {pages}")
    return Chunk(id=r.id, kind="row", text="\n".join(parts), document=r.document,
                 page=r.page, table=r.table, parameter=r.parameter)


def prose_chunks(pages: list[PageData], document: str) -> list[Chunk]:
    """One chunk per numbered clause, prefixed with the chapter/section heading."""
    chunks: list[Chunk] = []
    for page in pages:
        chapter, heading, body, heading_only = "", "", [], False

        def flush() -> None:
            text = " ".join(body).strip()
            if text and heading:
                context = " | ".join(x for x in (chapter, heading) if x)
                chunks.append(Chunk(id=f"P{len(chunks) + 1}", kind="prose",
                                    text=f"{context}\n{text}", document=document, page=page.page))

        lines = [ln.strip() for ln in page.prose.split("\n") if ln.strip()]
        for i, line in enumerate(lines):
            if line.startswith("Version"):
                continue  # page footer
            if CHAPTER_RE.match(line):
                flush()
                chapter, heading, body = f"{line} {lines[i + 1]}" if i + 1 < len(lines) else line, "", []
            elif m := CLAUSE_RE.match(line):
                title = line.partition(":")[2].strip()
                short = len(title) <= 45   # a heading such as "2.1 : METAL CONTAMINANTS"
                label = line if short else f"{m.group(1)} :"
                if heading and not body and heading_only:
                    heading = f"{heading} > {label}"   # e.g. 2.1 introducing 2.1.1
                else:
                    flush()
                    heading = label
                body = [] if short else [title]
                heading_only = short
            elif heading:
                body.append(line)
        flush()
    return chunks
