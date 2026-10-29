"""Parse the PDF and build the indexes.  Run from the project root:

    uv run python -m scripts.ingest            # SQLite + Chroma (needs OPENAI_API_KEY)
    uv run python -m scripts.ingest --no-embed # SQLite only (no OpenAI calls)
"""
import argparse

from app.chunking import prose_chunks, record_to_chunk
from app.config import settings
from app.indexing import build_chroma, save_sqlite
from app.ingestion import read_pdf
from app.table_parser import parse_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-embed", action="store_true", help="skip the Chroma/OpenAI step")
    args = parser.parse_args()

    pdf = settings.resolve(settings.pdf_path)
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}. Set PDF_PATH in .env.")

    pages = read_pdf(pdf)
    records = parse_table(pages, pdf.name)
    chunks = [record_to_chunk(r) for r in records] + prose_chunks(pages, pdf.name)
    save_sqlite(settings.resolve(settings.sqlite_path), records, chunks)

    flagged = [r for r in records if r.needs_review]
    print(f"{pdf.name}: {len(pages)} pages, {len(records)} table rows, "
          f"{len(chunks) - len(records)} prose chunks")
    print(f"Rows flagged for manual review: {len(flagged)}")
    for r in flagged:
        print(f"  page {r.page}: {r.original_row} -> {r.review_note}")

    if args.no_embed:
        print("Skipped embeddings (--no-embed).")
        return
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not set in .env; cannot build the Chroma index.")
    build_chroma(settings, chunks)
    print(f"Chroma index built with {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
