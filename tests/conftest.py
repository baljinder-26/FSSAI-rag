import pytest

from app.chunking import prose_chunks, record_to_chunk
from app.config import settings
from app.ingestion import read_pdf
from app.retrieval import Retriever
from app.table_parser import parse_table


@pytest.fixture(scope="session")
def parsed():
    pdf = settings.resolve(settings.pdf_path)
    if not pdf.exists():
        pytest.skip("selected PDF is not in data/")
    pages = read_pdf(pdf)
    records = parse_table(pages, pdf.name)
    chunks = [record_to_chunk(r) for r in records] + prose_chunks(pages, pdf.name)
    return records, chunks


@pytest.fixture(scope="session")
def retriever(parsed):
    records, chunks = parsed
    return Retriever(records, chunks)   # BM25 + structured only: no OpenAI calls


def find(records, parameter, food):
    return [r for r in records if r.parameter == parameter and r.food == food]
