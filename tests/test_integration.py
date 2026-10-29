"""Optional: real OpenAI calls. Skipped unless OPENAI_API_KEY is set and `scripts.ingest` was run."""
import pytest

from app.answer import answer_question, make_llm
from app.config import NOT_FOUND, settings
from app.indexing import load_sqlite, make_vector_search
from app.retrieval import Retriever

pytestmark = pytest.mark.skipif(
    not (settings.openai_api_key and settings.resolve(settings.sqlite_path).exists()),
    reason="needs OPENAI_API_KEY and an ingested index")


@pytest.fixture(scope="module")
def live():
    records, chunks = load_sqlite(settings.resolve(settings.sqlite_path))
    return Retriever(records, chunks, make_vector_search(settings)), make_llm(settings)


def test_live_exact_limit(live):
    resp = answer_question("What is the maximum limit of lead in tea?", *live)
    assert "5.0" in resp.answer and "dry matter" in resp.answer and resp.sources[0].page == 6


def test_live_not_found(live):
    assert answer_question("What is the limit of lead in rice?", *live).answer == NOT_FOUND
