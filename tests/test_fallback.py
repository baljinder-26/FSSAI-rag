import pytest
from fastapi.testclient import TestClient

from app import answer as answer_module
from app.answer import LLMAnswer, answer_question
from app.config import NOT_FOUND, ServiceError, Settings
from app.main import app, get_llm, get_retriever

TEA = "What is the limit of lead in tea?"


def never_called(question, evidence):
    raise AssertionError("the LLM must not be called without evidence")


@pytest.mark.parametrize("q", [
    "lead in rice",                          # food exists, but not under lead
    "What is the limit of aflatoxin in milk?",  # contaminant not in the document
    "What is the capital of India?",
])
def test_no_evidence_returns_exact_fallback_without_calling_llm(retriever, q):
    resp = answer_question(q, retriever, never_called)
    assert resp.answer == NOT_FOUND == "Information not found in the current guidelines."
    assert resp.sources == []


def test_llm_saying_not_found_or_citing_nothing_returns_fallback(retriever):
    for llm_answer in (LLMAnswer(answer=NOT_FOUND, evidence_ids=["E1"]),
                       LLMAnswer(answer="Lead in tea is 5.0", evidence_ids=[]),
                       LLMAnswer(answer="Lead in tea is 5.0", evidence_ids=["E99"])):
        assert answer_question(TEA, retriever, lambda q, e, a=llm_answer: a).answer == NOT_FOUND


def test_invented_number_is_blocked(retriever):
    llm = lambda q, e: LLMAnswer(answer="Lead in tea is 7.5 mg/kg.", evidence_ids=["E1"])
    assert answer_question(TEA, retriever, llm).answer == NOT_FOUND


def test_sources_carry_real_page_and_table(retriever):
    llm = lambda q, e: LLMAnswer(answer="Lead in tea: 5.0 on dry matter basis (page 6).",
                                 evidence_ids=["E1"])
    resp = answer_question(TEA, retriever, llm)
    (src,) = resp.sources
    assert src.document == "Compendium_Contaminants_Regulations.pdf"
    assert src.page == 6 and src.table.endswith("1. Lead")
    assert "Tea" in src.evidence and "5.0 on dry matter basis" in src.evidence


def test_api_returns_answer_and_sources(retriever):
    llm = lambda q, e: LLMAnswer(answer="Lead in tea: 5.0 on dry matter basis.", evidence_ids=["E1"])
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_llm] = lambda: llm
    try:
        body = TestClient(app).post("/query", json={"question": TEA}).json()
    finally:
        app.dependency_overrides.clear()
    assert body["sources"][0]["page"] == 6


def test_api_error_does_not_expose_api_key(retriever, monkeypatch):
    secret = "sk-test-SECRET-123"

    class BrokenChat:
        def __init__(self, **kwargs): ...
        def with_structured_output(self, _): return self
        def invoke(self, _): raise RuntimeError(f"Incorrect API key provided: {secret}")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", BrokenChat)
    llm = answer_module.make_llm(Settings(openai_api_key=secret, _env_file=None))
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_llm] = lambda: llm
    try:
        resp = TestClient(app).post("/query", json={"question": TEA})
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 503
    assert secret not in resp.text and "answer" not in resp.json()


def test_missing_key_gives_clear_config_error():
    with pytest.raises(ServiceError, match="OPENAI_API_KEY"):
        answer_module.make_llm(Settings(openai_api_key="", _env_file=None))
