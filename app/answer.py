"""Grounded answer generation with hard guardrails.

The LLM only sees the question and the retrieved evidence. After it answers we
verify (in plain Python) that it cited real evidence and that every number in its
answer appears in that evidence. Anything else becomes the exact fallback text.
Sources are built from our own metadata, never from text written by the model.
"""
import re
from typing import Callable

from pydantic import BaseModel, Field

from app.config import NOT_FOUND, ServiceError, Settings
from app.retrieval import Evidence, Retriever
from app.schemas import QueryResponse, Source

SYSTEM_PROMPT = f"""You answer questions about the FSSAI Food Safety and Standards (Contaminants, \
Toxins and Residues) Regulations using ONLY the numbered evidence supplied. Rules:
1. Never use outside knowledge. Never guess, convert units, calculate, or infer a limit.
2. Quote the limit exactly as printed, with its unit and any condition or basis \
(for example "on dry matter basis"), and mention the PDF page.
3. The food and the metal in the question must match the evidence row. A row for a different \
food or a different metal does NOT answer the question.
4. If several rows match (for example "Milks" and "Infant milk substitute"), report each one \
with its own food name instead of choosing silently.
5. If the evidence does not fully answer the question, reply with exactly: {NOT_FOUND}
6. If one part of a combined question is unsupported, answer the supported part and write \
{NOT_FOUND} for the other part.
List in evidence_ids the labels (such as E1) of every evidence item you used."""


class LLMAnswer(BaseModel):
    answer: str
    evidence_ids: list[str] = Field(default_factory=list)


LLM = Callable[[str, str], LLMAnswer]   # (question, evidence_text) -> LLMAnswer


def make_llm(settings: Settings) -> LLM:
    """Build the OpenAI chat call (via LangChain) with structured output."""
    if not settings.openai_api_key:
        raise ServiceError("OPENAI_API_KEY is not set. Add it to .env and restart the API.")
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model=settings.openai_chat_model, api_key=settings.openai_api_key,
                       temperature=0, max_retries=2, timeout=60).with_structured_output(LLMAnswer)

    def call(question: str, evidence_text: str) -> LLMAnswer:
        try:
            return model.invoke([
                ("system", SYSTEM_PROMPT),
                ("human", f"Evidence:\n{evidence_text}\n\nQuestion: {question}"),
            ])
        except Exception as exc:  # provider messages can contain key fragments: never echo them
            raise ServiceError(
                f"The OpenAI chat call failed ({type(exc).__name__}). Check OPENAI_API_KEY and "
                f"that model '{settings.openai_chat_model}' is available to your account "
                "(set OPENAI_CHAT_MODEL in .env).") from None

    return call


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)*", text))


def _format(evidence: list[Evidence]) -> str:
    return "\n".join(f"[{e.id}] (page {e.page}{', ' + e.table if e.table else ''}) {e.text}"
                     for e in evidence)


def answer_question(question: str, retriever: Retriever, llm: LLM) -> QueryResponse:
    evidence = retriever.retrieve(question)
    if not evidence:
        return QueryResponse(answer=NOT_FOUND)

    result = llm(question, _format(evidence))
    by_id = {e.id: e for e in evidence}
    cited = [by_id[i] for i in dict.fromkeys(result.evidence_ids) if i in by_id]
    answer = result.answer.strip()

    if not cited or answer == NOT_FOUND:
        return QueryResponse(answer=NOT_FOUND)

    # Every number in the answer must come from the cited evidence (or the question itself).
    allowed = _numbers(question).union(*(_numbers(f"page {e.page} {e.text}") for e in cited))
    if not _numbers(answer) <= allowed:
        return QueryResponse(answer=NOT_FOUND)

    sources = [Source(document=e.document, page=e.page, table=e.table, evidence=e.text)
               for e in cited]
    return QueryResponse(answer=answer, sources=sources)
