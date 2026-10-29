"""Pydantic models for the API and for parsed table records / chunks."""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Source(BaseModel):
    document: str
    page: int
    table: str | None = None
    evidence: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = []


class TableRecord(BaseModel):
    """One validated row of the regulatory table."""
    id: str
    parameter: str            # metal contaminant, e.g. "Lead"
    food: str                 # article of food
    limit_value: float | None  # numeric part of the limit
    limit_text: str           # limit exactly as printed (incl. condition)
    unit: str
    limit_type: str = "maximum"
    condition: str = ""       # e.g. "on dry matter basis"
    regulation: str
    document: str
    page: int
    page_end: int | None = None  # set when a row continues on the next page
    table: str
    original_row: str
    needs_review: bool = False
    review_note: str = ""


class Chunk(BaseModel):
    """A unit that is indexed (BM25 + Chroma). Kind is 'row' or 'prose'."""
    id: str
    kind: str
    text: str
    document: str
    page: int
    table: str | None = None
    parameter: str | None = None
