"""FastAPI backend:  GET /health, POST /query."""
from functools import lru_cache

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.answer import answer_question, make_llm
from app.config import ServiceError, settings
from app.indexing import load_sqlite, make_vector_search
from app.retrieval import Retriever
from app.schemas import QueryRequest, QueryResponse

app = FastAPI(title="FSSAI Regulatory Assistant")


@lru_cache
def get_retriever() -> Retriever:
    db = settings.resolve(settings.sqlite_path)
    if not db.exists():
        raise ServiceError("No index found. Run: uv run python -m scripts.ingest")
    records, chunks = load_sqlite(db)
    return Retriever(records, chunks, make_vector_search(settings))


@app.get("/health")
def health() -> dict:
    db = settings.resolve(settings.sqlite_path)
    indexed = db.exists()
    records = chunks = 0
    if indexed:
        r, c = load_sqlite(db)
        records, chunks = len(r), len(c)
    return {"status": "ok", "document": settings.resolve(settings.pdf_path).name,
            "indexed": indexed, "table_rows": records, "chunks": chunks,
            "chat_model": settings.openai_chat_model, "api_key_set": bool(settings.openai_api_key)}


@lru_cache
def get_llm():
    return make_llm(settings)


@app.exception_handler(ServiceError)
def service_error(_: Request, exc: ServiceError) -> JSONResponse:
    """Config/provider problems: a clear, secret-free message and no invented answer."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, retriever: Retriever = Depends(get_retriever),
          llm=Depends(get_llm)) -> QueryResponse:
    return answer_question(req.question, retriever, llm)
