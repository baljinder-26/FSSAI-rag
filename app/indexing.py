"""Persist records (SQLite), embed chunks (Chroma), and keyword-search chunks (BM25)."""
import re
import sqlite3
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.config import ServiceError, Settings
from app.schemas import Chunk, TableRecord


def tokenize(text: str) -> list[str]:
    """Lower-case words with a light plural stripper (milks -> milk, vegetables -> vegetable)."""
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) > 3 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


# ---------- SQLite ----------

def save_sqlite(path: Path, records: list[TableRecord], chunks: list[Chunk]) -> None:
    """Rewrite both tables, so running ingestion twice never duplicates rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript(
            "DROP TABLE IF EXISTS records; DROP TABLE IF EXISTS chunks;"
            "CREATE TABLE records (id TEXT PRIMARY KEY, parameter TEXT, food TEXT, page INTEGER,"
            " limit_value REAL, needs_review INTEGER, data TEXT);"
            "CREATE TABLE chunks (id TEXT PRIMARY KEY, kind TEXT, data TEXT);"
        )
        db.executemany(
            "INSERT INTO records VALUES (?,?,?,?,?,?,?)",
            [(r.id, r.parameter, r.food, r.page, r.limit_value, int(r.needs_review),
              r.model_dump_json()) for r in records])
        db.executemany("INSERT INTO chunks VALUES (?,?,?)",
                       [(c.id, c.kind, c.model_dump_json()) for c in chunks])


def load_sqlite(path: Path) -> tuple[list[TableRecord], list[Chunk]]:
    with sqlite3.connect(path) as db:
        records = [TableRecord.model_validate_json(d) for (d,) in
                   db.execute("SELECT data FROM records ORDER BY rowid")]
        chunks = [Chunk.model_validate_json(d) for (d,) in
                  db.execute("SELECT data FROM chunks ORDER BY rowid")]
    return records, chunks


# ---------- Chroma (dense vectors) ----------

def _chroma(settings: Settings):
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model,
                                  api_key=settings.openai_api_key or None)
    return Chroma(collection_name="fssai", embedding_function=embeddings,
                  persist_directory=str(settings.resolve(settings.chroma_path)))


def build_chroma(settings: Settings, chunks: list[Chunk]) -> None:
    store = _chroma(settings)
    store.delete_collection()          # start clean -> repeatable ingestion
    store = _chroma(settings)
    for i in range(0, len(chunks), 100):
        batch = chunks[i:i + 100]
        store.add_texts(
            texts=[c.text for c in batch],
            ids=[c.id for c in batch],
            metadatas=[{"kind": c.kind, "page": c.page, "parameter": c.parameter or ""} for c in batch])


def make_vector_search(settings: Settings):
    """Return fn(query, k) -> list of chunk ids ranked by embedding similarity."""
    store = _chroma(settings)

    def search(query: str, k: int) -> list[str]:
        try:
            return [d.id for d in store.similarity_search(query, k=k) if d.id]
        except Exception as exc:  # never echo the provider message: it can contain key fragments
            raise ServiceError(
                f"Embedding search failed ({type(exc).__name__}). Check OPENAI_API_KEY, "
                "OPENAI_EMBEDDING_MODEL and that scripts.ingest has been run.") from None

    return search


# ---------- BM25 (keywords) ----------

class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.ids = [c.id for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0]
