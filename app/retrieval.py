"""Hybrid retrieval (BM25 + Chroma) plus exact structured lookup over the table records.

Order of trust:
1. Structured lookup: the question's metal + food words are matched against the
   validated table rows. A row is used only if EVERY food word in the question is
   found in the row's food name (so "lead in rice" never returns a different food).
2. Prose chunks found by BM25/vector search, used only when they share the
   question's content words.
If nothing qualifies, `retrieve` returns [] and the caller answers with the fallback.
"""
import re
from dataclasses import dataclass
from typing import Callable

from app.indexing import BM25Index, tokenize
from app.schemas import Chunk, TableRecord

# Words that describe the *kind* of question, not the food being asked about.
STOP = set(tokenize(
    "what is are was the a an of in for on to and or with does do can there any tell me give show "
    "find about which how much many maximum minimum permissible permitted allowed acceptable limit "
    "level content concentration amount value ppm mg kg l per according under regulation fssai "
    "contaminant metal heavy food regulated restricted specified listed required safe"))
MAX_ROWS = 12


@dataclass
class Evidence:
    id: str                 # E1, E2... label shown to the LLM
    kind: str               # 'row' or 'prose'
    text: str
    document: str
    page: int
    table: str | None
    page_end: int | None = None


Vector = Callable[[str, int], list[str]]


class Retriever:
    def __init__(self, records: list[TableRecord], chunks: list[Chunk],
                 vector_search: Vector | None = None) -> None:
        self.records = {r.id: r for r in records}
        self.chunks = {c.id: c for c in chunks}
        self.bm25 = BM25Index(chunks)
        self.vector_search = vector_search
        names = sorted({r.parameter for r in records}, key=len, reverse=True)
        self.food_vocab = {w for r in records for w in tokenize(r.food)}
        self.known_vocab = self.food_vocab | {w for c in chunks for w in tokenize(c.text)}
        # "Methyl Mercury" also matches "methyl-mercury" / "methylmercury".
        self.metals = {n: re.compile(r"\b" + r"[ -]?".join(map(re.escape, n.lower().split())) + r"\b")
                       for n in names}

    # ---- question analysis ----
    def parse(self, question: str) -> tuple[list[str], list[str]]:
        """Return (metals mentioned, food words left over)."""
        q = question.lower()
        found: list[str] = []
        for name, pattern in self.metals.items():       # longest first: "methyl mercury" before "mercury"
            if pattern.search(q):
                found.append(name)
                q = pattern.sub(" ", q)
        words = [w for w in tokenize(q) if w not in STOP]
        return found, words

    # ---- structured lookup ----
    def food_words(self, words: list[str]) -> list[str]:
        """Question words that occur in some food name of the table."""
        return [w for w in words if w in self.food_vocab]

    def lookup(self, metals: list[str], words: list[str]) -> list[TableRecord]:
        """Rows whose food name contains every food word of the question."""
        if any(w not in self.known_vocab for w in words):
            return []   # asks about something the document never mentions (e.g. aflatoxin)
        pool = [r for r in self.records.values() if not metals or r.parameter in metals]
        need = set(self.food_words(words))
        if not need:
            return pool if metals and len(pool) <= 8 else []   # e.g. "methyl mercury limit"
        hits = [r for r in pool if need <= set(tokenize(r.food))]
        return sorted(hits, key=lambda r: len(tokenize(r.food)))[:MAX_ROWS]

    # ---- hybrid search over chunks ----
    def hybrid(self, question: str, k: int = 8) -> list[str]:
        rankings = [[i for i, _ in self.bm25.search(question, k)]]
        if self.vector_search:
            rankings.append(self.vector_search(question, k))
        score: dict[str, float] = {}
        for ranking in rankings:                         # reciprocal rank fusion
            for rank, cid in enumerate(ranking):
                score[cid] = score.get(cid, 0) + 1 / (60 + rank)
        return sorted(score, key=score.get, reverse=True)

    def retrieve(self, question: str) -> list[Evidence]:
        metals, words = self.parse(question)
        rows = self.lookup(metals, words)

        # Words already explained by table rows should not be required of the prose.
        used = set(self.food_words(words)) if rows else set()
        content = set(words) - used
        prose: list[Chunk] = []
        for cid in self.hybrid(question):
            c = self.chunks.get(cid)
            if not c or c.kind != "prose":
                continue
            # Only keep prose that actually shares the question's wording.
            if content and len(content & set(tokenize(c.text))) / len(content) >= 0.5:
                prose.append(c)

        evidence: list[Evidence] = []
        for r in rows:
            evidence.append(Evidence("", "row", self.row_text(r), r.document, r.page, r.table, r.page_end))
        for c in prose[:3]:
            evidence.append(Evidence("", "prose", c.text, c.document, c.page, None))
        for n, e in enumerate(evidence, start=1):
            e.id = f"E{n}"
        return evidence

    @staticmethod
    def row_text(r: TableRecord) -> str:
        text = (f"Metal: {r.parameter} | Food: {r.food} | Limit: {r.limit_text} "
                f"({r.limit_type}; unit {r.unit})")
        if r.page_end:
            text += f" | row continues on page {r.page_end}"
        return text
