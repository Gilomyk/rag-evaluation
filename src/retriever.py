"""
retriever.py

Łączy FAISS (vector search) i BM25 (keyword search) w jeden interfejs
retrievalu z trzema trybami: vector, bm25, hybrid (RRF).

Wczytuje indeksy zbudowane przez indexer.py - nie buduje ich od nowa.
"""

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi

from src.embeddings import EmbeddingModel
from src.indexer import (
    BM25_TOKENS_PATH,
    CHUNK_LOOKUP_PATH,
    FAISS_INDEX_PATH,
    tokenize_for_bm25,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Stała wygładzająca RRF - standardowa wartość używana w literaturze
# (np. Cormack et al. 2009, "Reciprocal Rank Fusion outperforms Condorcet...")
RRF_K = 60


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_title: str
    category: str
    text: str
    score: float          # znaczenie score zależy od trybu (patrz metody niżej)
    rank: int              # pozycja w wynikach (1 = najlepszy)


class Retriever:
    """Udostępnia vector_search, bm25_search i hybrid_search na tym samym
    korpusie chunków, wczytanym z indeksów zbudowanych przez indexer.py."""

    def __init__(self, embedding_model: EmbeddingModel | None = None):
        logger.info("Wczytuję indeksy...")

        self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

        with open(BM25_TOKENS_PATH, "rb") as f:
            bm25_tokens = pickle.load(f)
        self.bm25 = BM25Okapi(bm25_tokens)

        self.chunk_lookup: dict[str, dict] = json.loads(
            CHUNK_LOOKUP_PATH.read_text(encoding="utf-8")
        )

        # Model embeddingowy potrzebny tylko do embedowania query (nie chunków -
        # te są już zaembedowane i zapisane w faiss_index)
        self.embedding_model = embedding_model or EmbeddingModel()

        logger.info(
            f"Retriever gotowy: {self.faiss_index.ntotal} chunków w FAISS, "
            f"{len(bm25_tokens)} w BM25."
        )

    def _chunk_by_position(self, position: int) -> dict:
        """Zamienia pozycję numeryczną (zwracaną przez FAISS/BM25) na pełne
        metadane chunku, korzystając ze wspólnego lookupu."""
        return self.chunk_lookup[str(position)]

    def vector_search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Wyszukiwanie semantyczne przez FAISS (cosine similarity)."""
        query_embedding = self.embedding_model.embed_query(query).reshape(1, -1)
        scores, positions = self.faiss_index.search(query_embedding, k)

        results = []
        for rank, (position, score) in enumerate(zip(positions[0], scores[0]), start=1):
            if position == -1:  # FAISS zwraca -1, gdy brakuje wyników
                continue
            chunk = self._chunk_by_position(int(position))
            results.append(RetrievalResult(
                chunk_id=chunk["chunk_id"],
                doc_title=chunk["doc_title"],
                category=chunk["category"],
                text=chunk["text"],
                score=float(score),
                rank=rank,
            ))
        return results

    def bm25_search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Wyszukiwanie leksykalne przez BM25 (dopasowanie słów kluczowych)."""
        query_tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(query_tokens)

        # get_scores zwraca score dla WSZYSTKICH chunków - bierzemy top-k pozycji
        top_positions = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        results = []
        for rank, position in enumerate(top_positions, start=1):
            chunk = self._chunk_by_position(position)
            results.append(RetrievalResult(
                chunk_id=chunk["chunk_id"],
                doc_title=chunk["doc_title"],
                category=chunk["category"],
                text=chunk["text"],
                score=float(scores[position]),
                rank=rank,
            ))
        return results

    def hybrid_search(
        self, query: str, k: int = 5, candidate_pool: int = 20
    ) -> list[RetrievalResult]:
        """
        Łączy vector_search i bm25_search przez Reciprocal Rank Fusion (RRF).

        candidate_pool: ile wyników pobieramy z KAŻDEGO z dwóch źródeł przed
        fuzją (większe niż finalne k, żeby fuzja miała z czego wybierać -
        dokument słaby w jednym rankingu, ale mocny w drugim, powinien mieć
        szansę wypłynąć na górę połączonej listy).
        """
        vector_results = self.vector_search(query, k=candidate_pool)
        bm25_results = self.bm25_search(query, k=candidate_pool)

        # RRF score: sumujemy 1/(k + rank) po wszystkich listach, w których
        # dany chunk się pojawił. Chunk nieobecny w danej liście po prostu
        # nie dostaje wkładu z tej listy (nie karzemy go dodatkowo).
        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        for result in vector_results:
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + 1.0 / (RRF_K + result.rank)
            chunk_data[result.chunk_id] = result

        for result in bm25_results:
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + 1.0 / (RRF_K + result.rank)
            chunk_data[result.chunk_id] = result

        # Sortujemy po połączonym RRF score i bierzemy finalne top-k
        sorted_chunk_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:k]

        results = []
        for rank, chunk_id in enumerate(sorted_chunk_ids, start=1):
            source = chunk_data[chunk_id]
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                doc_title=source.doc_title,
                category=source.category,
                text=source.text,
                score=rrf_scores[chunk_id],   # tu score = RRF score, nie cosine/BM25
                rank=rank,
            ))
        return results

    def search(self, query: str, strategy: str, k: int = 5) -> list[RetrievalResult]:
        """
        Ujednolicony interfejs do wyboru strategii retrievalu po nazwie.
        Przydatne w rag_pipeline.py i evaluate.py, gdzie strategia jest
        parametrem, a nie zaszytą na sztywno decyzją w kodzie.
        """
        strategies = {
            "vector": self.vector_search,
            "bm25": self.bm25_search,
            "hybrid": self.hybrid_search,
        }
        if strategy not in strategies:
            raise ValueError(f"Nieznana strategia: '{strategy}'. Dostępne: {list(strategies.keys())}")
        return strategies[strategy](query, k=k)

if __name__ == "__main__":
    retriever = Retriever()

    test_query = "How does the pivot element work in partitioning?"

    print("\n=== VECTOR SEARCH ===")
    for r in retriever.vector_search(test_query, k=3):
        print(f"[{r.rank}] {r.chunk_id} (score={r.score:.3f}) - {r.doc_title}")

    print("\n=== BM25 SEARCH ===")
    for r in retriever.bm25_search(test_query, k=3):
        print(f"[{r.rank}] {r.chunk_id} (score={r.score:.3f}) - {r.doc_title}")

    print("\n=== HYBRID SEARCH (RRF) ===")
    for r in retriever.hybrid_search(test_query, k=3):
        print(f"[{r.rank}] {r.chunk_id} (score={r.score:.3f}) - {r.doc_title}")