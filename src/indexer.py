"""
indexer.py

Buduje dwa niezależne indeksy z chunków dokumentów:
- FAISS (dense/vector search) na znormalizowanych embeddingach
- BM25 (sparse/keyword search) na tokenizowanym tekście

Zapisuje oba na dysk, żeby retriever.py mógł je wczytać bez ponownego
budowania przy każdym uruchomieniu.

Uruchomienie: python -m src.indexer
"""

import json
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.embeddings import EmbeddingModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/documents")
CHUNKS_PATH = DATA_DIR / "chunks.json"

INDEX_DIR = Path("data/index")
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
BM25_TOKENS_PATH = INDEX_DIR / "bm25_tokens.pkl"
CHUNK_LOOKUP_PATH = INDEX_DIR / "chunk_lookup.json"


def load_chunks() -> list[dict]:
    """Wczytuje chunki zapisane przez chunker.py."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Brak {CHUNKS_PATH}. Najpierw uruchom: python -m src.chunker"
        )
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def tokenize_for_bm25(text: str) -> list[str]:
    """
    Prosta tokenizacja na potrzeby BM25: lowercase + split po białych znakach.

    BM25 to algorytm oparty na dopasowaniu słów kluczowych, więc nie
    potrzebuje zaawansowanej tokenizacji (jak np. tokenizer modelu
    transformerowego) - prosty split wystarcza i jest łatwy do wytłumaczenia.
    Celowo nie usuwamy stopwords ani nie robimy stemmingu na tym etapie MVP,
    żeby zachować prostotę; BM25 samo w sobie radzi sobie z tym nieźle
    poprzez wagowanie rzadkości termów (IDF).
    """
    return text.lower().split()


def build_faiss_index(chunks: list[dict], embedding_model: EmbeddingModel) -> faiss.Index:
    """
    Embeduje wszystkie chunki i buduje indeks FAISS typu IndexFlatIP
    (dokładne wyszukiwanie po iloczynie skalarnym = cosine similarity,
    bo embeddingi są znormalizowane - patrz embeddings.py).

    IndexFlatIP jest wystarczający dla ~150-200 dokumentów: dokładny,
    prosty, nie wymaga treningu (w przeciwieństwie do IndexIVF), a przy
    tej skali różnica w szybkości względem indeksów przybliżonych jest
    pomijalna.
    """
    texts = [chunk["text"] for chunk in chunks]
    logger.info(f"Embeduję {len(texts)} chunków...")
    embeddings = embedding_model.embed_texts(texts)

    index = faiss.IndexFlatIP(embedding_model.embedding_dim)
    index.add(embeddings)
    logger.info(f"Zbudowano FAISS index: {index.ntotal} wektorów, dim={embedding_model.embedding_dim}")

    return index


def build_bm25_tokens(chunks: list[dict]) -> list[list[str]]:
    """Tokenizuje wszystkie chunki na potrzeby BM25."""
    tokenized = [tokenize_for_bm25(chunk["text"]) for chunk in chunks]
    logger.info(f"Stokenizowano {len(tokenized)} chunków dla BM25.")
    return tokenized


def build_indexes(model_name: str | None = None) -> None:
    """Buduje i zapisuje na dysk oba indeksy (FAISS + BM25) oraz lookup chunków."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()

    embedding_model = EmbeddingModel(model_name) if model_name else EmbeddingModel()

    # --- FAISS ---
    faiss_index = build_faiss_index(chunks, embedding_model)
    faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
    logger.info(f"Zapisano FAISS index -> {FAISS_INDEX_PATH}")

    # --- BM25 ---
    # Zapisujemy same tokeny, nie obiekt BM25Okapi - budowa z tokenów jest
    # szybka (ułamek sekundy przy naszej skali), więc nie ma sensu
    # picklować całego obiektu, wystarczy input do jego konstruktora.
    bm25_tokens = build_bm25_tokens(chunks)
    with open(BM25_TOKENS_PATH, "wb") as f:
        pickle.dump(bm25_tokens, f)
    logger.info(f"Zapisano tokeny BM25 -> {BM25_TOKENS_PATH}")

    # --- Chunk lookup ---
    # Mapping: pozycja w indeksie (0, 1, 2, ...) -> pełne metadane chunku.
    # FAISS i BM25 obie zwracają wyniki jako indeksy pozycyjne (int), więc
    # ten sam lookup działa dla obu - kluczowe jest zachowanie identycznej
    # kolejności chunks list przy budowie FAISS, BM25 i lookup (co gwarantujemy,
    # bo wszystkie trzy iterują po tej samej liście `chunks` w tej samej kolejności).
    chunk_lookup = {str(idx): chunk for idx, chunk in enumerate(chunks)}
    CHUNK_LOOKUP_PATH.write_text(
        json.dumps(chunk_lookup, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Zapisano chunk lookup ({len(chunk_lookup)} wpisów) -> {CHUNK_LOOKUP_PATH}")

    logger.info("Indeksowanie zakończone.")


if __name__ == "__main__":
    build_indexes()