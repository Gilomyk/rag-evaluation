"""
embeddings.py

Cienki wrapper na sentence-transformers. Ładuje model raz i udostępnia
funkcje do embedowania batcha tekstów (chunki) oraz pojedynczego query.

Moduł celowo nie ma efektów ubocznych (nie zapisuje plików) - cache'owanie
embeddingów chunków robimy w indexer.py, żeby ten moduł był łatwy do
testowania w izolacji.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Wrapper na SentenceTransformer - ładuje model raz, udostępnia
    proste metody do embedowania chunków i zapytań."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        logger.info(f"Ładuję model embeddingowy: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model załadowany. Wymiar embeddingu: {self.embedding_dim}")

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embeduje listę tekstów (np. chunków dokumentów).

        Zwraca macierz numpy o kształcie (len(texts), embedding_dim),
        znormalizowaną do długości jednostkowej (norm=1), żeby cosine
        similarity można było liczyć jako zwykły iloczyn skalarny -
        to jest wymagane przez FAISS IndexFlatIP, którego użyjemy w indexer.py.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeduje pojedyncze zapytanie użytkownika.

        Zwraca wektor 1D o kształcie (embedding_dim,), znormalizowany
        tak samo jak embed_texts (spójność jest kluczowa - inaczej
        cosine similarity/iloczyn skalarny nie miałby sensu).
        """
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)


if __name__ == "__main__":
    # szybki sanity check
    model = EmbeddingModel()
    sample_texts = [
        "Quicksort is a divide-and-conquer sorting algorithm.",
        "A hash table maps keys to values using a hash function.",
    ]
    embeddings = model.embed_texts(sample_texts)
    logger.info(f"Kształt embeddingów: {embeddings.shape}")

    query_embedding = model.embed_query("How does sorting by partitioning work?")
    logger.info(f"Kształt embeddingu query: {query_embedding.shape}")

    # sprawdzenie normalizacji - norma każdego wektora powinna być bliska 1.0
    norms = np.linalg.norm(embeddings, axis=1)
    logger.info(f"Normy embeddingów (powinny być ~1.0): {norms}")