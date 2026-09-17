"""
rag_pipeline.py

Spina retriever.py i llm.py w jedno wywołanie: question -> RAGAnswer.

Inicjalizuje EmbeddingModel i Retriever raz i trzyma je w pamięci - kluczowe
przy wielokrotnych wywołaniach (np. w evaluate.py na całym golden datasecie),
żeby nie ładować modelu embeddingowego za każdym razem od nowa.
"""

import logging

from src.embeddings import EmbeddingModel
from src.llm import DEFAULT_MODEL, RAGAnswer, generate_answer
from src.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_STRATEGY = "hybrid"
DEFAULT_K = 5


class RAGPipeline:
    """Pełny pipeline RAG: pytanie -> retrieval -> generacja odpowiedzi.

    Ładuje model embeddingowy i indeksy raz przy inicjalizacji - kolejne
    wywołania ask() są tanie (bez ponownego ładowania modelu)."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        llm_model_name: str = DEFAULT_MODEL,
    ):
        logger.info("Inicjalizuję RAGPipeline...")
        self.embedding_model = embedding_model or EmbeddingModel()
        self.retriever = Retriever(embedding_model=self.embedding_model)
        self.llm_model_name = llm_model_name
        logger.info("RAGPipeline gotowy.")

    def ask(
        self,
        question: str,
        strategy: str = DEFAULT_STRATEGY,
        k: int = DEFAULT_K,
    ) -> RAGAnswer:
        """
        Wykonuje pełen cykl: retrieval (wybraną strategią) -> generacja
        odpowiedzi przez LLM na podstawie znalezionego kontekstu.
        """
        logger.info(f"Pytanie: '{question}' | strategia={strategy} | k={k}")

        retrieved_chunks = self.retriever.search(question, strategy=strategy, k=k)

        if not retrieved_chunks:
            logger.warning("Retriever nie znalazł żadnych chunków dla tego pytania.")

        answer = generate_answer(
            question=question,
            context_chunks=retrieved_chunks,
            model_name=self.llm_model_name,
        )
        return answer


if __name__ == "__main__":
    pipeline = RAGPipeline()

    test_question = "What is the difference between the Lomuto and Hoare partition schemes?"

    for strategy in ["vector", "bm25", "hybrid"]:
        print(f"\n{'=' * 60}")
        print(f"STRATEGIA: {strategy}")
        print("=" * 60)

        result = pipeline.ask(test_question, strategy=strategy, k=3)

        print(f"\nOdpowiedź:\n{result.answer}")
        print(f"\nŹródła:")
        for idx, chunk in enumerate(result.context_chunks, start=1):
            print(f"  [{idx}] {chunk.chunk_id} - {chunk.doc_title}")