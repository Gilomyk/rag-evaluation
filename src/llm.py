"""
llm.py

Wrapper na lokalny model Ollama (llama3.2). Buduje prompt z kontekstu
(chunków znalezionych przez retriever) i generuje odpowiedź z cytowaniami
źródeł.

Zakłada uruchomiony `ollama serve` i ściągnięty model (domyślnie llama3.2).
"""

import logging
from dataclasses import dataclass

import ollama

from src.retriever import RetrievalResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.2"

SYSTEM_PROMPT = """You are a helpful assistant answering questions using ONLY the provided context.

Rules:
- Answer using only information found in the numbered context passages below.
- After each claim, cite the passage number it came from, like [1] or [2].
- If the context does not contain enough information to answer the question, say so explicitly instead of guessing.
- Do not use any knowledge outside the provided context, even if you know the answer.
- Be concise and direct."""


@dataclass
class RAGAnswer:
    question: str
    answer: str
    context_chunks: list[RetrievalResult]  # chunki użyte jako kontekst, w kolejności [1], [2], ...
    model_name: str


def build_context_block(chunks: list[RetrievalResult]) -> str:
    """Formatuje chunki jako numerowany blok kontekstu do wstrzyknięcia w prompt.

    Numeracja od 1 (nie od 0), żeby cytowania w stylu [1] były naturalne
    dla modelu i czytelne dla człowieka."""
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(f"[{idx}] (source: {chunk.doc_title})\n{chunk.text}")
    return "\n\n".join(lines)


def build_user_prompt(question: str, chunks: list[RetrievalResult]) -> str:
    context_block = build_context_block(chunks)
    return f"""Context passages:

{context_block}

Question: {question}

Answer (remember to cite passage numbers like [1]):"""


def generate_answer(
    question: str,
    context_chunks: list[RetrievalResult],
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.1,
) -> RAGAnswer:
    """
    Generuje odpowiedź na pytanie, korzystając wyłącznie z podanych chunków
    jako kontekstu.

    temperature=0.1 (nisko, ale nie zero) - chcemy odpowiedzi maksymalnie
    zdeterminowane przez kontekst, nie kreatywne, ale zupełne zero czasem
    prowadzi do zapętleń w niektórych modelach, więc zostawiamy mały margines.
    """
    user_prompt = build_user_prompt(question, context_chunks)

    logger.info(f"Wysyłam zapytanie do {model_name} ({len(context_chunks)} chunków kontekstu)...")

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": temperature},
    )

    answer_text = response["message"]["content"]

    return RAGAnswer(
        question=question,
        answer=answer_text,
        context_chunks=context_chunks,
        model_name=model_name,
    )


if __name__ == "__main__":
    # szybki sanity check - wymaga zbudowanych indeksów (indexer.py) i
    # uruchomionej Ollamy z modelem llama3.2
    from src.retriever import Retriever

    retriever = Retriever()
    test_question = "How does the pivot element work in quicksort partitioning?"

    retrieved_chunks = retriever.hybrid_search(test_question, k=3)

    result = generate_answer(test_question, retrieved_chunks)

    print(f"\nPytanie: {result.question}")
    print(f"\nOdpowiedź:\n{result.answer}")
    print(f"\nUżyte źródła:")
    for idx, chunk in enumerate(result.context_chunks, start=1):
        print(f"  [{idx}] {chunk.chunk_id} - {chunk.doc_title}")