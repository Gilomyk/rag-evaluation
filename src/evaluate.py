"""
evaluate.py

Ewaluacja systemu RAG w dwóch niezależnych częściach:

1. Retrieval evaluation (bez LLM): hit_rate@k, recall@k, MRR dla strategii
   vector / bm25 / hybrid - testuje wyłącznie jakość wyszukiwania.
2. Answer quality evaluation (wymaga LLM): answer_correctness (embedding
   similarity do reference_answer) oraz groundedness (LLM-as-judge) dla
   wygenerowanych odpowiedzi.

Uruchomienie: python -m src.evaluate
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import ollama

from src.embeddings import EmbeddingModel
from src.llm import DEFAULT_MODEL
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path("data/evaluation/golden_dataset.json")
RESULTS_PATH = Path("data/evaluation/results.json")

STRATEGIES = ["vector", "bm25", "hybrid"]
DEFAULT_K = 5

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator judging whether an answer is fully supported by given context passages.

Respond with ONLY a JSON object, no other text, no markdown formatting:
{"score": <float between 0.0 and 1.0>, "reasoning": "<one short sentence>"}

Scoring guide:
- 1.0: every claim in the answer is directly supported by the context.
- 0.5: the answer is partially supported, some claims are not backed by the context.
- 0.0: the answer contains claims not supported by the context at all (hallucination), or ignores the context entirely."""


def load_golden_dataset() -> list[dict]:
    if not GOLDEN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Brak {GOLDEN_DATASET_PATH}. Zapisz golden dataset pod tą ścieżką."
        )
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


# ---------- RETRIEVAL EVALUATION ----------

@dataclass
class RetrievalEvalResult:
    question_id: str
    strategy: str
    expected_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    hit: bool
    recall: float
    reciprocal_rank: float


def evaluate_retrieval_for_question(
    retriever: Retriever, question_entry: dict, strategy: str, k: int
) -> RetrievalEvalResult:
    """Sprawdza, czy retriever (dana strategia) znalazł oczekiwane chunki
    dla jednego pytania z golden datasetu."""
    results = retriever.search(question_entry["question"], strategy=strategy, k=k)
    retrieved_ids = [r.chunk_id for r in results]
    expected_ids = set(question_entry["expected_chunk_ids"])

    found = [cid for cid in retrieved_ids if cid in expected_ids]
    hit = len(found) > 0
    recall = len(set(found)) / len(expected_ids) if expected_ids else 0.0

    # Reciprocal rank: 1/pozycja pierwszego trafienia, 0.0 jeśli brak trafienia.
    # To uzupełnia hit_rate/recall o informację "JAK WYSOKO" był dobry wynik.
    reciprocal_rank = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in expected_ids:
            reciprocal_rank = 1.0 / rank
            break

    return RetrievalEvalResult(
        question_id=question_entry["question_id"],
        strategy=strategy,
        expected_chunk_ids=list(expected_ids),
        retrieved_chunk_ids=retrieved_ids,
        hit=hit,
        recall=recall,
        reciprocal_rank=reciprocal_rank,
    )


def evaluate_retrieval(
    retriever: Retriever, golden_dataset: list[dict], k: int = DEFAULT_K
) -> tuple[list[RetrievalEvalResult], dict]:
    """Uruchamia retrieval evaluation dla wszystkich pytań x wszystkich strategii.
    Zwraca listę szczegółowych wyników oraz zagregowane metryki per strategia."""
    all_results: list[RetrievalEvalResult] = []

    for strategy in STRATEGIES:
        for question_entry in golden_dataset:
            result = evaluate_retrieval_for_question(retriever, question_entry, strategy, k)
            all_results.append(result)

    summary = {}
    for strategy in STRATEGIES:
        strategy_results = [r for r in all_results if r.strategy == strategy]
        n = len(strategy_results)
        summary[strategy] = {
            "hit_rate": sum(r.hit for r in strategy_results) / n,
            "recall": sum(r.recall for r in strategy_results) / n,
            "mrr": sum(r.reciprocal_rank for r in strategy_results) / n,
            "n_questions": n,
        }

    return all_results, summary


# ---------- ANSWER QUALITY EVALUATION ----------

@dataclass
class AnswerEvalResult:
    question_id: str
    strategy: str
    question: str
    generated_answer: str
    reference_answer: str
    answer_correctness: float
    groundedness: float
    groundedness_reasoning: str


def compute_answer_correctness(
    embedding_model: EmbeddingModel, generated_answer: str, reference_answer: str
) -> float:
    """Cosine similarity między embeddingiem wygenerowanej odpowiedzi a odpowiedzią
    referencyjną. Embeddingi są znormalizowane (patrz embeddings.py), więc
    iloczyn skalarny = cosine similarity."""
    gen_emb = embedding_model.embed_query(generated_answer)
    ref_emb = embedding_model.embed_query(reference_answer)
    return float(np.dot(gen_emb, ref_emb))


def judge_groundedness(
    question: str, context_chunks: list, answer: str, judge_model: str = DEFAULT_MODEL
) -> tuple[float, str]:
    """
    LLM-as-judge: osobne wywołanie modelu w roli sędziego, oceniające czy
    wygenerowana odpowiedź jest w pełni poparta dostarczonym kontekstem
    (bez halucynacji wykraczających poza niego).

    Świadomie oceniamy względem CAŁEGO dostarczonego kontekstu, nie tylko
    chunków faktycznie zacytowanych w odpowiedzi - pytamy "czy to, co model
    powiedział, da się uzasadnić tym, co miał do dyspozycji", a nie tylko
    "czy cytowania [1]/[2] są poprawne" (to osobny, węższy problem).
    """
    context_block = "\n\n".join(
        f"[{i}] {c.text}" for i, c in enumerate(context_chunks, start=1)
    )
    user_prompt = f"""Context passages:

{context_block}

Question: {question}

Answer to evaluate: {answer}

Judge whether the answer is fully supported by the context passages above."""

    response = ollama.chat(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.0},
    )
    raw = response["message"]["content"].strip()

    # Obrona przed markdown code fences, które model czasem dodaje mimo
    # jawnej instrukcji "no markdown formatting" w prompt systemowym.
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
        return float(parsed["score"]), str(parsed.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Nie udało się sparsować odpowiedzi judge'a: {raw!r} ({e})")
        return 0.0, f"PARSE_ERROR: {raw[:100]}"


def evaluate_answers(
    pipeline: RAGPipeline,
    embedding_model: EmbeddingModel,
    golden_dataset: list[dict],
    k: int = DEFAULT_K,
) -> tuple[list[AnswerEvalResult], dict]:
    """Generuje odpowiedzi dla każdego pytania x każdej strategii i ocenia je
    metryką matematyczną (answer_correctness) oraz LLM-as-judge (groundedness)."""
    all_results: list[AnswerEvalResult] = []

    for strategy in STRATEGIES:
        for question_entry in golden_dataset:
            logger.info(f"[{strategy}] Generuję odpowiedź dla {question_entry['question_id']}...")

            rag_answer = pipeline.ask(question_entry["question"], strategy=strategy, k=k)

            correctness = compute_answer_correctness(
                embedding_model, rag_answer.answer, question_entry["reference_answer"]
            )
            groundedness, reasoning = judge_groundedness(
                question_entry["question"], rag_answer.context_chunks, rag_answer.answer
            )

            all_results.append(AnswerEvalResult(
                question_id=question_entry["question_id"],
                strategy=strategy,
                question=question_entry["question"],
                generated_answer=rag_answer.answer,
                reference_answer=question_entry["reference_answer"],
                answer_correctness=correctness,
                groundedness=groundedness,
                groundedness_reasoning=reasoning,
            ))

    summary = {}
    for strategy in STRATEGIES:
        strategy_results = [r for r in all_results if r.strategy == strategy]
        n = len(strategy_results)
        summary[strategy] = {
            "avg_answer_correctness": sum(r.answer_correctness for r in strategy_results) / n,
            "avg_groundedness": sum(r.groundedness for r in strategy_results) / n,
            "n_questions": n,
        }

    return all_results, summary


# ---------- MAIN ----------

def print_summary_table(retrieval_summary: dict, answer_summary: dict) -> None:
    print("\n" + "=" * 70)
    print("RETRIEVAL METRICS")
    print("=" * 70)
    print(f"{'Strategy':<10} {'Hit Rate':<12} {'Recall':<12} {'MRR':<12}")
    for strategy, m in retrieval_summary.items():
        print(f"{strategy:<10} {m['hit_rate']:<12.3f} {m['recall']:<12.3f} {m['mrr']:<12.3f}")

    print("\n" + "=" * 70)
    print("ANSWER QUALITY METRICS")
    print("=" * 70)
    print(f"{'Strategy':<10} {'Correctness':<14} {'Groundedness':<14}")
    for strategy, m in answer_summary.items():
        print(f"{strategy:<10} {m['avg_answer_correctness']:<14.3f} {m['avg_groundedness']:<14.3f}")
    print()


def run_full_evaluation(k: int = DEFAULT_K) -> None:
    golden_dataset = load_golden_dataset()
    logger.info(f"Wczytano {len(golden_dataset)} pytań z golden datasetu.")

    embedding_model = EmbeddingModel()
    retriever = Retriever(embedding_model=embedding_model)
    pipeline = RAGPipeline(embedding_model=embedding_model)

    logger.info("=== Ewaluacja retrievalu (bez LLM) ===")
    retrieval_results, retrieval_summary = evaluate_retrieval(retriever, golden_dataset, k=k)

    logger.info("=== Ewaluacja jakości odpowiedzi (generacja + judge) ===")
    answer_results, answer_summary = evaluate_answers(pipeline, embedding_model, golden_dataset, k=k)

    print_summary_table(retrieval_summary, answer_summary)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps({
            "k": k,
            "retrieval_summary": retrieval_summary,
            "answer_summary": answer_summary,
            "retrieval_details": [asdict(r) for r in retrieval_results],
            "answer_details": [asdict(r) for r in answer_results],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Pełne wyniki zapisane -> {RESULTS_PATH}")


if __name__ == "__main__":
    run_full_evaluation()