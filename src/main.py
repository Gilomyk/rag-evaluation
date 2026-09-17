"""
main.py

CLI entrypoint spinający cały projekt RAG + Evaluation w jedno miejsce.

Użycie:
    python -m src.main build-index
    python -m src.main ask "How does quicksort partition an array?" --strategy hybrid --k 5
    python -m src.main evaluate --k 5
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def cmd_build_index(args: argparse.Namespace) -> None:
    """Pełen pipeline przygotowania danych: pobranie dokumentów -> chunking -> indeksowanie.

    Kroki są rozdzielone (każdy moduł czyta/zapisuje własne pliki pośrednie),
    więc jeśli np. tylko zmienisz parametry chunkowania, możesz odpalić
    ponownie samo `python -m src.chunker` + `python -m src.indexer`, bez
    ponownego pobierania dokumentów z Wikipedii.
    """
    from src.chunker import chunk_documents
    from src.indexer import build_indexes
    from src.loaders import load_all_documents

    logger.info("Krok 1/3: Pobieram dokumenty z Wikipedii...")
    load_all_documents()

    logger.info("Krok 2/3: Dzielę dokumenty na chunki...")
    chunk_documents()

    logger.info("Krok 3/3: Buduję indeksy FAISS + BM25...")
    build_indexes()

    logger.info("Gotowe. Indeksy zapisane w data/index/.")


def cmd_ask(args: argparse.Namespace) -> None:
    """Zadaje pojedyncze pytanie do systemu RAG i wypisuje odpowiedź wraz ze źródłami."""
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    result = pipeline.ask(args.question, strategy=args.strategy, k=args.k)

    print(f"\nPytanie: {result.question}")
    print(f"Strategia: {args.strategy} | k={args.k}")
    print(f"\nOdpowiedź:\n{result.answer}")
    print(f"\nŹródła:")
    for idx, chunk in enumerate(result.context_chunks, start=1):
        print(f"  [{idx}] {chunk.chunk_id} - {chunk.doc_title}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Uruchamia pełną ewaluację (retrieval + answer quality) na golden dataset."""
    from src.evaluate import run_full_evaluation

    run_full_evaluation(k=args.k)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-evaluation",
        description="RAG + Evaluation - mały system RAG z porównaniem strategii retrievalu.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build-index
    parser_build = subparsers.add_parser(
        "build-index", help="Pobiera dokumenty, dzieli na chunki, buduje indeksy FAISS + BM25."
    )
    parser_build.set_defaults(func=cmd_build_index)

    # ask
    parser_ask = subparsers.add_parser("ask", help="Zadaje pojedyncze pytanie do systemu RAG.")
    parser_ask.add_argument("question", type=str, help="Pytanie do zadania.")
    parser_ask.add_argument(
        "--strategy",
        type=str,
        default="hybrid",
        choices=["vector", "bm25", "hybrid"],
        help="Strategia retrievalu (domyślnie: hybrid).",
    )
    parser_ask.add_argument(
        "--k", type=int, default=5, help="Liczba chunków do pobrania jako kontekst (domyślnie: 5)."
    )
    parser_ask.set_defaults(func=cmd_ask)

    # evaluate
    parser_eval = subparsers.add_parser(
        "evaluate", help="Uruchamia pełną ewaluację (retrieval + answer quality) na golden dataset."
    )
    parser_eval.add_argument(
        "--k", type=int, default=5, help="Liczba chunków do pobrania jako kontekst (domyślnie: 5)."
    )
    parser_eval.set_defaults(func=cmd_evaluate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()