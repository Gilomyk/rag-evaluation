"""
chunker.py

Dzieli dokumenty tekstowe na chunki (fixed-size, word-based, z overlapem).
Zapisuje wynik jako data/documents/chunks.json - wejście dla embeddings.py / indexer.py.

Uruchomienie: python -m src.chunker
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/documents")
RAW_DIR = DATA_DIR / "raw"
METADATA_PATH = DATA_DIR / "documents_metadata.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"

# Parametry chunkowania - słowa, nie tokeny (patrz wyjaśnienie w odpowiedzi)
DEFAULT_CHUNK_SIZE = 200   # słów na chunk
DEFAULT_OVERLAP = 40       # słów zachodzenia między kolejnymi chunkami


@dataclass
class Chunk:
    chunk_id: str          # np. "quicksort_0"
    doc_title: str
    doc_filename: str
    category: str           # "core" | "distractor" - dziedziczone z dokumentu
    chunk_index: int        # numer chunku w obrębie dokumentu (od 0)
    text: str
    char_start: int          # offset w oryginalnym tekście (do debugowania/podglądu)
    char_end: int
    num_words: int


def load_documents_metadata() -> list[dict]:
    """Wczytuje metadane dokumentów zapisane przez loaders.py."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Brak {METADATA_PATH}. Najpierw uruchom: python -m src.loaders"
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[tuple[str, int, int]]:
    """
    Dzieli tekst na fragmenty po `chunk_size` słów, z zachodzeniem `overlap` słów
    między kolejnymi chunkami.

    Zwraca listę krotek (chunk_text, char_start, char_end).

    Overlap ma zapobiec sytuacji, w której istotna informacja zostanie
    "przecięta" dokładnie na granicy dwóch chunków i nie znajdzie się w całości
    w żadnym z nich.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap musi być mniejszy niż chunk_size, inaczej nie ma postępu.")

    words = text.split()
    if not words:
        return []

    chunks: list[tuple[str, int, int]] = []
    step = chunk_size - overlap

    word_start_idx = 0
    while word_start_idx < len(words):
        word_end_idx = min(word_start_idx + chunk_size, len(words))
        chunk_words = words[word_start_idx:word_end_idx]
        chunk_str = " ".join(chunk_words)

        # Odtwarzamy przybliżony offset znakowy w oryginalnym tekście
        # (przybliżony, bo split()/join() normalizuje białe znaki - wystarczające
        # do podglądu/debugowania, nie jest krytyczne dla działania systemu)
        char_start = text.find(chunk_words[0], 0 if not chunks else chunks[-1][2] - 200)
        char_end = char_start + len(chunk_str)

        chunks.append((chunk_str, max(char_start, 0), char_end))

        if word_end_idx == len(words):
            break
        word_start_idx += step

    return chunks


def chunk_documents(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Wczytuje wszystkie dokumenty z raw/ i dzieli je na chunki."""
    documents_metadata = load_documents_metadata()
    all_chunks: list[Chunk] = []

    for doc_meta in documents_metadata:
        filepath = RAW_DIR / doc_meta["filename"]
        text = filepath.read_text(encoding="utf-8")

        raw_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        doc_slug = Path(doc_meta["filename"]).stem
        for idx, (chunk_str, char_start, char_end) in enumerate(raw_chunks):
            chunk = Chunk(
                chunk_id=f"{doc_slug}_{idx}",
                doc_title=doc_meta["title"],
                doc_filename=doc_meta["filename"],
                category=doc_meta["category"],
                chunk_index=idx,
                text=chunk_str,
                char_start=char_start,
                char_end=char_end,
                num_words=len(chunk_str.split()),
            )
            all_chunks.append(chunk)

        logger.info(f"{doc_meta['title']}: {len(raw_chunks)} chunków")

    CHUNKS_PATH.write_text(
        json.dumps([asdict(c) for c in all_chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Zapisano {len(all_chunks)} chunków łącznie -> {CHUNKS_PATH}")

    return all_chunks


if __name__ == "__main__":
    chunk_documents()