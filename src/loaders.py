"""
loaders.py

Pobiera artykuły z angielskiej Wikipedii i zapisuje je jako pliki tekstowe
wraz z metadanymi (tytuł, URL, kategoria: core/distractor).

Uruchomienie: python -m src.loaders
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import wikipediaapi

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Ścieżki wyjściowe
DATA_DIR = Path("data/documents")
RAW_DIR = DATA_DIR / "raw"
METADATA_PATH = DATA_DIR / "documents_metadata.json"

# "core" -> artykuły, do których będziemy pisać pytania w golden dataset
CORE_ARTICLES = [
    "Quicksort",
    "Merge sort",
    "Bubble sort",
    "Binary search algorithm",
    "Binary search tree",
    "AVL tree",
    "Heap (data structure)",
    "Hash table",
    "Linked list",
    "Stack (abstract data type)",
    "Queue (abstract data type)",
    "Graph (discrete mathematics)",
    "Dijkstra's algorithm",
    "Depth-first search",
    "Dynamic programming",
]

# "distractor" -> szum w indeksie, celowo spoza dziedziny, bez pytań w golden dataset
DISTRACTOR_ARTICLES = [
    "Italian cuisine",
    "Photosynthesis",
    "Renaissance",
    "Volcano",
    "History of chess",
]


@dataclass
class DocumentMetadata:
    title: str
    filename: str
    url: str
    category: str  # "core" | "distractor"
    num_characters: int


def slugify(title: str) -> str:
    """Zamienia tytuł artykułu na bezpieczną nazwę pliku, np.
    'Binary search tree' -> 'binary_search_tree'."""
    safe = title.lower()
    safe = safe.replace("(", "").replace(")", "")
    safe = safe.replace("'", "")
    safe = "_".join(safe.split())
    return safe


def fetch_article(wiki: wikipediaapi.Wikipedia, title: str) -> wikipediaapi.WikipediaPage:
    """Pobiera pojedynczy artykuł i weryfikuje, że istnieje."""
    page = wiki.page(title)
    if not page.exists():
        raise ValueError(f"Artykuł '{title}' nie istnieje na Wikipedii (sprawdź tytuł).")
    return page


def save_article(page: wikipediaapi.WikipediaPage, category: str) -> DocumentMetadata:
    """Zapisuje pełny tekst artykułu (page.text) do pliku .txt i zwraca metadane."""
    filename = f"{slugify(page.title)}.txt"
    filepath = RAW_DIR / filename

    # page.text zwraca czysty tekst (wszystkie sekcje spłaszczone),
    # bez wiki-markupu i bez infoboxów/tabel - dokładnie to, czego chcemy do chunkowania
    filepath.write_text(page.text, encoding="utf-8")

    return DocumentMetadata(
        title=page.title,
        filename=filename,
        url=page.fullurl,
        category=category,
        num_characters=len(page.text),
    )


def load_all_documents() -> list[DocumentMetadata]:
    """Pobiera wszystkie artykuły (core + distractor) i zapisuje je na dysk
    wraz z plikiem metadanych."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # user_agent jest wymagany przez politykę Wikipedia API - identyfikuje klienta
    wiki = wikipediaapi.Wikipedia(
        user_agent="rag-evaluation-portfolio-project (contact: example@example.com)",
        language="en",
    )

    all_metadata: list[DocumentMetadata] = []

    for title in CORE_ARTICLES:
        logger.info(f"Pobieram (core): {title}")
        page = fetch_article(wiki, title)
        metadata = save_article(page, category="core")
        all_metadata.append(metadata)

    for title in DISTRACTOR_ARTICLES:
        logger.info(f"Pobieram (distractor): {title}")
        page = fetch_article(wiki, title)
        metadata = save_article(page, category="distractor")
        all_metadata.append(metadata)

    METADATA_PATH.write_text(
        json.dumps([asdict(m) for m in all_metadata], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Zapisano {len(all_metadata)} artykułów. Metadane: {METADATA_PATH}")

    return all_metadata


if __name__ == "__main__":
    load_all_documents()