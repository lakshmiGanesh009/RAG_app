"""Save and search embedded chunks on disk.

Persists between runs, so documents only need ingesting once.

Usage:

    from rag_app.store import VectorStore

    store = VectorStore()
    store.add(chunks)                       # from rag_app.chunking

    for hit in store.search("how much did we spend?"):
        print(f"{hit.score:.2f}  {hit.source} p{hit.page}  {hit.heading}")

    print(store.stats())
    store.reset()                           # start again from empty
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb

from rag_app.chunking import Chunk
from rag_app.config import Settings, get_settings
from rag_app.embedding import embed_passages, embed_query, embedding_dimensions

log = logging.getLogger(__name__)


class StoreError(Exception):
    """Raised when the store cannot be read or written."""


@dataclass(frozen=True)
class SearchResult:
    """One chunk that matched a question, and how well it matched."""

    text: str
    source: str
    page: int | None
    heading: str | None
    score: float
    """Similarity from 0 to 1, where 1 is a perfect match."""


@dataclass(frozen=True)
class Stats:
    """A summary of what is currently stored."""

    total_chunks: int
    chunks_by_source: dict[str, int]
    collection: str
    location: Path

    def __str__(self) -> str:
        if not self.total_chunks:
            return f"Store '{self.collection}' is empty ({self.location})."
        lines = [
            f"Store '{self.collection}' holds {self.total_chunks} chunk(s) "
            f"from {len(self.chunks_by_source)} document(s):"
        ]
        for source, count in sorted(self.chunks_by_source.items()):
            lines.append(f"  {count:5}  {source}")
        lines.append(f"Location: {self.location}")
        return "\n".join(lines)


class VectorStore:
    """Chunks and their vectors, stored on disk so they survive a restart."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._settings.store_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._settings.store_dir))
        self._collection = self._open()

    # ---------------------------------------------------------------- write

    def add(self, chunks: list[Chunk]) -> int:
        """Store chunks, replacing anything already held for the same files.

        Every document mentioned in `chunks` is cleared first, so editing a file
        and ingesting it again leaves no stale text behind. Ingesting an
        unchanged file is harmless and changes nothing.

        Returns:
            How many chunks were stored.
        """
        if not chunks:
            return 0

        for source in sorted({chunk.source for chunk in chunks}):
            self.remove_source(source)

        vectors = embed_passages([chunk.text for chunk in chunks], self._settings)
        self._collection.upsert(
            ids=[_chunk_id(chunk) for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[_metadata(chunk) for chunk in chunks],
        )

        log.info("Stored %d chunk(s)", len(chunks))
        return len(chunks)

    def remove_source(self, source: str) -> None:
        """Forget everything that came from one file."""
        self._collection.delete(where={"source": source})

    # ----------------------------------------------------------------- read

    def search(
        self,
        question: str,
        top_k: int | None = None,
        min_relevance: float | None = None,
    ) -> list[SearchResult]:
        """Find the chunks most similar to a question.

        Args:
            question: The question, in plain language.
            top_k: How many results to return. Defaults to the setting.
            min_relevance: Drop results scoring below this. Defaults to the
                setting.

        Returns:
            Matches, best first.
        """
        wanted = top_k if top_k is not None else self._settings.top_k
        floor = (
            min_relevance if min_relevance is not None else self._settings.min_relevance
        )

        if self._collection.count() == 0:
            log.warning("Nothing has been ingested yet, so there is nothing to find.")
            return []

        vector = embed_query(question, self._settings)
        try:
            raw = self._collection.query(
                query_embeddings=[vector],
                n_results=wanted,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as error:
            raise StoreError(_search_advice(error, self._settings)) from error

        results: list[SearchResult] = []
        for text, meta, distance in zip(
            raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        ):
            # Chroma reports cosine distance, where 0 means identical.
            score = 1.0 - float(distance)
            if score < floor:
                continue
            results.append(
                SearchResult(
                    text=text,
                    source=str(meta.get("source", "unknown")),
                    page=meta.get("page"),
                    heading=meta.get("heading"),
                    score=score,
                )
            )
        return results

    def count(self) -> int:
        """Return how many chunks are stored."""
        return self._collection.count()

    def stats(self) -> Stats:
        """Summarise what is stored, broken down by document."""
        stored = self._collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in stored["metadatas"] or []:
            source = str(meta.get("source", "unknown"))
            counts[source] = counts.get(source, 0) + 1
        return Stats(
            total_chunks=self._collection.count(),
            chunks_by_source=counts,
            collection=self._settings.collection_name,
            location=self._settings.store_dir,
        )

    # ---------------------------------------------------------------- admin

    def reset(self) -> None:
        """Delete everything stored and start from empty.

        The documents on disk are untouched; ingest them again to rebuild.
        """
        self._client.delete_collection(self._settings.collection_name)
        self._collection = self._open()
        log.info("Store cleared")

    # ------------------------------------------------------------ internals

    def _open(self):
        """Open the collection, creating it if this is the first run."""
        return self._client.get_or_create_collection(
            name=self._settings.collection_name,
            # Cosine similarity suits normalised text embeddings.
            configuration={"hnsw": {"space": "cosine"}},
            # We supply our own vectors, so Chroma must not make its own.
            embedding_function=None,
        )


def _chunk_id(chunk: Chunk) -> str:
    """Build an id from the chunk's own content.

    The same chunk always gets the same id, so re-ingesting a file overwrites
    rather than adding a second copy.
    """
    fingerprint = f"{chunk.source}|{chunk.index}|{chunk.text}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]


def _metadata(chunk: Chunk) -> dict[str, str | int]:
    """Pack the details needed to cite this chunk later.

    Empty values are left out rather than stored as nulls, so reading them back
    gives a plain None.
    """
    meta: dict[str, str | int] = {
        "source": chunk.source,
        "index": chunk.index,
        "token_count": chunk.token_count,
    }
    if chunk.page is not None:
        meta["page"] = chunk.page
    if chunk.heading:
        meta["heading"] = chunk.heading
    return meta


def _search_advice(error: Exception, settings: Settings) -> str:
    """Explain a failed search, in particular the common dimension mismatch."""
    message = str(error)
    if "dimension" in message.lower():
        return (
            f"This store was built with a different embedding model, so its "
            f"vectors are the wrong size for '{settings.embedding_model}' "
            f"({embedding_dimensions(settings)} numbers each). Clear the store "
            f"and ingest your documents again. Original error: {message}"
        )
    return f"Search failed: {message}"
