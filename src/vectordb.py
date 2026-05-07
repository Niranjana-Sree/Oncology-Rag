"""
src/vectordb.py — ChromaDB vector store for MedRAG.

Two persistent collections:
  medrag_64  — 64-dim  MRL coarse vectors (Stage 1 fast scan)
  medrag_384 — 384-dim MRL fine vectors   (Stage 2 precision rerank)

Public API
----------
get_client()                        → chromadb.PersistentClient
get_collection(dim)                 → chromadb.Collection
store_chunks(chunks, embeddings, dim) → int  (number stored)
query_collection(query_vec, dim, n_results, where) → list[RetrievedChunk]
sanity_check()                      → dict  (collection counts + sample)
delete_collections()                → None  (wipe and recreate)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import config
from src.models import RetrievedChunk

logger = logging.getLogger("medrag.vectordb")

# ---------------------------------------------------------------------------
# Client — singleton, created once
# ---------------------------------------------------------------------------
_client_cache: dict = {}


def get_client():
    """Return a persistent ChromaDB client, creating it once."""
    if "client" not in _client_cache:
        import chromadb
        logger.info("Connecting to ChromaDB at: %s", config.CHROMADB_PATH)
        client = chromadb.PersistentClient(path=str(config.CHROMADB_PATH))
        _client_cache["client"] = client
        logger.info("ChromaDB connected.")
    return _client_cache["client"]


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

def _collection_name(dim: int) -> str:
    """Return the collection name for a given embedding dimension."""
    if dim <= 64:
        return config.COLLECTION_COARSE   # medrag_64
    return config.COLLECTION_FINE         # medrag_384


def get_collection(dim: int):
    """
    Get or create the ChromaDB collection for the given dimension.

    Args:
        dim: Embedding dimension — 64 for coarse, 384 for fine.

    Returns:
        chromadb.Collection instance.
    """
    client = get_client()
    name = _collection_name(dim)
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for all queries
    )
    logger.debug("Collection '%s' ready — count=%d", name, collection.count())
    return collection


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def store_chunks(
    chunks: list,
    embeddings: list[list[float]],
    dim: int,
    batch_size: int = 512,
) -> int:
    """
    Upsert chunks and their embeddings into the appropriate ChromaDB collection.

    Uses upsert (not add) so re-running build_index.py is idempotent —
    existing chunks are updated rather than duplicated.

    Args:
        chunks:     List of Chunk dataclass instances.
        embeddings: Parallel list of embedding vectors (one per chunk).
        dim:        Embedding dimension — determines which collection to use.
        batch_size: Number of chunks per ChromaDB upsert call.

    Returns:
        Total number of chunks successfully upserted.

    Raises:
        ValueError: if len(chunks) != len(embeddings).
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"store_chunks: chunks ({len(chunks)}) and embeddings "
            f"({len(embeddings)}) must have the same length"
        )
    if not chunks:
        logger.warning("store_chunks: called with empty chunk list — nothing stored")
        return 0

    collection = get_collection(dim)
    total = len(chunks)
    stored = 0
    t0 = time.time()

    logger.info(
        "store_chunks: upserting %d chunks into '%s' (batch_size=%d)",
        total, collection.name, batch_size,
    )

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_chunks     = chunks[start:end]
        batch_embeddings = embeddings[start:end]

        ids       = [c.chunk_id for c in batch_chunks]
        documents = [c.text for c in batch_chunks]
        metadatas = [
            {
                "doc_id":      c.doc_id,
                "category":    c.category,
                "source_file": c.source_file,
                "chunk_index": c.chunk_index,
                "strategy":    c.strategy,
                "delegated_strategy": c.metadata.get("delegated_strategy", c.strategy),
            }
            for c in batch_chunks
        ]

        collection.upsert(
            ids=ids,
            embeddings=batch_embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        stored += len(batch_chunks)

        if (start // batch_size) % 10 == 0 or end == total:
            elapsed = time.time() - t0
            pct = stored / total * 100
            rate = stored / elapsed if elapsed > 0 else 0
            logger.info(
                "  [%6d/%6d] %.1f%%  %.0f chunks/s  elapsed=%.1fs",
                stored, total, pct, rate, elapsed,
            )

    elapsed = time.time() - t0
    logger.info(
        "store_chunks: done — %d chunks in '%s', %.1fs total",
        stored, collection.name, elapsed,
    )
    return stored


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query_collection(
    query_vector: list[float],
    dim: int,
    n_results: int = 100,
    where: Optional[dict] = None,
) -> list[RetrievedChunk]:
    """
    Query a ChromaDB collection with a single embedding vector.

    Args:
        query_vector: L2-normalised embedding of the user query.
        dim:          Which collection to query (64 or 384).
        n_results:    Maximum number of results to return.
        where:        Optional ChromaDB metadata filter dict.
                      e.g. {"category": {"$in": ["clinical_guideline"]}}

    Returns:
        List of RetrievedChunk sorted by relevance (most relevant first).
    """
    collection = get_collection(dim)
    count = collection.count()

    if count == 0:
        logger.warning("query_collection: collection '%s' is empty", collection.name)
        return []

    # Cap n_results to actual collection size
    n_results = min(n_results, count)

    kwargs: dict = {
        "query_embeddings": [query_vector],
        "n_results":        n_results,
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    t0 = time.time()
    results = collection.query(**kwargs)
    elapsed = time.time() - t0

    logger.debug(
        "query_collection: '%s' returned %d results in %.3fs",
        collection.name, len(results["ids"][0]), elapsed,
    )

    retrieved: list[RetrievedChunk] = []
    ids       = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    stage = "stage1" if dim <= 64 else "stage2"

    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        # ChromaDB cosine distance = 1 - cosine_similarity
        # Convert to similarity score in [0, 1]
        score = max(0.0, 1.0 - float(distance))
        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", ""),
                text=text,
                category=meta.get("category", ""),
                source_file=meta.get("source_file", ""),
                score=score,
                stage=stage,
                metadata=meta,
            )
        )

    return retrieved


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check() -> dict:
    """
    Verify both ChromaDB collections are populated and queryable.

    Returns a dict with counts, a sample chunk from each collection,
    and a basic query test result.
    """
    import numpy as np

    logger.info("Running ChromaDB sanity check...")
    report: dict = {}

    for dim, label in [(config.DIM_COARSE, "coarse"), (config.DIM_FINE, "fine")]:
        col = get_collection(dim)
        count = col.count()
        report[label] = {"collection": col.name, "count": count}

        if count == 0:
            logger.warning("sanity_check: collection '%s' is EMPTY", col.name)
            report[label]["status"] = "EMPTY"
            continue

        # Peek at first item
        peek = col.peek(limit=1)
        sample_id   = peek["ids"][0]
        sample_text = peek["documents"][0][:80]
        report[label]["sample_id"]   = sample_id
        report[label]["sample_text"] = sample_text

        # Run a random query to confirm querying works
        rng = np.random.default_rng(42)
        random_vec = rng.standard_normal(dim).astype(np.float32)
        random_vec /= np.linalg.norm(random_vec)
        results = query_collection(random_vec.tolist(), dim=dim, n_results=3)
        report[label]["query_test_hits"] = len(results)
        report[label]["status"] = "OK" if results else "QUERY_FAILED"

        logger.info(
            "  [%s] collection='%s'  count=%d  query_hits=%d  status=%s",
            label, col.name, count, len(results), report[label]["status"],
        )

    return report


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def delete_collections() -> None:
    """
    Delete and recreate both collections (full wipe).
    Use before re-running build_index.py from scratch.
    """
    client = get_client()
    for dim in [config.DIM_COARSE, config.DIM_FINE]:
        name = _collection_name(dim)
        try:
            client.delete_collection(name)
            logger.info("Deleted collection '%s'", name)
        except Exception:
            logger.debug("Collection '%s' did not exist — nothing to delete", name)
    # Recreate empty collections
    get_collection(config.DIM_COARSE)
    get_collection(config.DIM_FINE)
    logger.info("Collections recreated (empty).")
