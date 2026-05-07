"""
src/retriever.py — 3-stage retrieval pipeline for MedRAG.

Stage 1: coarse_search()   — 64-dim MRL vectors, ChromaDB, top-100 candidates
Stage 2: fine_rerank()     — 384-dim MRL vectors, re-score top-100 → top-20
Stage 3: cross_encoder     — handled by src/reranker.py (Feature 4.3)

Public API
----------
stage1_coarse_search(query, analysed_query, n_results, where) → list[RetrievedChunk]
stage2_fine_rerank(query, stage1_chunks, n_results)           → list[RetrievedChunk]
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import config
from src.models import RetrievedChunk, AnalysedQuery

logger = logging.getLogger("medrag.retriever")


# ---------------------------------------------------------------------------
# Stage 1 — Coarse search (64-dim)
# ---------------------------------------------------------------------------

def stage1_coarse_search(
    translated_query: str,
    analysed_query: Optional[AnalysedQuery] = None,
    n_results: int = None,
    where: Optional[dict] = None,
) -> list[RetrievedChunk]:
    """
    Stage 1: Fast coarse retrieval using 64-dim MRL embeddings.

    Embeds the query (and all expanded queries if available) at 64-dim,
    queries the medrag_64 ChromaDB collection, and returns the union of
    results deduplicated and sorted by score.

    Multi-query fusion:
        If analysed_query.expanded_queries contains additional search strings
        (synonyms, HyDE text, rephrased versions), each is embedded and
        searched independently. Results are fused by taking the maximum
        score across all queries for each chunk_id (score fusion).

    Args:
        translated_query:  English query string (from full_query_expansion).
        analysed_query:    Optional AnalysedQuery with expanded_queries for
                           multi-query fusion. If None, single-query search.
        n_results:         Number of candidates to retrieve (default: STAGE1_TOP_K=100).
        where:             Optional ChromaDB metadata filter.
                           e.g. {"category": "clinical_guideline"}

    Returns:
        List of RetrievedChunk sorted by score descending, up to n_results.
    """
    from src.embedder import embed_query, embed_text
    from src.vectordb import query_collection

    if n_results is None:
        n_results = config.STAGE1_TOP_K

    t0 = time.time()

    # Build list of all search strings: original + expanded
    search_strings: list[str] = [translated_query]

    if analysed_query and analysed_query.expanded_queries:
        # Add synonym terms and multi-query versions (skip very short strings)
        for s in analysed_query.expanded_queries:
            s = s.strip()
            if len(s) > 5 and s not in search_strings:
                search_strings.append(s)

    logger.info(
        "stage1_coarse_search: %d search strings, n_results=%d",
        len(search_strings), n_results,
    )

    # Score fusion map: chunk_id → max score across all queries
    score_map:    dict[str, float]         = {}
    chunk_map:    dict[str, RetrievedChunk] = {}

    for i, search_str in enumerate(search_strings):
        # Embed at 64-dim (coarse)
        qvec = embed_query(search_str, dim=config.DIM_COARSE)[0].tolist()

        # Query ChromaDB coarse collection
        results = query_collection(
            query_vector=qvec,
            dim=config.DIM_COARSE,
            n_results=n_results,
            where=where,
        )

        for chunk in results:
            cid = chunk.chunk_id
            if cid not in score_map or chunk.score > score_map[cid]:
                score_map[cid]  = chunk.score
                chunk_map[cid]  = chunk

        logger.debug(
            "  [%d/%d] %r → %d hits",
            i + 1, len(search_strings), search_str[:50], len(results),
        )

    # Build final ranked list using max-score fusion
    fused: list[RetrievedChunk] = []
    for cid, chunk in chunk_map.items():
        chunk.score = score_map[cid]
        chunk.stage = "stage1"
        fused.append(chunk)

    fused.sort(key=lambda c: c.score, reverse=True)
    final = fused[:n_results]

    elapsed = time.time() - t0
    logger.info(
        "stage1_coarse_search: %d candidates in %.3fs "
        "(searched %d strings, union=%d unique)",
        len(final), elapsed, len(search_strings), len(fused),
    )
    return final


# ---------------------------------------------------------------------------
# Stage 2 — Fine rerank (384-dim)
# ---------------------------------------------------------------------------

def stage2_fine_rerank(
    translated_query: str,
    stage1_chunks: list[RetrievedChunk],
    n_results: int = None,
) -> list[RetrievedChunk]:
    """
    Stage 2: Fine reranking of Stage 1 candidates using 384-dim MRL embeddings.

    Takes the top-N chunks from Stage 1, re-scores each using the 384-dim
    collection (higher precision), and returns the top n_results.

    Why a second stage instead of just querying 384-dim directly?
        Querying 384-dim over all 33,592 chunks is slower than querying
        64-dim. Stage 1 rapidly narrows the search space to ~100 candidates;
        Stage 2 re-scores only those 100 with higher-quality vectors.

    Implementation:
        Rather than re-querying ChromaDB for each candidate (100 round-trips),
        we embed the query at 384-dim and compute dot-product similarity
        directly against the 384-dim vectors fetched from the fine collection
        using their chunk_ids.

    Args:
        translated_query: English query string.
        stage1_chunks:    Output of stage1_coarse_search().
        n_results:        Number of chunks to return (default: STAGE2_TOP_K=20).

    Returns:
        List of RetrievedChunk re-scored at 384-dim, sorted by score,
        up to n_results.
    """
    from src.embedder import embed_query
    from src.vectordb import get_collection

    if n_results is None:
        n_results = config.STAGE2_TOP_K

    if not stage1_chunks:
        logger.warning("stage2_fine_rerank: empty stage1_chunks — returning []")
        return []

    t0 = time.time()

    # Embed query at 384-dim
    qvec = embed_query(translated_query, dim=config.DIM_FINE)[0]  # shape (384,)

    # Fetch 384-dim vectors for stage1 chunk_ids from ChromaDB
    collection = get_collection(config.DIM_FINE)
    candidate_ids = [c.chunk_id for c in stage1_chunks]

    try:
        fetched = collection.get(
            ids=candidate_ids,
            include=["embeddings", "documents", "metadatas"],
        )
    except Exception as exc:
        logger.error("stage2_fine_rerank: ChromaDB fetch failed (%s)", exc)
        # Fall back: return stage1 results unchanged, tagged as stage2
        for c in stage1_chunks[:n_results]:
            c.stage = "stage2"
        return stage1_chunks[:n_results]

    if not fetched["ids"]:
        logger.warning("stage2_fine_rerank: no vectors fetched from fine collection")
        return stage1_chunks[:n_results]

    import numpy as np

    qvec_np = qvec.astype(np.float32)   # shape (384,)

    # Score each candidate: dot product with 384-dim query (both unit vectors)
    reranked: list[RetrievedChunk] = []
    for cid, emb, doc, meta in zip(
        fetched["ids"],
        fetched["embeddings"],
        fetched["documents"],
        fetched["metadatas"],
    ):
        emb_np = np.array(emb, dtype=np.float32)
        score  = float(np.dot(qvec_np, emb_np))   # cosine sim for unit vectors

        reranked.append(
            RetrievedChunk(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                text=doc,
                category=meta.get("category", ""),
                source_file=meta.get("source_file", ""),
                score=score,
                stage="stage2",
                metadata=meta,
            )
        )

    reranked.sort(key=lambda c: c.score, reverse=True)
    final = reranked[:n_results]

    elapsed = time.time() - t0
    logger.info(
        "stage2_fine_rerank: %d → %d chunks in %.3fs  "
        "(top score: %.3f, bottom: %.3f)",
        len(stage1_chunks), len(final), elapsed,
        final[0].score if final else 0,
        final[-1].score if final else 0,
    )
    return final
