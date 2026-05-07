"""
src/reranker.py — Stage 3 cross-encoder reranking for MedRAG.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to jointly encode (query, chunk)
pairs and produce a single relevance score per pair.

Unlike bi-encoders (Stage 1 and 2), a cross-encoder reads query and chunk
together in the same forward pass — capturing fine-grained semantic
interaction that independent embeddings miss.

Public API
----------
cross_encoder_rerank(query, chunks, top_k) → list[RetrievedChunk]
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import config
from src.models import RetrievedChunk

logger = logging.getLogger("medrag.reranker")

# Module-level model cache
_model_cache: dict = {}


def _get_cross_encoder(model_name: str):
    """Load and cache the cross-encoder model."""
    if model_name not in _model_cache:
        logger.info("Loading cross-encoder: %s", model_name)
        t0 = time.time()
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            model = CrossEncoder(model_name, device="cpu")
            _model_cache[model_name] = model
            logger.info("Cross-encoder loaded in %.1fs", time.time() - t0)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load cross-encoder '{model_name}': {exc}"
            ) from exc
    return _model_cache[model_name]


def cross_encoder_rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = None,
    model_name: Optional[str] = None,
) -> list[RetrievedChunk]:
    """
    Stage 3: Cross-encoder reranking of Stage 2 candidates.

    Scores each (query, chunk_text) pair jointly using a cross-encoder.
    The cross-encoder reads both texts together in a single BERT forward
    pass, producing a scalar relevance logit that captures fine-grained
    semantic interaction impossible with independent bi-encoder embeddings.

    Why cross-encoder for Stage 3:
        - Stage 1/2 embed query and chunk independently → miss interaction
        - Cross-encoder: "Does THIS chunk specifically answer THIS query?"
        - Much higher precision, but slower (one forward pass per pair)
        - Feasible because we only score top-20 chunks from Stage 2

    Scoring:
        Raw logits from ms-marco-MiniLM are converted to sigmoid scores
        in [0, 1] for interpretability. Higher = more relevant.

    Args:
        query:      English query string.
        chunks:     List of RetrievedChunk from stage2_fine_rerank().
        top_k:      Number of chunks to return (default: STAGE3_TOP_K=5).
        model_name: Cross-encoder model ID (default: config.RERANKER_MODEL).

    Returns:
        List of RetrievedChunk re-scored by cross-encoder, sorted by score,
        up to top_k. Each chunk's score is updated to the cross-encoder score.
        stage field set to "stage3".
    """
    import math

    if top_k is None:
        top_k = config.STAGE3_TOP_K
    if model_name is None:
        model_name = config.RERANKER_MODEL

    if not chunks:
        logger.warning("cross_encoder_rerank: empty chunk list — returning []")
        return []

    t0 = time.time()

    try:
        model = _get_cross_encoder(model_name)

        # Build (query, passage) pairs for cross-encoder
        pairs = [[query, c.text] for c in chunks]

        # Score all pairs — returns raw logits (unbounded floats)
        logits = model.predict(pairs, show_progress_bar=False)

        # Convert logits to sigmoid scores in [0, 1]
        scores = [1.0 / (1.0 + math.exp(-float(l))) for l in logits]

        # Attach scores and tag stage
        import copy
        reranked: list[RetrievedChunk] = []
        for chunk, score in zip(chunks, scores):
            c = copy.copy(chunk)
            c.score = score
            c.stage = "stage3"
            reranked.append(c)

        # Sort by cross-encoder score descending
        reranked.sort(key=lambda c: c.score, reverse=True)
        final = reranked[:top_k]

        elapsed = time.time() - t0
        logger.info(
            "cross_encoder_rerank: %d → %d chunks in %.3fs  "
            "(top: %.4f, bottom: %.4f)",
            len(chunks), len(final), elapsed,
            final[0].score if final else 0,
            final[-1].score if final else 0,
        )
        return final

    except Exception as exc:
        logger.error(
            "cross_encoder_rerank: failed (%s) — falling back to Stage2 top-%d",
            exc, top_k,
        )
        # Fallback: return Stage2 results re-tagged as stage3
        fallback = []
        import copy
        for c in chunks[:top_k]:
            cf = copy.copy(c)
            cf.stage = "stage3"
            fallback.append(cf)
        return fallback
