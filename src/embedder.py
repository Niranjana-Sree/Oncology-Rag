"""
src/embedder.py — MRL embedding for MedRAG.

Uses nomic-ai/nomic-embed-text-v1.5 which implements Matryoshka
Representation Learning (MRL): a single model produces coherent
embeddings at multiple dimensions by truncating the full vector.

After truncation the vector MUST be L2-renormalised so cosine
similarity remains valid (truncated vectors are no longer unit-length).

Public API
----------
embed_text(texts, dim)           → np.ndarray  shape (N, dim)
embed_knowledge_base(chunks, dim) → list[list[float]]  (batch, with logging)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

import config

logger = logging.getLogger("medrag.embedder")

# ---------------------------------------------------------------------------
# Module-level model cache (loaded once, reused across calls)
# ---------------------------------------------------------------------------
_model_cache: dict[str, object] = {}


def _get_model(model_name: str):
    """
    Load the SentenceTransformer model on CPU and cache it.
    CPU is used explicitly to avoid MPS OOM and slowness on Apple Silicon
    for batch inference of large corpora.
    """
    if model_name not in _model_cache:
        logger.info("Loading embedding model: %s (device=cpu)", model_name)
        t0 = time.time()
        from sentence_transformers import SentenceTransformer
        import torch

        # Force CPU — avoids MPS OOM and is faster for batch_size>1 on Mac
        device = "cpu"
        logger.info("Using device: %s", device)

        loaded = False
        for candidate in [model_name, "sentence-transformers/all-MiniLM-L6-v2"]:
            try:
                model = SentenceTransformer(
                    candidate,
                    trust_remote_code=True,
                    device=device,
                )
                _model_cache[model_name] = model
                if candidate != model_name:
                    logger.warning(
                        "Primary model '%s' unavailable — using fallback '%s'.",
                        model_name, candidate,
                    )
                else:
                    logger.info("Model loaded in %.1fs", time.time() - t0)
                loaded = True
                break
            except Exception as exc:
                logger.warning("Could not load '%s': %s — trying fallback", candidate, exc)

        if not loaded:
            raise RuntimeError(
                f"Could not load any embedding model. "
                f"Tried: {model_name}, sentence-transformers/all-MiniLM-L6-v2"
            )
    return _model_cache[model_name]


# ---------------------------------------------------------------------------
# Core embedding function
# ---------------------------------------------------------------------------

def embed_text(
    texts: list[str],
    dim: int = 768,
    model_name: Optional[str] = None,
    batch_size: int = 128,
    prefix: str = "search_document: ",
) -> np.ndarray:
    """
    Embed a list of texts and return MRL vectors at the requested dimension.

    MRL truncation + renormalisation:
        1. Encode at full 768-dim with L2 normalisation.
        2. Slice the first `dim` dimensions.
        3. Renormalise the truncated vector to unit length.
        This preserves cosine similarity validity at any sub-dimension.

    Args:
        texts:      List of strings to embed. Must be non-empty.
        dim:        Output dimension. 128 for coarse Stage-1 scan,
                    768 for fine Stage-2 rerank. Any value 1–768 is valid.
        model_name: HuggingFace model ID. Defaults to config.EMBEDDING_MODEL.
        batch_size: Sentences per encoding batch (tune for VRAM/RAM).
        prefix:     nomic-embed prefix token. Use "search_document: " when
                    embedding corpus chunks and "search_query: " for queries.

    Returns:
        np.ndarray of shape (len(texts), dim), float32, L2-normalised.

    Raises:
        ValueError: if texts is empty or dim is out of range.
        RuntimeError: if the model cannot be loaded.
    """
    if not texts:
        raise ValueError("embed_text: texts list must not be empty")
    if not (1 <= dim <= 768):
        raise ValueError(f"embed_text: dim must be 1–768, got {dim}")

    if model_name is None:
        model_name = config.EMBEDDING_MODEL

    # Add nomic-embed prefix to every text
    prefixed = [f"{prefix}{t}" for t in texts]

    model = _get_model(model_name)

    logger.debug(
        "Embedding %d texts at dim=%d (batch_size=%d)", len(texts), dim, batch_size
    )
    t0 = time.time()

    # Encode at full 768-dim, already L2-normalised by sentence-transformers
    full_embeddings: np.ndarray = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )  # shape: (N, 768)

    # MRL truncation: take first `dim` dimensions
    actual_dim = full_embeddings.shape[1]
    if dim < actual_dim:
        truncated = full_embeddings[:, :dim].copy()
    elif dim > actual_dim:
        # Fallback model has fewer dims than requested — use what we have
        logger.warning(
            "Requested dim=%d but model only produces %d dims — using %d",
            dim, actual_dim, actual_dim,
        )
        truncated = full_embeddings
    else:
        truncated = full_embeddings

    # Renormalise truncated vectors to unit length
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid division by zero
    renormed = truncated / norms                # shape: (N, dim), unit vectors

    elapsed = time.time() - t0
    logger.debug(
        "Embedded %d texts → shape %s in %.2fs",
        len(texts), renormed.shape, elapsed,
    )

    return renormed.astype(np.float32)


def embed_query(
    query: str,
    dim: int = 768,
    model_name: Optional[str] = None,
) -> np.ndarray:
    """
    Embed a single query string with the correct nomic-embed query prefix.

    Args:
        query:      The user query string.
        dim:        Output dimension (128 or 768).
        model_name: Optional model override.

    Returns:
        np.ndarray of shape (1, dim), float32, L2-normalised.
    """
    return embed_text(
        texts=[query],
        dim=dim,
        model_name=model_name,
        batch_size=1,
        prefix="search_query: ",
    )


# ---------------------------------------------------------------------------
# Knowledge-base batch embedder (used by build_index.py)
# ---------------------------------------------------------------------------

def embed_knowledge_base(
    chunks: list,
    dim: int = 768,
    model_name: Optional[str] = None,
    batch_size: int = 128,
) -> list[list[float]]:
    """
    Embed a list of Chunk objects in batches with progress logging.

    Extracts chunk.text from each Chunk, embeds with the document prefix,
    and returns the vectors as plain Python lists (JSON-serialisable).

    Args:
        chunks:     List of Chunk dataclass instances.
        dim:        Embedding dimension (128 or 768).
        model_name: Optional model override.
        batch_size: Texts per encoding batch.

    Returns:
        List of length len(chunks), each element a list[float] of length dim.
    """
    if not chunks:
        return []

    texts = [c.text for c in chunks]
    total = len(texts)

    logger.info(
        "embed_knowledge_base: %d chunks, dim=%d, batch_size=%d",
        total, dim, batch_size,
    )

    t0 = time.time()
    all_embeddings: list[list[float]] = []

    # Process in batches so we can log progress
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]

        batch_vecs = embed_text(
            texts=batch_texts,
            dim=dim,
            model_name=model_name,
            batch_size=batch_size,
            prefix="search_document: ",
        )  # shape: (batch, dim)

        all_embeddings.extend(batch_vecs.tolist())

        if (start // batch_size) % 10 == 0 or end == total:
            elapsed = time.time() - t0
            pct = end / total * 100
            rate = end / elapsed if elapsed > 0 else 0
            logger.info(
                "  [%5d/%5d] %.1f%%  %.0f chunks/s  elapsed=%.1fs",
                end, total, pct, rate, elapsed,
            )

    elapsed = time.time() - t0
    logger.info(
        "embed_knowledge_base: done — %d vectors, dim=%d, %.1fs total",
        len(all_embeddings), dim, elapsed,
    )
    return all_embeddings


# ---------------------------------------------------------------------------
# Sanity-check helpers
# ---------------------------------------------------------------------------

def verify_unit_norm(vectors: np.ndarray, tol: float = 1e-5) -> bool:
    """Return True if all rows of `vectors` are approximately unit-length."""
    norms = np.linalg.norm(vectors, axis=1)
    return bool(np.all(np.abs(norms - 1.0) < tol))


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between each row of a and each row of b.
    Both must be L2-normalised (unit vectors).
    Returns shape (len(a), len(b)).
    """
    return (a @ b.T).astype(np.float32)
