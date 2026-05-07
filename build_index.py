"""
build_index.py — Phase 2 entry point.

Loads all chunks from chunks/all_chunks.json, embeds them at two MRL
dimensions (coarse + fine), stores both sets in ChromaDB, then runs
a sanity check to confirm the index is queryable.

Usage:
    python build_index.py [--chunks-file chunks/all_chunks.json]
                          [--batch-size 128]
                          [--wipe]
                          [--log-level INFO]

Flags:
    --wipe        Delete and recreate both collections before indexing.
                  Use when re-indexing from scratch after re-chunking.
    --batch-size  Chunks per embedding batch (default 128).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.embedder import embed_knowledge_base
from src.vectordb import store_chunks, sanity_check, delete_collections, get_collection

logger = logging.getLogger("medrag.build_index")


# ---------------------------------------------------------------------------
# Chunk proxy — avoids loading the full dataclass machinery
# ---------------------------------------------------------------------------

class _ChunkProxy:
    """Lightweight stand-in for a Chunk dataclass, read from JSON."""
    __slots__ = (
        "chunk_id", "doc_id", "text", "category",
        "source_file", "chunk_index", "strategy", "metadata",
    )

    def __init__(self, d: dict) -> None:
        self.chunk_id    = d["chunk_id"]
        self.doc_id      = d["doc_id"]
        self.text        = d["text"]
        self.category    = d["category"]
        self.source_file = d["source_file"]
        self.chunk_index = d["chunk_index"]
        self.strategy    = d["strategy"]
        self.metadata    = d.get("metadata", {})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    chunks_file = Path(args.chunks_file).resolve()
    if not chunks_file.exists():
        logger.error("Chunks file not found: %s", chunks_file)
        logger.error("Run  python run_chunking.py  first.")
        sys.exit(1)

    logger.info("=" * 65)
    logger.info("MedRAG Build Index")
    logger.info("  chunks_file : %s", chunks_file)
    logger.info("  dim_coarse  : %d  (collection: %s)", config.DIM_COARSE, config.COLLECTION_COARSE)
    logger.info("  dim_fine    : %d  (collection: %s)", config.DIM_FINE,   config.COLLECTION_FINE)
    logger.info("  batch_size  : %d", args.batch_size)
    logger.info("  wipe        : %s", args.wipe)
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # Step 1: Optionally wipe existing collections
    # ------------------------------------------------------------------
    if args.wipe:
        logger.info("--wipe flag set — deleting existing collections...")
        delete_collections()
        logger.info("Collections wiped and recreated (empty).")

    # ------------------------------------------------------------------
    # Step 2: Check if already indexed (skip if counts match)
    # ------------------------------------------------------------------
    col_coarse = get_collection(config.DIM_COARSE)
    col_fine   = get_collection(config.DIM_FINE)
    existing_coarse = col_coarse.count()
    existing_fine   = col_fine.count()

    # ------------------------------------------------------------------
    # Step 3: Load chunks from JSON
    # ------------------------------------------------------------------
    logger.info("Loading chunks from %s ...", chunks_file.name)
    t0 = time.time()
    with open(chunks_file, encoding="utf-8") as fh:
        data = json.load(fh)

    all_chunk_dicts = data["chunks"]
    total = len(all_chunk_dicts)
    logger.info("Loaded %d chunks in %.1fs", total, time.time() - t0)

    # Print per-category breakdown
    from collections import Counter
    cat_counts = Counter(d["category"] for d in all_chunk_dicts)
    for cat, count in sorted(cat_counts.items()):
        logger.info("  %-25s : %d chunks", cat, count)

    if existing_coarse == total and existing_fine == total and not args.wipe:
        logger.info(
            "Both collections already have %d chunks — index is up to date. "
            "Use --wipe to force rebuild.",
            total,
        )
        _print_sanity(total)
        return

    chunks = [_ChunkProxy(d) for d in all_chunk_dicts]

    # ------------------------------------------------------------------
    # Step 4: Embed + store at DIM_COARSE
    # ------------------------------------------------------------------
    logger.info("-" * 65)
    logger.info("STAGE A — Embedding at dim=%d (coarse / Stage-1)", config.DIM_COARSE)
    logger.info("-" * 65)
    t0 = time.time()
    vecs_coarse = embed_knowledge_base(
        chunks,
        dim=config.DIM_COARSE,
        batch_size=args.batch_size,
    )
    embed_time_coarse = time.time() - t0
    logger.info(
        "Embedding done: %d vectors, dim=%d, %.1fs (%.0f chunks/s)",
        len(vecs_coarse), config.DIM_COARSE,
        embed_time_coarse, total / embed_time_coarse,
    )

    t0 = time.time()
    n_coarse = store_chunks(chunks, vecs_coarse, dim=config.DIM_COARSE, batch_size=512)
    logger.info(
        "Stored %d chunks in '%s' in %.1fs",
        n_coarse, config.COLLECTION_COARSE, time.time() - t0,
    )

    # ------------------------------------------------------------------
    # Step 5: Embed + store at DIM_FINE
    # ------------------------------------------------------------------
    logger.info("-" * 65)
    logger.info("STAGE B — Embedding at dim=%d (fine / Stage-2)", config.DIM_FINE)
    logger.info("-" * 65)
    t0 = time.time()
    vecs_fine = embed_knowledge_base(
        chunks,
        dim=config.DIM_FINE,
        batch_size=args.batch_size,
    )
    embed_time_fine = time.time() - t0
    logger.info(
        "Embedding done: %d vectors, dim=%d, %.1fs (%.0f chunks/s)",
        len(vecs_fine), config.DIM_FINE,
        embed_time_fine, total / embed_time_fine,
    )

    t0 = time.time()
    n_fine = store_chunks(chunks, vecs_fine, dim=config.DIM_FINE, batch_size=512)
    logger.info(
        "Stored %d chunks in '%s' in %.1fs",
        n_fine, config.COLLECTION_FINE, time.time() - t0,
    )

    # ------------------------------------------------------------------
    # Step 6: Sanity check
    # ------------------------------------------------------------------
    _print_sanity(total)

    # ------------------------------------------------------------------
    # Step 7: Summary
    # ------------------------------------------------------------------
    total_time = embed_time_coarse + embed_time_fine
    logger.info("=" * 65)
    logger.info("BUILD INDEX COMPLETE")
    logger.info("  Chunks indexed     : %d", total)
    logger.info("  Coarse collection  : %s (%d)", config.COLLECTION_COARSE, n_coarse)
    logger.info("  Fine collection    : %s (%d)", config.COLLECTION_FINE,   n_fine)
    logger.info("  Total embed time   : %.1fs", total_time)
    logger.info("  Avg throughput     : %.0f chunks/s", total / total_time)
    logger.info("=" * 65)

    if n_coarse != total or n_fine != total:
        logger.error("MISMATCH: expected %d in both collections", total)
        sys.exit(1)


def _print_sanity(expected: int) -> None:
    logger.info("-" * 65)
    logger.info("Running sanity check...")
    report = sanity_check()
    all_ok = True
    for label, info in report.items():
        status = info.get("status", "UNKNOWN")
        count  = info.get("count", 0)
        ok     = status == "OK" and count == expected
        flag   = "✓" if ok else "✗"
        logger.info(
            "  %s [%s] collection=%-12s  count=%-6d  query_hits=%s  status=%s",
            flag, label, info["collection"], count,
            info.get("query_test_hits", "-"), status,
        )
        if not ok:
            all_ok = False
    if all_ok:
        logger.info("Sanity check PASSED — both collections ready for retrieval.")
    else:
        logger.warning("Sanity check FAILED — check logs above.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MedRAG vector index builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--chunks-file",
        default=str(config.CHUNKS_DIR / "all_chunks.json"),
        help="Path to all_chunks.json produced by run_chunking.py",
    )
    p.add_argument(
        "--batch-size", type=int, default=128,
        help="Chunks per embedding batch",
    )
    p.add_argument(
        "--wipe", action="store_true",
        help="Delete and recreate both collections before indexing",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return p


if __name__ == "__main__":
    main(_build_parser().parse_args())
