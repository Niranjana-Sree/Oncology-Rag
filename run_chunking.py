"""
run_chunking.py — Phase 1 entry point.

Loads all documents from data/, chunks each one using agentic_chunk()
(which auto-selects the best strategy per document), and saves all chunks
to chunks/all_chunks.json.

Usage:
    python run_chunking.py [--data-dir data/] [--output-dir chunks/]
                           [--max-tokens 512] [--overlap 64]
                           [--strategy auto|sentence|recursive|semantic]
                           [--log-level INFO]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Make project root importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402 — sets up logging
from src.document_loader import load_directory
from src.chunker import agentic_chunk, sentence_split, recursive_split, semantic_chunk, detect_doc_type
from src.models import DocCategory, ChunkStrategy

logger = logging.getLogger("medrag.run_chunking")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_doc(doc, strategy: str, max_tokens: int, overlap: int):
    """Dispatch to the right chunking function based on strategy arg."""
    if strategy == "auto":
        return agentic_chunk(doc, max_tokens=max_tokens, overlap_tokens=overlap)
    elif strategy == ChunkStrategy.SENTENCE.value:
        return sentence_split(doc, max_tokens=max_tokens, overlap_tokens=overlap)
    elif strategy == ChunkStrategy.RECURSIVE.value:
        return recursive_split(doc, max_tokens=max_tokens, overlap_tokens=overlap)
    elif strategy == ChunkStrategy.SEMANTIC.value:
        return semantic_chunk(doc, max_tokens=max_tokens, overlap_tokens=overlap)
    else:
        logger.warning("Unknown strategy '%s' — falling back to agentic", strategy)
        return agentic_chunk(doc, max_tokens=max_tokens, overlap_tokens=overlap)


def _chunk_to_dict(chunk) -> dict:
    """Serialise a Chunk dataclass to a JSON-safe dict."""
    return {
        "chunk_id":      chunk.chunk_id,
        "doc_id":        chunk.doc_id,
        "text":          chunk.text,
        "category":      chunk.category,
        "source_file":   chunk.source_file,
        "chunk_index":   chunk.chunk_index,
        "strategy":      chunk.strategy,
        "metadata":      chunk.metadata,
        # embeddings are empty at this stage — populated by build_index.py
        "embedding_128": chunk.embedding_128,
        "embedding_768": chunk.embedding_768,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    data_dir   = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "all_chunks.json"

    logger.info("=" * 60)
    logger.info("MedRAG Chunking Pipeline")
    logger.info("  data_dir   : %s", data_dir)
    logger.info("  output_dir : %s", output_dir)
    logger.info("  strategy   : %s", args.strategy)
    logger.info("  max_tokens : %d", args.max_tokens)
    logger.info("  overlap    : %d", args.overlap)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load documents from oncology corpus subdirectories
    # (drug_monographs removed — oncology-only corpus)
    # ------------------------------------------------------------------
    category_map = {
        "clinical_guidelines": DocCategory.CLINICAL_GUIDELINE.value,
        "research_papers":     DocCategory.RESEARCH_PAPER.value,
        "general":             DocCategory.GENERAL.value,
    }

    all_docs = []
    for subdir, category in category_map.items():
        dir_path = data_dir / subdir
        docs = load_directory(dir_path, category=category)
        all_docs.extend(docs)
        logger.info("Loaded %d docs from %s/", len(docs), subdir)

    if not all_docs:
        logger.error("No documents found in %s — aborting.", data_dir)
        sys.exit(1)

    logger.info("Total documents to chunk: %d", len(all_docs))

    # ------------------------------------------------------------------
    # Step 2: Chunk each document
    # ------------------------------------------------------------------
    t0 = time.time()
    all_chunks = []
    doc_stats: list[dict] = []
    failed_docs: list[str] = []

    for i, doc in enumerate(all_docs, 1):
        try:
            detected = detect_doc_type(doc) if args.strategy == "auto" else args.strategy
            logger.info(
                "[%d/%d] Chunking %-42s  category=%-20s  detected=%s",
                i, len(all_docs), doc.doc_id[:42], doc.category, detected,
            )
            chunks = _chunk_doc(doc, args.strategy, args.max_tokens, args.overlap)

            all_chunks.extend(chunks)
            doc_stats.append({
                "doc_id":    doc.doc_id,
                "category":  doc.category,
                "strategy":  detected,
                "chunks":    len(chunks),
                "words":     doc.word_count,
            })

        except Exception as exc:
            logger.error("FAILED to chunk '%s': %s", doc.doc_id, exc, exc_info=True)
            failed_docs.append(doc.doc_id)

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Step 3: Check for duplicate chunk_ids (should never happen)
    # ------------------------------------------------------------------
    ids = [c.chunk_id for c in all_chunks]
    unique_ids = set(ids)
    if len(ids) != len(unique_ids):
        dupes = len(ids) - len(unique_ids)
        logger.warning("%d duplicate chunk_ids detected — check doc_id uniqueness", dupes)
    else:
        logger.info("chunk_id uniqueness check: OK (%d unique)", len(unique_ids))

    # ------------------------------------------------------------------
    # Step 4: Save to chunks/all_chunks.json
    # ------------------------------------------------------------------
    output_payload = {
        "meta": {
            "total_chunks":    len(all_chunks),
            "total_docs":      len(all_docs),
            "failed_docs":     failed_docs,
            "strategy":        args.strategy,
            "max_tokens":      args.max_tokens,
            "overlap_tokens":  args.overlap,
            "elapsed_seconds": round(elapsed, 2),
            "doc_stats":       doc_stats,
        },
        "chunks": [_chunk_to_dict(c) for c in all_chunks],
    }

    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Step 5: Print summary report
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("CHUNKING COMPLETE")
    logger.info("  Total documents : %d", len(all_docs))
    logger.info("  Failed docs     : %d  %s", len(failed_docs), failed_docs or "")
    logger.info("  Total chunks    : %d", len(all_chunks))
    logger.info("  Output file     : %s", output_file)
    logger.info("  Elapsed         : %.1f seconds", elapsed)
    logger.info("=" * 60)

    # Per-category breakdown
    from collections import Counter
    cat_counts = Counter(c.category for c in all_chunks)
    strat_counts = Counter(c.metadata.get("delegated_strategy", c.strategy) for c in all_chunks)

    logger.info("Chunks by category:")
    for cat, count in sorted(cat_counts.items()):
        logger.info("  %-25s : %d", cat, count)

    logger.info("Chunks by delegated strategy:")
    for strat, count in sorted(strat_counts.items()):
        logger.info("  %-25s : %d", strat, count)

    if failed_docs:
        logger.warning("The following documents failed to chunk:")
        for d in failed_docs:
            logger.warning("  - %s", d)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MedRAG document chunking pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",   default=str(config.DATA_DIR),   help="Root data directory")
    p.add_argument("--output-dir", default=str(config.CHUNKS_DIR),  help="Output directory for chunks")
    p.add_argument("--max-tokens", type=int, default=config.MAX_CHUNK_SIZE, help="Max tokens per chunk")
    p.add_argument("--overlap",    type=int, default=config.CHUNK_OVERLAP,  help="Overlap tokens between chunks")
    p.add_argument(
        "--strategy",
        choices=["auto", "sentence", "recursive", "semantic"],
        default="auto",
        help="Chunking strategy. 'auto' uses agentic_chunk() to pick per document.",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity",
    )
    return p


if __name__ == "__main__":
    main(_build_parser().parse_args())
