"""
api.py — FastAPI REST API for MedRAG.

Endpoints:
  POST /api/query   — run full pipeline, return answer + sources
  GET  /api/health  — check model and DB status
  GET  /api/metrics — runtime statistics

Usage:
    source .venv/bin/activate
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.models import (
    QueryRequest, QueryResponse, DualQueryResponse, HealthResponse,
    MetricsResponse, SourceChunk,
)

logger = logging.getLogger("medrag.api")

app = FastAPI(
    title="MedRAG — Oncology Q&A API",
    description="Multilingual medical question answering using RAG over oncology documents.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins for development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup state
# ---------------------------------------------------------------------------

_start_time: float = time.time()
_stats: dict = {
    "total_queries":     0,
    "total_latency_ms":  0.0,
    "latencies":         [],         # rolling window for p95
    "fallback_count":    0,
    "query_type_counts": defaultdict(int),
    "language_counts":   defaultdict(int),
    "rl_scores":         [],
}


@app.on_event("startup")
async def startup_event() -> None:
    """Pre-warm models and verify ChromaDB on startup."""
    logger.info("MedRAG API starting up...")

    # Verify ChromaDB collections are populated
    try:
        from src.vectordb import get_collection
        coarse = get_collection(config.DIM_COARSE)
        fine   = get_collection(config.DIM_FINE)
        logger.info(
            "ChromaDB OK — coarse=%d chunks, fine=%d chunks",
            coarse.count(), fine.count(),
        )
    except Exception as exc:
        logger.error("ChromaDB startup check failed: %s", exc)

    logger.info("MedRAG API ready at http://%s:%d", config.API_HOST, config.API_PORT)


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Submit a medical question and receive a grounded oncology answer.

    Runs the full pipeline:
      LAQA → Query Expansion → Agentic RAG → Prompt → Generator → RLAIF
    """
    global _stats

    logger.info(
        "POST /api/query — query=%r  lang=%s  type=%s  top_k=%d",
        request.query[:60], request.language, request.query_type, request.top_k,
    )

    t0 = time.time()

    try:
        from src.pipeline import answer_query

        response = answer_query(
            query=request.query,
            language=request.language,
            query_type=request.query_type,
            top_k=request.top_k,
            agentic_pattern=request.agentic_pattern,
            use_hyde=True,
            use_multi_query=True,
            use_rlaif=config.RLAIF_ENABLED,
        )

    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(exc)[:200]}",
        )

    latency_ms = (time.time() - t0) * 1000

    # Update stats
    _stats["total_queries"]     += 1
    _stats["total_latency_ms"]  += latency_ms
    _stats["latencies"].append(latency_ms)
    if len(_stats["latencies"]) > 1000:
        _stats["latencies"] = _stats["latencies"][-1000:]
    _stats["query_type_counts"][response.query_type] += 1
    _stats["language_counts"][response.detected_language] += 1
    if "fallback" in response.model_used:
        _stats["fallback_count"] += 1
    if response.rl_score:
        _stats["rl_scores"].append(response.rl_score.composite)
        if len(_stats["rl_scores"]) > 1000:
            _stats["rl_scores"] = _stats["rl_scores"][-1000:]

    # Build source chunks for response
    sources = [
        SourceChunk(
            chunk_id=c.chunk_id,
            document=Path(c.source_file).name,
            category=c.category,
            text=c.text[:300],
            score=round(c.score, 4),
        )
        for c in response.retrieved_chunks[:request.top_k]
    ]

    return QueryResponse(
        answer=response.answer,
        query_type=response.query_type,
        language=response.detected_language,
        sources=sources,
        agentic_pattern_used=response.agentic_pattern_used,
        rl_score=response.rl_score.composite if response.rl_score else None,
        latency_ms=round(latency_ms, 1),
        model_used=response.model_used,
    )


# ---------------------------------------------------------------------------
# POST /api/query/dual
# ---------------------------------------------------------------------------

@app.post("/api/query/dual", response_model=DualQueryResponse)
async def dual_query_endpoint(request: QueryRequest) -> DualQueryResponse:
    """
    Submit a medical question and receive two grounded answers:
    one written for patients and one for clinical reference.

    Runs LAQA + retrieval once, then generates both answers.
    """
    logger.info(
        "POST /api/query/dual — query=%r  lang=%s  type=%s",
        request.query[:60], request.language, request.query_type,
    )

    try:
        from src.pipeline import answer_query_dual

        result = answer_query_dual(
            query=request.query,
            language=request.language,
            query_type=request.query_type,
            top_k=request.top_k,
            agentic_pattern=request.agentic_pattern,
            use_hyde=True,
            use_multi_query=True,
        )
    except Exception as exc:
        logger.error("Dual pipeline error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(exc)[:200]}",
        )

    sources = [
        SourceChunk(
            chunk_id=c.chunk_id,
            document=Path(c.source_file).name,
            category=c.category,
            text=c.text[:300],
            score=round(c.score, 4),
        )
        for c in result["sources"][:request.top_k]
    ]

    return DualQueryResponse(
        patient_answer=result["patient_answer"],
        doctor_answer=result["doctor_answer"],
        query_type=result["query_type"],
        language=result["language"],
        sources=sources,
        agentic_pattern_used=result["agentic_pattern_used"],
        agent_iterations=result["agent_iterations"],
        latency_ms=result["latency_ms"],
        model_used=result["model_used"],
    )


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """Check API server, ChromaDB, and model status."""
    chromadb_ok    = False
    coarse_count   = 0
    fine_count     = 0
    medgemma_loaded = False

    try:
        from src.vectordb import get_collection
        coarse_count   = get_collection(config.DIM_COARSE).count()
        fine_count     = get_collection(config.DIM_FINE).count()
        chromadb_ok    = coarse_count > 0 and fine_count > 0
    except Exception as exc:
        logger.warning("Health check — ChromaDB error: %s", exc)

    # Check if MedGemma is cached in memory
    try:
        from src.generator import _medgemma_cache
        medgemma_loaded = "model" in _medgemma_cache
    except Exception:
        pass

    uptime = time.time() - _start_time

    return HealthResponse(
        status="ok" if chromadb_ok else "degraded",
        medgemma_loaded=medgemma_loaded,
        chromadb_connected=chromadb_ok,
        chunks_128_count=coarse_count,
        chunks_768_count=fine_count,
        uptime_seconds=round(uptime, 1),
    )


# ---------------------------------------------------------------------------
# GET /api/metrics
# ---------------------------------------------------------------------------

@app.get("/api/metrics", response_model=MetricsResponse)
async def metrics_endpoint() -> MetricsResponse:
    """Return aggregated runtime statistics."""
    total = _stats["total_queries"]

    avg_latency = (
        _stats["total_latency_ms"] / total if total > 0 else 0.0
    )

    # p95 latency
    lats = sorted(_stats["latencies"])
    p95_latency = lats[int(len(lats) * 0.95)] if lats else 0.0

    # Average RL score
    rl_scores = _stats["rl_scores"]
    avg_rl = sum(rl_scores) / len(rl_scores) if rl_scores else 0.0

    # Fallback rate
    fallback_rate = _stats["fallback_count"] / max(total, 1)

    # Distributions
    qt_counts = dict(_stats["query_type_counts"])
    lang_counts = dict(_stats["language_counts"])

    qt_dist = {k: round(v / max(total, 1), 3) for k, v in qt_counts.items()}
    lang_dist = {k: round(v / max(total, 1), 3) for k, v in lang_counts.items()}

    return MetricsResponse(
        total_queries=total,
        avg_latency_ms=round(avg_latency, 1),
        p95_latency_ms=round(p95_latency, 1),
        avg_rl_score=round(avg_rl, 4),
        fallback_rate=round(fallback_rate, 4),
        query_type_distribution=qt_dist,
        language_distribution=lang_dist,
    )


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "MedRAG Oncology Q&A API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/api/health",
        "query":   "POST /api/query",
    }


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower(),
    )
