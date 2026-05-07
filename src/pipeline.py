"""
src/pipeline.py — End-to-end MedRAG pipeline orchestrator.

answer_query() is the single entry point that wires every module:
  LAQA → Query Expansion → Agentic RAG → Prompt Builder → Generator → RL Scorer

Returns a complete PipelineResponse ready for the FastAPI layer.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional

import config
from src.models import (
    AnalysedQuery, PipelineResponse, RetrievedChunk,
    GeneratedAnswer, RLScore, QueryRequest,
)

logger = logging.getLogger("medrag.pipeline")


def answer_query(
    query: str,
    language: str = "auto",
    query_type: str = "auto",
    top_k: int = None,
    agentic_pattern: str = "auto",
    use_hyde: bool = True,
    use_multi_query: bool = True,
    use_rlaif: bool = None,
) -> PipelineResponse:
    """
    Full end-to-end MedRAG pipeline.

    Steps:
      1. LAQA           — detect language, classify query type, extract NER
      2. Query Expansion — translate, synonym expansion, HyDE, multi-query
      3. Agentic RAG     — orchestrate retrieval pattern, run 3-stage retrieval
      4. Prompt Builder  — construct query-type-specific prompt with context
      5. Generator       — generate answer (MedGemma → Claude fallback)
      6. Post Checks     — safety, length, grounding validation
      7. RL Scorer       — RLAIF scoring + SQLite storage (if enabled)

    Args:
        query:           Raw user query in any supported language.
        language:        "auto" or a Language enum value to force.
        query_type:      "auto" or a QueryType enum value to force.
        top_k:           Number of final chunks (default STAGE3_TOP_K=5).
        agentic_pattern: "auto" or force "router"/"iterative"/"decomposition"/"standard".
        use_hyde:        Whether to run HyDE expansion (needs Claude API).
        use_multi_query: Whether to run multi-query generation (needs Claude API).
        use_rlaif:       Whether to run RLAIF scoring. Defaults to config.RLAIF_ENABLED.

    Returns:
        PipelineResponse with all fields populated.
    """
    from src.laqa import laqa_pipeline
    from src.query_expansion import full_query_expansion
    from src.agentic_rag import orchestrator
    from src.prompt_builder import build_prompt
    from src.generator import generate
    from src.rl_scorer import compute_reward

    if top_k is None:
        top_k = config.STAGE3_TOP_K
    if use_rlaif is None:
        use_rlaif = config.RLAIF_ENABLED

    total_t0 = time.time()

    logger.info("=" * 55)
    logger.info("MedRAG Pipeline START")
    logger.info("  query=%r", query[:80])
    logger.info("  language=%s  query_type=%s  top_k=%d", language, query_type, top_k)
    logger.info("=" * 55)

    # ------------------------------------------------------------------
    # Step 1: LAQA
    # ------------------------------------------------------------------
    t0 = time.time()
    force_lang = None if language == "auto" else language
    force_type = None if query_type == "auto" else query_type

    analysed: AnalysedQuery = laqa_pipeline(
        query,
        force_language=force_lang,
        force_query_type=force_type,
    )
    logger.info(
        "[1/7] LAQA: lang=%s  type=%s  entities=%s  (%.2fs)",
        analysed.detected_language, analysed.query_type,
        analysed.entity_texts[:3], time.time() - t0,
    )

    # ------------------------------------------------------------------
    # Step 2: Query Expansion
    # ------------------------------------------------------------------
    t0 = time.time()
    expanded: AnalysedQuery = full_query_expansion(
        analysed,
        use_hyde=use_hyde,
        use_multi_query=use_multi_query,
        use_synonyms=True,
    )
    logger.info(
        "[2/7] Expansion: translated=%r  expanded_count=%d  (%.2fs)",
        expanded.translated_query[:60],
        len(expanded.expanded_queries),
        time.time() - t0,
    )

    # ------------------------------------------------------------------
    # Step 3: Agentic RAG — retrieval
    # ------------------------------------------------------------------
    t0 = time.time()

    # Override pattern if specified
    if agentic_pattern != "auto":
        from src.retriever import stage1_coarse_search, stage2_fine_rerank
        from src.reranker import cross_encoder_rerank
        from src.agentic_rag import (
            router_agent, iterative_retrieval_agent, decomposition_agent
        )

        translated = expanded.translated_query
        agent_iters = 1
        if agentic_pattern == "router":
            chunks = router_agent(translated, expanded, n_results=top_k)
            pattern_used = "router"
        elif agentic_pattern == "iterative":
            chunks, agent_iters = iterative_retrieval_agent(translated, expanded, n_results=top_k)
            pattern_used = "iterative"
        elif agentic_pattern == "decomposition":
            chunks = decomposition_agent(translated, expanded, n_results=top_k)
            pattern_used = "decomposition"
        else:
            s1 = stage1_coarse_search(translated, analysed_query=expanded,
                                      n_results=config.STAGE1_TOP_K)
            s2 = stage2_fine_rerank(translated, s1, n_results=config.STAGE2_TOP_K)
            chunks = cross_encoder_rerank(translated, s2, top_k=top_k)
            pattern_used = "standard"
    else:
        chunks, pattern_used, agent_iters = orchestrator(expanded, n_results=top_k)

    logger.info(
        "[3/7] Retrieval: pattern=%s  chunks=%d  top_score=%.4f  (%.2fs)",
        pattern_used, len(chunks),
        chunks[0].score if chunks else 0.0,
        time.time() - t0,
    )

    # ------------------------------------------------------------------
    # Step 4: Prompt Builder
    # ------------------------------------------------------------------
    t0 = time.time()
    prompt_dict = build_prompt(
        query=expanded.translated_query,
        chunks=chunks,
        query_type=expanded.query_type,
        detected_language=expanded.detected_language,
        max_chunks=min(top_k, 5),
    )
    logger.info(
        "[4/7] Prompt: type=%s  system=%dc  user=%dc  (%.2fs)",
        prompt_dict["query_type"],
        len(prompt_dict["system"]),
        len(prompt_dict["user"]),
        time.time() - t0,
    )

    # ------------------------------------------------------------------
    # Step 5 + 6: Generate + Post-checks
    # ------------------------------------------------------------------
    t0 = time.time()
    generated: GeneratedAnswer = generate(
        prompt_dict=prompt_dict,
        chunks=chunks,
        query=expanded.translated_query,
        max_tokens=1024,
        prefer_local=True,
    )
    logger.info(
        "[5+6/7] Generate: model=%s  len=%d  safety=%s  flags=%s  (%.2fs)",
        generated.model_used,
        len(generated.answer_text),
        generated.passed_safety_check,
        generated.safety_flags or "none",
        time.time() - t0,
    )

    # ------------------------------------------------------------------
    # Step 7: RL Scoring
    # ------------------------------------------------------------------
    rl_score: Optional[RLScore] = None
    if use_rlaif and config.ANTHROPIC_API_KEY:
        t0 = time.time()
        group_id = hashlib.md5(query.encode()).hexdigest()[:12]
        try:
            rl_score = compute_reward(
                query=expanded.translated_query,
                answer=generated.answer_text,
                chunks=chunks,
                query_type=expanded.query_type,
                model_used=generated.model_used,
                group_id=group_id,
            )
            logger.info(
                "[7/7] RLAIF: composite=%.3f  (%.2fs)",
                rl_score.composite, time.time() - t0,
            )
        except Exception as exc:
            logger.warning("[7/7] RLAIF scoring failed (%s) — skipping", exc)

    # ------------------------------------------------------------------
    # Assemble PipelineResponse
    # ------------------------------------------------------------------
    total_latency_ms = (time.time() - total_t0) * 1000

    # Clean up model_used label
    model_label = generated.model_used
    if model_label.startswith("claude-claude-"):
        model_label = model_label.replace("claude-claude-", "claude-")
    elif model_label == "claude-fallback":
        model_label = "claude-fallback"

    response = PipelineResponse(
        query=query,
        answer=generated.answer_text,
        query_type=expanded.query_type,
        detected_language=expanded.detected_language,
        retrieved_chunks=chunks,
        agentic_pattern_used=pattern_used,
        model_used=model_label,
        rl_score=rl_score,
        total_latency_ms=total_latency_ms,
        agent_iterations=agent_iters,
        query_type_confidence=analysed.query_type_confidence,
    )

    logger.info(
        "MedRAG Pipeline DONE — %.0fms  model=%s  rl=%.3f",
        total_latency_ms,
        model_label,
        rl_score.composite if rl_score else -1,
    )

    return response


def answer_query_dual(
    query: str,
    language: str = "auto",
    query_type: str = "auto",
    top_k: int = None,
    agentic_pattern: str = "auto",
    use_hyde: bool = True,
    use_multi_query: bool = True,
) -> dict:
    """
    Runs LAQA + retrieval once, then generates two answers in parallel:
    one patient-facing and one for clinical reference.

    Returns a plain dict (not PipelineResponse) suitable for the dual endpoint.
    """
    from src.laqa import laqa_pipeline
    from src.query_expansion import full_query_expansion
    from src.agentic_rag import orchestrator
    from src.prompt_builder import build_prompt
    from src.generator import generate, clean_answer

    if top_k is None:
        top_k = config.STAGE3_TOP_K

    total_t0 = time.time()

    logger.info("=" * 55)
    logger.info("MedRAG Dual Pipeline START  query=%r", query[:80])
    logger.info("=" * 55)

    # Step 1: LAQA
    force_lang = None if language == "auto" else language
    force_type = None if query_type == "auto" else query_type
    analysed: AnalysedQuery = laqa_pipeline(
        query, force_language=force_lang, force_query_type=force_type,
    )
    logger.info("[1/5] LAQA: lang=%s  type=%s", analysed.detected_language, analysed.query_type)

    # Step 2: Query Expansion
    expanded: AnalysedQuery = full_query_expansion(
        analysed, use_hyde=use_hyde, use_multi_query=use_multi_query, use_synonyms=True,
    )
    logger.info("[2/5] Expansion: translated=%r", expanded.translated_query[:60])

    # Step 3: Retrieval (once — shared for both audiences)
    chunks, pattern_used, agent_iters = orchestrator(expanded, n_results=top_k)
    logger.info("[3/5] Retrieval: pattern=%s  chunks=%d", pattern_used, len(chunks))

    # Step 4: Build both prompts
    common = dict(
        query=expanded.translated_query,
        chunks=chunks,
        query_type=expanded.query_type,
        detected_language=expanded.detected_language,
        max_chunks=min(top_k, 5),
    )
    patient_prompt = build_prompt(**common, audience="patient")
    doctor_prompt  = build_prompt(**common, audience="doctor")
    logger.info("[4/5] Prompts built for patient + doctor audiences")

    # Step 5: Generate both answers sequentially
    patient_gen = generate(
        prompt_dict=patient_prompt, chunks=chunks,
        query=expanded.translated_query, max_tokens=1024, prefer_local=True,
    )
    doctor_gen = generate(
        prompt_dict=doctor_prompt, chunks=chunks,
        query=expanded.translated_query, max_tokens=1024, prefer_local=True,
    )
    logger.info("[5/5] Generated patient + doctor answers  model=%s", patient_gen.model_used)

    latency_ms = round((time.time() - total_t0) * 1000, 1)
    logger.info("MedRAG Dual Pipeline DONE — %.0fms", latency_ms)

    return {
        "patient_answer":       clean_answer(patient_gen.answer_text),
        "doctor_answer":        clean_answer(doctor_gen.answer_text),
        "query_type":           expanded.query_type,
        "language":             expanded.detected_language,
        "sources":              chunks,
        "agentic_pattern_used": pattern_used,
        "agent_iterations":     agent_iters,
        "model_used":           patient_gen.model_used,
        "latency_ms":           latency_ms,
    }
