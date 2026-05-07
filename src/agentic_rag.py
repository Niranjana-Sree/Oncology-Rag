"""
src/agentic_rag.py — Agentic RAG orchestration patterns for MedRAG.

Three retrieval patterns implemented across features 4.4–4.7:
  4.4  router_agent()              — route query to best corpus partition
  4.5  iterative_retrieval_agent() — think-act-observe-reflect loop
  4.6  decomposition_agent()       — break complex query into sub-questions
  4.7  orchestrator()              — select pattern, run full 3-stage retrieval

All agents return list[RetrievedChunk] after full 3-stage retrieval.
orchestrator() returns tuple[list[RetrievedChunk], str, int]:
  - chunks, pattern_used, agent_iterations
"""

from __future__ import annotations

import logging
from typing import Optional

import config
from src.models import RetrievedChunk, AnalysedQuery

logger = logging.getLogger("medrag.agentic_rag")


# ---------------------------------------------------------------------------
# Router Agent  (Feature 4.4)
# ---------------------------------------------------------------------------

# Category routing rules for each query type
_QUERY_TYPE_TO_CATEGORIES: dict[str, list[str]] = {
    "fact_verification": ["clinical_guideline"],           # authoritative source only
    "mcqa":              ["clinical_guideline", "general"], # guidelines + textbooks
    "lfqa":              ["general", "research_paper"],     # detailed explanations
    "jeopardy":          ["general", "clinical_guideline"], # broad knowledge
    "qa":                [],                                # all categories
    "fill_blank":        [],                                # all categories
}


def _build_category_filter(categories: list[str]) -> Optional[dict]:
    """
    Build a ChromaDB `where` filter for category restriction.
    Returns None if categories is empty (search all).
    """
    if not categories:
        return None
    if len(categories) == 1:
        return {"category": categories[0]}
    return {"category": {"$in": categories}}


def router_agent(
    translated_query: str,
    analysed_query: AnalysedQuery,
    n_results: int = None,
) -> list[RetrievedChunk]:
    """
    Router Agent: route the query to the most relevant corpus partition.

    Uses the query type from LAQA to determine which document categories
    are most likely to contain the answer, then restricts Stage 1 retrieval
    to those categories.

    Routing rules:
      fact_verification → clinical_guideline only
          (claims must be verified against authoritative guidelines)
      mcqa              → clinical_guideline + general
          (multiple choice needs authoritative + comprehensive sources)
      lfqa              → general + research_paper
          (long-form explanations need textbooks and research)
      jeopardy          → general + clinical_guideline
          (broad medical knowledge)
      qa / fill_blank   → all categories (no restriction)

    After routing, runs full Stage 1 → 2 → 3 retrieval within the
    selected partition.

    Args:
        translated_query: English query string.
        analysed_query:   AnalysedQuery with query_type and expanded_queries.
        n_results:        Final number of chunks to return (default STAGE3_TOP_K=5).

    Returns:
        Top n_results RetrievedChunk objects from the routed partition.
    """
    from src.retriever import stage1_coarse_search, stage2_fine_rerank
    from src.reranker import cross_encoder_rerank

    if n_results is None:
        n_results = config.STAGE3_TOP_K

    query_type = analysed_query.query_type
    categories = _QUERY_TYPE_TO_CATEGORIES.get(query_type, [])
    where_filter = _build_category_filter(categories)

    logger.info(
        "router_agent: query_type=%s → categories=%s",
        query_type, categories or "ALL",
    )

    # Stage 1 — restricted to routed categories
    stage1 = stage1_coarse_search(
        translated_query,
        analysed_query=analysed_query,
        n_results=config.STAGE1_TOP_K,
        where=where_filter,
    )

    if not stage1:
        logger.warning(
            "router_agent: no results in routed categories %s — retrying without filter",
            categories,
        )
        stage1 = stage1_coarse_search(
            translated_query,
            analysed_query=analysed_query,
            n_results=config.STAGE1_TOP_K,
        )

    # Stage 2
    stage2 = stage2_fine_rerank(
        translated_query, stage1, n_results=config.STAGE2_TOP_K
    )

    # Stage 3
    stage3 = cross_encoder_rerank(
        translated_query, stage2, top_k=n_results
    )

    logger.info(
        "router_agent: %d → %d → %d chunks  [type=%s, cats=%s]",
        len(stage1), len(stage2), len(stage3),
        query_type, categories or "ALL",
    )
    return stage3


# ---------------------------------------------------------------------------
# Iterative Retrieval Agent  (Feature 4.5)
# ---------------------------------------------------------------------------

_COVERAGE_EVAL_PROMPT = """You are an oncology retrieval quality evaluator.

Query: {query}

Retrieved context:
{context}

Evaluate whether the retrieved context sufficiently covers the query.
Respond with a JSON object with exactly these fields:
{{
  "coverage_score": <float 0.0-1.0>,
  "is_sufficient": <true/false>,
  "missing_aspects": ["aspect1", "aspect2"],
  "follow_up_query": "<refined search query to find missing information, or empty string if sufficient>"
}}

coverage_score: 0.0 = completely missing, 1.0 = fully covered
is_sufficient: true if coverage_score >= 0.7
missing_aspects: list what is missing (empty list if sufficient)
follow_up_query: a targeted query to retrieve the missing information"""


def _evaluate_coverage(
    query: str,
    chunks: list[RetrievedChunk],
) -> dict:
    """
    Ask Claude to evaluate whether retrieved chunks sufficiently cover the query.
    Returns a dict with coverage_score, is_sufficient, missing_aspects, follow_up_query.
    Falls back to sufficient=True if Claude API is unavailable.
    """
    import json

    # Build context from top chunks
    context_parts = []
    for i, c in enumerate(chunks[:5], 1):
        context_parts.append(f"[{i}] {c.text[:300]}")
    context = "\n\n".join(context_parts)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

        prompt = _COVERAGE_EVAL_PROMPT.format(query=query, context=context)
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Parse JSON from response
        import re
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            logger.debug(
                "coverage_eval: score=%.2f sufficient=%s follow_up=%r",
                result.get("coverage_score", 0),
                result.get("is_sufficient", True),
                result.get("follow_up_query", "")[:50],
            )
            return result

    except Exception as exc:
        logger.warning("_evaluate_coverage: Claude failed (%s) — assuming sufficient", exc)

    return {
        "coverage_score": 1.0,
        "is_sufficient": True,
        "missing_aspects": [],
        "follow_up_query": "",
    }


def iterative_retrieval_agent(
    translated_query: str,
    analysed_query: AnalysedQuery,
    max_iterations: int = 3,
    coverage_threshold: float = 0.70,
    n_results: int = None,
) -> list[RetrievedChunk]:
    """
    Iterative Retrieval Agent: think-act-observe-reflect loop.

    Best for LFQA queries where a single retrieval pass may miss aspects
    of a complex multi-faceted medical question.

    Loop (max_iterations):
      THINK   — what do I need to find?
      ACT     — retrieve chunks for current query
      OBSERVE — evaluate coverage with Claude
      REFLECT — if insufficient, refine query and iterate

    Iteration 1: retrieve with original query + expansions
    Iteration 2: retrieve with Claude-suggested follow-up query
    Iteration 3: merge all results, final cross-encoder rerank

    Args:
        translated_query:   English query string.
        analysed_query:     AnalysedQuery with expanded_queries.
        max_iterations:     Maximum retrieval rounds (default 3).
        coverage_threshold: Coverage score above which to stop early (default 0.70).
        n_results:          Final chunks to return (default STAGE3_TOP_K=5).

    Returns:
        Top n_results RetrievedChunk from merged iterative retrieval.
    """
    from src.retriever import stage1_coarse_search, stage2_fine_rerank
    from src.reranker import cross_encoder_rerank

    if n_results is None:
        n_results = config.STAGE3_TOP_K

    logger.info(
        "iterative_retrieval_agent: query=%r max_iter=%d threshold=%.2f",
        translated_query[:60], max_iterations, coverage_threshold,
    )

    all_chunks: dict[str, RetrievedChunk] = {}   # chunk_id → best chunk
    current_query = translated_query
    current_analysed = analysed_query
    actual_iters = 1

    for iteration in range(1, max_iterations + 1):
        actual_iters = iteration
        logger.info("  [iter %d/%d] query=%r", iteration, max_iterations, current_query[:60])

        # ACT — retrieve
        stage1 = stage1_coarse_search(
            current_query,
            analysed_query=current_analysed if iteration == 1 else None,
            n_results=config.STAGE1_TOP_K,
        )
        stage2 = stage2_fine_rerank(current_query, stage1, n_results=config.STAGE2_TOP_K)

        # Accumulate unique chunks (keep highest score per chunk_id)
        new_count = 0
        for chunk in stage2:
            cid = chunk.chunk_id
            if cid not in all_chunks or chunk.score > all_chunks[cid].score:
                all_chunks[cid] = chunk
                new_count += 1

        logger.info(
            "  [iter %d] retrieved %d new chunks (pool=%d total)",
            iteration, new_count, len(all_chunks),
        )

        # OBSERVE — evaluate coverage
        current_top = sorted(all_chunks.values(), key=lambda c: c.score, reverse=True)
        eval_result = _evaluate_coverage(translated_query, current_top[:5])

        coverage = eval_result.get("coverage_score", 1.0)
        is_sufficient = eval_result.get("is_sufficient", True)
        follow_up = eval_result.get("follow_up_query", "").strip()
        missing = eval_result.get("missing_aspects", [])

        logger.info(
            "  [iter %d] coverage=%.2f sufficient=%s missing=%s",
            iteration, coverage, is_sufficient, missing[:2],
        )

        # REFLECT — stop or refine
        if is_sufficient or coverage >= coverage_threshold:
            logger.info("  Coverage sufficient (%.2f) — stopping at iter %d", coverage, iteration)
            break

        if not follow_up or iteration == max_iterations:
            logger.info("  No follow-up query — stopping at iter %d", iteration)
            break

        # Prepare next iteration with follow-up query
        logger.info("  Follow-up query: %r", follow_up[:60])
        current_query = follow_up
        current_analysed = None  # no expansion for follow-up queries

    # Final cross-encoder rerank over accumulated pool
    pool = sorted(all_chunks.values(), key=lambda c: c.score, reverse=True)
    top_for_rerank = pool[:config.STAGE2_TOP_K]
    final = cross_encoder_rerank(translated_query, top_for_rerank, top_k=n_results)

    logger.info(
        "iterative_retrieval_agent: done — pool=%d → final=%d chunks",
        len(all_chunks), len(final),
    )
    return final, actual_iters


# ---------------------------------------------------------------------------
# Decomposition Agent  (Feature 4.6)
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """You are an expert oncology question decomposer.

Break the following complex medical query into {n} simpler, atomic sub-questions.
Each sub-question should:
- Be self-contained and answerable independently
- Cover a distinct aspect of the original query
- Be specific enough to retrieve targeted information from an oncology corpus

Return ONLY a numbered list of sub-questions, one per line.
Do not include any explanation or the original query.

Complex query: {query}"""


def _decompose_query(query: str, n: int = 3) -> list[str]:
    """
    Use Claude to decompose a complex query into n atomic sub-questions.
    Falls back to [query] if Claude API is unavailable.
    """
    import re

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

        prompt = _DECOMPOSE_PROMPT.format(query=query, n=n)
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Parse numbered list
        sub_questions = []
        for line in raw.splitlines():
            line = line.strip()
            match = re.match(r"^[\d]+[.)]\s+(.+)$", line)
            if match:
                sq = match.group(1).strip()
                if len(sq) > 10:
                    sub_questions.append(sq)

        if sub_questions:
            logger.info(
                "_decompose_query: decomposed into %d sub-questions", len(sub_questions)
            )
            for i, sq in enumerate(sub_questions, 1):
                logger.debug("  sub-q%d: %r", i, sq[:60])
            return sub_questions[:n]

        logger.warning("_decompose_query: could not parse response — using original query")
        return [query]

    except Exception as exc:
        logger.warning("_decompose_query: Claude failed (%s) — using original query", exc)
        return [query]


def decomposition_agent(
    translated_query: str,
    analysed_query: AnalysedQuery,
    n_subquestions: int = 3,
    n_results: int = None,
) -> list[RetrievedChunk]:
    """
    Decomposition Agent: break complex queries into atomic sub-questions.

    Best for jeopardy-style and complex multi-part queries where a single
    embedding cannot capture all the information needs simultaneously.

    Algorithm:
      1. Use Claude to decompose the query into n atomic sub-questions.
      2. Run full Stage 1 → 2 retrieval independently for each sub-question.
      3. Merge all retrieved chunks into a unified pool (max-score fusion).
      4. Run cross-encoder rerank on the merged pool using the ORIGINAL query.

    The cross-encoder in step 4 uses the original query — not sub-questions —
    so the final ranking reflects relevance to the complete question.

    Args:
        translated_query:  English query string.
        analysed_query:    AnalysedQuery with expanded_queries.
        n_subquestions:    Number of sub-questions to generate (default 3).
        n_results:         Final chunks to return (default STAGE3_TOP_K=5).

    Returns:
        Top n_results RetrievedChunk from merged decomposed retrieval.
    """
    from src.retriever import stage1_coarse_search, stage2_fine_rerank
    from src.reranker import cross_encoder_rerank

    if n_results is None:
        n_results = config.STAGE3_TOP_K

    logger.info(
        "decomposition_agent: query=%r n_sub=%d",
        translated_query[:60], n_subquestions,
    )

    # Step 1: Decompose into sub-questions
    sub_questions = _decompose_query(translated_query, n=n_subquestions)
    logger.info(
        "decomposition_agent: %d sub-questions generated", len(sub_questions)
    )

    # Step 2: Retrieve for each sub-question independently
    all_chunks: dict[str, RetrievedChunk] = {}

    for i, sub_q in enumerate(sub_questions, 1):
        logger.info("  [sub-q %d/%d] %r", i, len(sub_questions), sub_q[:60])

        # Use expanded queries only for the first sub-question
        # to avoid redundant expansion overhead
        aq = analysed_query if i == 1 else None

        stage1 = stage1_coarse_search(
            sub_q,
            analysed_query=aq,
            n_results=config.STAGE1_TOP_K,
        )
        stage2 = stage2_fine_rerank(sub_q, stage1, n_results=config.STAGE2_TOP_K)

        # Max-score fusion across all sub-questions
        new_count = 0
        for chunk in stage2:
            cid = chunk.chunk_id
            if cid not in all_chunks or chunk.score > all_chunks[cid].score:
                all_chunks[cid] = chunk
                new_count += 1

        logger.info(
            "  [sub-q %d] retrieved %d chunks (pool=%d)",
            i, new_count, len(all_chunks),
        )

    # Step 3: Sort merged pool by score
    pool = sorted(all_chunks.values(), key=lambda c: c.score, reverse=True)
    logger.info(
        "decomposition_agent: merged pool=%d unique chunks from %d sub-questions",
        len(pool), len(sub_questions),
    )

    # Step 4: Cross-encoder rerank using ORIGINAL query (not sub-questions)
    top_for_rerank = pool[:config.STAGE2_TOP_K]
    final = cross_encoder_rerank(translated_query, top_for_rerank, top_k=n_results)

    logger.info(
        "decomposition_agent: done — pool=%d → final=%d chunks",
        len(pool), len(final),
    )
    return final


# ---------------------------------------------------------------------------
# Orchestrator  (Feature 4.7)
# ---------------------------------------------------------------------------

# Pattern selection rules based on query type
_QUERY_TYPE_TO_PATTERN: dict[str, str] = {
    "fact_verification": "router",        # route to clinical_guideline partition
    "mcqa":              "router",        # route to authoritative sources
    "fill_blank":        "router",        # simple retrieval from right partition
    "lfqa":              "iterative",     # multi-aspect coverage needed
    "jeopardy":          "decomposition", # complex implicit question
    "qa":                "standard",      # standard 3-stage retrieval
}


def orchestrator(
    analysed_query: AnalysedQuery,
    n_results: int = None,
) -> tuple[list[RetrievedChunk], str]:
    """
    Agentic RAG orchestrator: select and run the best retrieval pattern.

    Pattern selection (based on query_type from LAQA):
      fact_verification → router_agent()        (clinical_guideline partition)
      mcqa              → router_agent()        (guideline + general partition)
      fill_blank        → router_agent()        (all categories, routed)
      lfqa              → iterative_retrieval_agent() (think-act-observe-reflect)
      jeopardy          → decomposition_agent() (sub-question decomposition)
      qa                → standard 3-stage      (stage1 → stage2 → stage3)

    Args:
        analysed_query: Enriched AnalysedQuery from full_query_expansion().
                        Must have translated_query and expanded_queries set.
        n_results:      Number of final chunks to return (default STAGE3_TOP_K=5).

    Returns:
        Tuple of:
          - list[RetrievedChunk]: top n_results chunks, stage="stage3"
          - str: pattern name used ("router"|"iterative"|"decomposition"|"standard")
    """
    from src.retriever import stage1_coarse_search, stage2_fine_rerank
    from src.reranker import cross_encoder_rerank

    if n_results is None:
        n_results = config.STAGE3_TOP_K

    query        = analysed_query.translated_query or analysed_query.original_query
    query_type   = analysed_query.query_type
    pattern      = _QUERY_TYPE_TO_PATTERN.get(query_type, "standard")

    logger.info(
        "orchestrator: query_type=%s → pattern=%s  query=%r",
        query_type, pattern, query[:60],
    )

    # ------------------------------------------------------------------
    # Dispatch to the selected pattern
    # ------------------------------------------------------------------
    agent_iters = 1   # default for non-iterative patterns

    if pattern == "router":
        chunks = router_agent(query, analysed_query, n_results=n_results)

    elif pattern == "iterative":
        chunks, agent_iters = iterative_retrieval_agent(
            query, analysed_query,
            max_iterations=3,
            coverage_threshold=0.70,
            n_results=n_results,
        )

    elif pattern == "decomposition":
        chunks = decomposition_agent(
            query, analysed_query,
            n_subquestions=3,
            n_results=n_results,
        )

    else:
        # Standard 3-stage retrieval — no agentic wrapping
        stage1 = stage1_coarse_search(
            query,
            analysed_query=analysed_query,
            n_results=config.STAGE1_TOP_K,
        )
        stage2 = stage2_fine_rerank(query, stage1, n_results=config.STAGE2_TOP_K)
        chunks = cross_encoder_rerank(query, stage2, top_k=n_results)

    logger.info(
        "orchestrator: done — pattern=%s  chunks=%d  top_score=%.4f  iters=%d",
        pattern,
        len(chunks),
        chunks[0].score if chunks else 0.0,
        agent_iters,
    )
    return chunks, pattern, agent_iters
