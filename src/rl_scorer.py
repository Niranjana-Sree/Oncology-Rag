"""
src/rl_scorer.py — RLAIF scoring and GRPO reward computation for MedRAG.

Features implemented across 5.5–5.6:
  5.5  ai_score_answer()   — Claude scores answer on 5 clinical dimensions
  5.6  compute_reward()    — weighted composite score + SQLite storage

The scores drive GRPO (Group Relative Policy Optimization) training signals
for periodic fine-tuning of the generator model.

Scoring dimensions (weights from config.RLAIF_WEIGHTS):
  medical_accuracy  0.35 — factual correctness of clinical content
  faithfulness      0.25 — grounded in retrieved context, no hallucination
  completeness      0.20 — all aspects of the query addressed
  safety            0.15 — no harmful advice, appropriate medical disclaimers
  clarity           0.05 — clear, well-structured, readable answer

Public API
----------
ai_score_answer(query, answer, chunks, query_type) → RLScore
compute_reward(query, answer, chunks, query_type)  → RLScore (stored in SQLite)
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

import config
from src.models import RLScore, RetrievedChunk

logger = logging.getLogger("medrag.rl_scorer")

# ---------------------------------------------------------------------------
# Scoring prompt
# ---------------------------------------------------------------------------

_SCORING_PROMPT = """You are an expert oncology medical answer evaluator.

Score the following medical Q&A on exactly 5 dimensions.
Return a JSON object only — no explanation outside the JSON.

Query: {query}
Query Type: {query_type}

Retrieved Context (what the answer should be grounded in):
{context}

Generated Answer:
{answer}

Score each dimension from 0.0 to 1.0:

{{
  "medical_accuracy": <float>,
  "faithfulness": <float>,
  "completeness": <float>,
  "safety": <float>,
  "clarity": <float>,
  "feedback": "<one sentence explaining the main strength or weakness>"
}}

Scoring rubric:
- medical_accuracy: Is the clinical content factually correct? (0=wrong, 1=fully accurate)
- faithfulness: Is the answer grounded in the context? (0=hallucinated, 1=fully grounded)
- completeness: Does it address all aspects of the query? (0=missing key info, 1=complete)
- safety: No harmful advice, appropriate caveats for medical info? (0=dangerous, 1=safe)
- clarity: Well-structured, clear, readable? (0=confusing, 1=excellent)

Return ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Feature 5.5 — ai_score_answer()
# ---------------------------------------------------------------------------

def ai_score_answer(
    query: str,
    answer: str,
    chunks: list[RetrievedChunk],
    query_type: str = "qa",
) -> RLScore:
    """
    Score a generated answer using Claude as an AI evaluator (RLAIF).

    Sends the query, retrieved context, and generated answer to Claude
    and asks it to score on 5 clinical quality dimensions.

    Falls back to a default neutral score (0.5 on all dimensions) if
    the Claude API is unavailable or returns malformed JSON.

    Args:
        query:      The original user query (English).
        answer:     The generated answer text to score.
        chunks:     Retrieved chunks used to generate the answer.
        query_type: QueryType value — used to set scoring expectations.

    Returns:
        RLScore with per-dimension scores, composite score, and feedback.
    """
    # Format context for scoring (top 3 chunks)
    context_parts = []
    for i, c in enumerate(chunks[:3], 1):
        context_parts.append(f"[{i}] {c.text[:300]}")
    context = "\n\n".join(context_parts) if context_parts else "No context provided."

    prompt = _SCORING_PROMPT.format(
        query=query,
        query_type=query_type,
        context=context,
        answer=answer[:1500],  # cap answer length for scoring
    )

    try:
        import anthropic
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()

        # Parse JSON from response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {raw[:100]}")

        scores = json.loads(json_match.group())

        # Extract and clamp each dimension to [0, 1]
        def _clamp(val, default=0.5):
            try:
                return max(0.0, min(1.0, float(val)))
            except (TypeError, ValueError):
                return default

        medical_accuracy = _clamp(scores.get("medical_accuracy"))
        faithfulness     = _clamp(scores.get("faithfulness"))
        completeness     = _clamp(scores.get("completeness"))
        safety           = _clamp(scores.get("safety"))
        clarity          = _clamp(scores.get("clarity"))
        feedback         = str(scores.get("feedback", ""))[:500]

        # Compute weighted composite score
        weights = config.RLAIF_WEIGHTS
        composite = (
            medical_accuracy * weights["medical_accuracy"] +
            faithfulness     * weights["faithfulness"]     +
            completeness     * weights["completeness"]     +
            safety           * weights["safety"]           +
            clarity          * weights["clarity"]
        )

        rl_score = RLScore(
            medical_accuracy=medical_accuracy,
            faithfulness=faithfulness,
            completeness=completeness,
            safety=safety,
            clarity=clarity,
            composite=round(composite, 4),
            raw_feedback=feedback,
        )

        logger.info(
            "ai_score_answer: composite=%.3f  [acc=%.2f faith=%.2f comp=%.2f "
            "safe=%.2f clar=%.2f]",
            composite, medical_accuracy, faithfulness,
            completeness, safety, clarity,
        )
        return rl_score

    except Exception as exc:
        logger.warning(
            "ai_score_answer: Claude scoring failed (%s) — returning neutral score", exc
        )
        # Neutral fallback score
        weights = config.RLAIF_WEIGHTS
        neutral = 0.5
        composite = sum(neutral * w for w in weights.values())
        return RLScore(
            medical_accuracy=neutral,
            faithfulness=neutral,
            completeness=neutral,
            safety=neutral,
            clarity=neutral,
            composite=round(composite, 4),
            raw_feedback=f"Scoring unavailable: {exc}",
        )


# ---------------------------------------------------------------------------
# Feature 5.6 — compute_reward() + SQLite storage
# ---------------------------------------------------------------------------

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS rewards (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         REAL    NOT NULL,
    query             TEXT    NOT NULL,
    query_type        TEXT    NOT NULL,
    answer_preview    TEXT    NOT NULL,
    model_used        TEXT    NOT NULL DEFAULT 'unknown',
    medical_accuracy  REAL    NOT NULL,
    faithfulness      REAL    NOT NULL,
    completeness      REAL    NOT NULL,
    safety            REAL    NOT NULL,
    clarity           REAL    NOT NULL,
    composite         REAL    NOT NULL,
    feedback          TEXT,
    -- GRPO fields
    group_id          TEXT,     -- groups answers for the same query
    advantage         REAL      -- GRPO advantage = score - group_mean
);
CREATE INDEX IF NOT EXISTS idx_rewards_timestamp  ON rewards(timestamp);
CREATE INDEX IF NOT EXISTS idx_rewards_query_type ON rewards(query_type);
CREATE INDEX IF NOT EXISTS idx_rewards_composite  ON rewards(composite);
CREATE INDEX IF NOT EXISTS idx_rewards_group_id   ON rewards(group_id);
"""


def _get_db_connection() -> sqlite3.Connection:
    """Open and return a SQLite connection to the rewards database."""
    db_path = config.REWARDS_DB_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_DB_SCHEMA)
    conn.commit()
    return conn


def compute_reward(
    query: str,
    answer: str,
    chunks: list[RetrievedChunk],
    query_type: str = "qa",
    model_used: str = "unknown",
    group_id: Optional[str] = None,
) -> RLScore:
    """
    Score the answer with RLAIF and persist the reward to SQLite.

    Workflow:
      1. Call ai_score_answer() to get per-dimension scores from Claude.
      2. Compute weighted composite reward signal.
      3. Compute GRPO advantage if group_id is provided:
           advantage = composite - mean(composite for same group_id)
      4. Store everything in results/rewards.db.
      5. Return the RLScore.

    GRPO (Group Relative Policy Optimization):
      Multiple answers are generated for the same query (a "group").
      Each answer's advantage = its score - the group mean score.
      Positive advantage → answer is above average → reinforce.
      Negative advantage → answer is below average → suppress.

    Args:
        query:      Original user query (English).
        answer:     Generated answer text.
        chunks:     Retrieved context chunks.
        query_type: QueryType value.
        model_used: Which model generated the answer ("medgemma-4b" or "claude-fallback").
        group_id:   Optional group identifier for GRPO (e.g. query hash).
                    If provided, advantage is computed relative to the group.

    Returns:
        RLScore with all dimensions, composite, and GRPO advantage stored.
    """
    # Step 1: Score with Claude RLAIF
    rl_score = ai_score_answer(query, answer, chunks, query_type=query_type)

    # Step 2: Compute GRPO advantage if group_id provided
    advantage: Optional[float] = None
    if group_id:
        try:
            conn = _get_db_connection()
            row = conn.execute(
                "SELECT AVG(composite) as mean_score FROM rewards WHERE group_id = ?",
                (group_id,),
            ).fetchone()
            conn.close()

            if row and row["mean_score"] is not None:
                group_mean = float(row["mean_score"])
                advantage  = round(rl_score.composite - group_mean, 4)
                logger.debug(
                    "GRPO: group_id=%s  composite=%.3f  mean=%.3f  advantage=%.3f",
                    group_id, rl_score.composite, group_mean, advantage,
                )
        except Exception as exc:
            logger.warning("compute_reward: GRPO advantage failed (%s)", exc)

    # Step 3: Store in SQLite
    try:
        conn = _get_db_connection()
        conn.execute(
            """INSERT INTO rewards
               (timestamp, query, query_type, answer_preview, model_used,
                medical_accuracy, faithfulness, completeness, safety, clarity,
                composite, feedback, group_id, advantage)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                query[:500],
                query_type,
                answer[:200],
                model_used,
                rl_score.medical_accuracy,
                rl_score.faithfulness,
                rl_score.completeness,
                rl_score.safety,
                rl_score.clarity,
                rl_score.composite,
                rl_score.raw_feedback,
                group_id,
                advantage,
            ),
        )
        conn.commit()
        conn.close()
        logger.info(
            "compute_reward: stored reward composite=%.3f  group_id=%s  advantage=%s",
            rl_score.composite, group_id, f"{advantage:.3f}" if advantage else "N/A",
        )
    except Exception as exc:
        logger.error("compute_reward: SQLite storage failed (%s)", exc)

    return rl_score


def get_reward_stats() -> dict:
    """
    Return aggregate statistics from the rewards database.
    Useful for monitoring training signal quality.
    """
    try:
        conn = _get_db_connection()
        row = conn.execute("""
            SELECT
                COUNT(*)           AS total,
                AVG(composite)     AS avg_composite,
                MIN(composite)     AS min_composite,
                MAX(composite)     AS max_composite,
                AVG(medical_accuracy) AS avg_accuracy,
                AVG(faithfulness)     AS avg_faithfulness,
                AVG(safety)           AS avg_safety
            FROM rewards
        """).fetchone()

        type_rows = conn.execute("""
            SELECT query_type, COUNT(*) as count, AVG(composite) as avg_score
            FROM rewards GROUP BY query_type
        """).fetchall()
        conn.close()

        return {
            "total_records":   row["total"],
            "avg_composite":   round(row["avg_composite"] or 0, 4),
            "min_composite":   round(row["min_composite"] or 0, 4),
            "max_composite":   round(row["max_composite"] or 0, 4),
            "avg_accuracy":    round(row["avg_accuracy"] or 0, 4),
            "avg_faithfulness":round(row["avg_faithfulness"] or 0, 4),
            "avg_safety":      round(row["avg_safety"] or 0, 4),
            "by_query_type":   {
                r["query_type"]: {
                    "count": r["count"],
                    "avg_score": round(r["avg_score"] or 0, 4),
                }
                for r in type_rows
            },
        }
    except Exception as exc:
        logger.error("get_reward_stats: failed (%s)", exc)
        return {"error": str(exc)}
