"""
src/prompt_builder.py — Query-type-specific prompt construction for MedRAG.

Builds structured system + user prompts for all 6 oncology query types.
Each prompt template is designed to:
  - Ground the answer in retrieved context chunks
  - Enforce appropriate answer format for the query type
  - Include safety instructions for medical content
  - Cite source documents

Public API
----------
build_prompt(query, chunks, query_type, language, audience) → dict with system + user keys
format_context(chunks)                                      → formatted context string
"""

from __future__ import annotations

import logging
from typing import Optional

from src.models import RetrievedChunk

logger = logging.getLogger("medrag.prompt_builder")

# ---------------------------------------------------------------------------
# System prompt — shared base across all query types
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are MedRAG, an expert oncology medical question answering assistant.

You answer questions based ONLY on the provided context from oncology textbooks,
clinical guidelines, and research papers.

Rules you must follow:
1. Base your answer strictly on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information, say so clearly.
3. Always maintain medical accuracy — do not speculate or hallucinate drug names,
   dosages, or clinical recommendations.
4. For dosage or treatment questions, note that clinical decisions must be made
   by qualified healthcare professionals.
5. Cite the source document when referencing specific facts.
6. Be concise and clinically precise."""

# ---------------------------------------------------------------------------
# Audience-specific system prompt additions
# ---------------------------------------------------------------------------

_PATIENT_SYSTEM_ADDITION = """

You are a compassionate medical assistant explaining to a patient who has no medical background.

STRICT RULES for patient mode:
- Use simple everyday words only
- Never use medical jargon without explaining it
- If you must use a medical term, immediately explain it:
  Example: 'cisplatin (a strong cancer-fighting medicine)'
- Use short sentences — maximum 20 words per sentence
- Show empathy and warmth in your tone
- Acknowledge that cancer treatment is difficult
- Focus on what this means for the patient's daily life
- Always end with this exact line:
  'Please talk to your doctor or nurse before making any decisions about your treatment.'
- Do NOT use bullet points with * or dashes
- Do NOT use ## headings
- Write in warm conversational paragraphs"""

_DOCTOR_SYSTEM_ADDITION = """

You are a clinical decision support system for oncology healthcare professionals.

RULES for doctor mode:
- Use precise medical terminology
- Include specific dosages, staging criteria, grading systems where relevant
- Reference clinical guidelines (NCCN, ASCO, ESMO)
- Include mechanism of action where relevant
- Structure with clear clinical headings
- Include contraindications and monitoring parameters
- Be concise and evidence-based"""

# ---------------------------------------------------------------------------
# Query-type-specific prompt templates
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, str] = {

    "qa": """Answer the following oncology question based on the provided context.
Be direct and specific. Include relevant dosages, mechanisms, or clinical details
from the context. Keep your answer to 3-5 sentences.

Context:
{context}

Question: {query}

Answer:""",

    "mcqa": """Answer the following multiple-choice oncology question.
First identify the correct option (A, B, C, or D), then explain why it is correct
using evidence from the context. Also briefly explain why the other options are incorrect.

Context:
{context}

Question: {query}

Answer (format: "Correct answer: X) [option text]\n\nExplanation: ...")""",

    "lfqa": """Provide a comprehensive, detailed answer to the following oncology question.
Structure your answer with clear sections covering: mechanism/background, clinical
application, evidence/guidelines, and key considerations. Use information from the
provided context. Aim for 200-400 words.

Context:
{context}

Question: {query}

Detailed Answer:""",

    "jeopardy": """The following is a Jeopardy-style clue about an oncology concept, drug,
or procedure. Identify what is being described, then provide a brief explanation
using evidence from the context.

Context:
{context}

Clue: {query}

Answer (format: "What is [answer]?\n\nExplanation: ...")""",

    "fact_verification": """Evaluate whether the following oncology claim is TRUE, FALSE,
or UNCERTAIN based on the provided context from clinical guidelines and research.

Provide:
1. Verdict: TRUE / FALSE / UNCERTAIN
2. Evidence: quote or paraphrase the relevant context that supports your verdict
3. Nuance: any important caveats or conditions that affect the verdict

Context:
{context}

Claim: {query}

Verification:""",

    "fill_blank": """Complete the following oncology sentence by filling in the blank (___).
Provide the answer and a brief 2-3 sentence clinical explanation using the context.

Context:
{context}

Sentence: {query}

Completed sentence and explanation:""",
}

# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------

def format_context(
    chunks: list[RetrievedChunk],
    max_chunks: int = 5,
    max_chars_per_chunk: int = 600,
) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Each chunk is presented with its source document and relevance score.
    Chunks are truncated at max_chars_per_chunk to keep prompts manageable.

    Args:
        chunks:              List of RetrievedChunk (sorted by score descending).
        max_chunks:          Maximum number of chunks to include.
        max_chars_per_chunk: Maximum characters per chunk text.

    Returns:
        Formatted multi-line string ready for prompt injection.
    """
    if not chunks:
        return "No relevant context found."

    parts: list[str] = []
    for i, chunk in enumerate(chunks[:max_chunks], 1):
        text = chunk.text.strip()
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."

        source = chunk.source_file.split("/")[-1] if "/" in chunk.source_file else chunk.source_file
        parts.append(
            f"[{i}] Source: {source} (category: {chunk.category}, relevance: {chunk.score:.3f})\n"
            f"{text}"
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    query_type: str = "qa",
    detected_language: str = "en",
    max_chunks: int = 5,
    audience: str = "patient",
) -> dict[str, str]:
    """
    Build a complete prompt dict for the LLM generator.

    Selects the appropriate template for the query type, formats the
    retrieved context, and adds language-specific and audience-specific
    instructions.

    Args:
        query:             The user query (translated to English).
        chunks:            Top retrieved chunks from the retrieval pipeline.
        query_type:        One of the 6 QueryType values.
        detected_language: Source language of the original query.
        max_chunks:        Maximum context chunks to include (default 5).
        audience:          "patient" or "doctor" — controls tone and style.

    Returns:
        Dict with keys:
          "system": system prompt string
          "user":   user message string (context + query)
          "query_type": the query type used
          "chunk_count": number of chunks included
    """
    from src.models import QueryType

    # Normalise query type — fall back to QA if unknown
    valid_types = {qt.value for qt in QueryType} - {"unknown"}
    if query_type not in valid_types:
        logger.warning(
            "build_prompt: unknown query_type=%r — defaulting to qa", query_type
        )
        query_type = QueryType.QA.value

    # Format context block
    context = format_context(chunks, max_chunks=max_chunks)
    chunk_count = min(len(chunks), max_chunks)

    # Get query-type-specific template
    template = _TEMPLATES.get(query_type, _TEMPLATES["qa"])

    # Build user message from template
    user_message = template.format(context=context, query=query)

    # Add audience-specific instructions
    system = _SYSTEM_PROMPT
    if audience == "doctor":
        system += _DOCTOR_SYSTEM_ADDITION
    else:
        system += _PATIENT_SYSTEM_ADDITION

    # Add language note if query was translated
    if detected_language not in ("en", "unknown"):
        lang_names = {
            "hi": "Hindi", "ta": "Tamil",
            "te": "Telugu", "hinglish": "Hinglish",
        }
        lang_name = lang_names.get(detected_language, detected_language)
        system += (
            f"\n\nNote: The original query was in {lang_name}. "
            f"Please answer in {lang_name}."
        )

    logger.debug(
        "build_prompt: query_type=%s  chunks=%d  lang=%s  audience=%s  "
        "system_len=%d  user_len=%d",
        query_type, chunk_count, detected_language, audience,
        len(system), len(user_message),
    )

    return {
        "system":      system,
        "user":        user_message,
        "query_type":  query_type,
        "chunk_count": chunk_count,
    }
