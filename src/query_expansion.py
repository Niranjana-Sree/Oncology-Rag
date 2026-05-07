"""
src/query_expansion.py — Query translation and expansion for MedRAG.

Features implemented across 3.5–3.9:
  3.5  translate_to_english()   — deep-translator + Claude for Hinglish
  3.6  expand_synonyms()        — medical synonym dictionary expansion
  3.7  hyde_expansion()         — hypothetical document embedding via Claude
  3.8  multi_query_generation() — 3 rephrased versions via Claude
  3.9  full_query_expansion()   — wires all expansion steps together
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("medrag.query_expansion")


# ---------------------------------------------------------------------------
# Translation  (Feature 3.5)
# ---------------------------------------------------------------------------

def _translate_with_deep_translator(text: str, source_lang: str) -> str:
    """
    Translate text to English using deep-translator (GoogleTranslator).

    Args:
        text:        Text to translate.
        source_lang: ISO 639-1 language code ("hi", "ta", "te").

    Returns:
        Translated English string, or original text on failure.
    """
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(
            source=source_lang, target="en"
        ).translate(text)
        if translated and translated.strip():
            logger.debug(
                "deep-translator: %s → en: %r → %r", source_lang, text[:60], translated[:60]
            )
            return translated.strip()
        return text
    except Exception as exc:
        logger.warning(
            "deep-translator failed for lang=%s (%s) — returning original", source_lang, exc
        )
        return text


def _translate_hinglish_with_claude(text: str) -> str:
    """
    Translate Hinglish (romanised Hindi mixed with English) to English
    using Claude API.

    Hinglish requires semantic understanding that rule-based translators
    cannot provide — Claude interprets the mixed-language context correctly.

    Returns original text if Claude API is unavailable.
    """
    try:
        import anthropic
        import config

        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        prompt = (
            "Translate the following Hinglish text (romanised Hindi mixed with English) "
            "into clear, natural English. "
            "Preserve medical terminology exactly as-is. "
            "Return ONLY the English translation, nothing else.\n\n"
            f"Hinglish: {text}"
        )
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        translated = response.content[0].text.strip()
        logger.debug("Claude Hinglish → en: %r → %r", text[:60], translated[:60])
        return translated

    except Exception as exc:
        logger.warning(
            "Claude Hinglish translation failed (%s) — using text as-is", exc
        )
        # Best-effort: strip obvious Hindi words, return remainder
        import re
        # Remove common Hinglish filler words
        hinglish_fillers = r"\b(kya|hai|hain|ka|ki|ke|ko|se|mein|par|aur|nahi|nahin|toh|yeh|woh|kuch|bahut|iska|uska)\b"
        cleaned = re.sub(hinglish_fillers, "", text, flags=re.IGNORECASE)
        cleaned = " ".join(cleaned.split())
        return cleaned if cleaned.strip() else text


def translate_to_english(
    text: str,
    detected_language: str,
) -> str:
    """
    Translate a query to English based on its detected language.

    Language routing:
      en       → return as-is (no translation needed)
      hi       → deep-translator (Google Translate, Hindi)
      ta       → deep-translator (Google Translate, Tamil)
      te       → deep-translator (Google Translate, Telugu)
      hinglish → Claude API (context-aware mixed-language translation)
      unknown  → return as-is

    Args:
        text:              Raw query string.
        detected_language: Language code from detect_language().

    Returns:
        English translation of the query.
    """
    from src.models import Language

    lang = detected_language.lower().strip()

    if lang in (Language.ENGLISH.value, Language.UNKNOWN.value, ""):
        logger.debug("translate_to_english: language=%s — no translation needed", lang)
        return text

    if lang == Language.HINDI.value:
        logger.info("translate_to_english: Hindi → deep-translator")
        return _translate_with_deep_translator(text, source_lang="hi")

    if lang == Language.TAMIL.value:
        logger.info("translate_to_english: Tamil → deep-translator")
        return _translate_with_deep_translator(text, source_lang="ta")

    if lang == Language.TELUGU.value:
        logger.info("translate_to_english: Telugu → deep-translator")
        return _translate_with_deep_translator(text, source_lang="te")

    if lang == Language.HINGLISH.value:
        logger.info("translate_to_english: Hinglish → Claude API")
        return _translate_hinglish_with_claude(text)

    logger.warning("translate_to_english: unknown language '%s' — returning as-is", lang)
    return text


# ---------------------------------------------------------------------------
# Synonym expansion  (Feature 3.6)
# ---------------------------------------------------------------------------

def expand_synonyms(
    query: str,
    entity_texts: list[str],
) -> list[str]:
    """
    Expand a query using the oncology medical synonym dictionary.

    Two expansion strategies are combined:
      1. Entity-based: for each NER entity extracted by LAQA, look up all
         synonyms in the medical_synonyms dictionary and add them as
         alternative search terms.
      2. Token-based: scan individual tokens in the query for dictionary
         matches (catches terms the NER may have missed).

    The result is a list of unique expanded terms — NOT full query strings.
    These terms are added to the query vector space by query_expansion.py's
    full_query_expansion() to broaden retrieval coverage.

    Args:
        query:        English query string (translated if originally non-English).
        entity_texts: List of entity text strings from extract_medical_entities().

    Returns:
        Deduplicated list of synonym terms (does not include the original
        query tokens — those are already in the query embedding).
        Returns empty list if no synonyms found.
    """
    from src.medical_synonyms import get_synonyms, get_canonical, REVERSE_MAP

    import re

    seen:   set[str]  = set()
    result: set[str]  = set()

    def _add_synonyms(term: str) -> None:
        """Look up synonyms for a term and add to result set."""
        syns = get_synonyms(term)
        for syn in syns:
            lower = syn.lower()
            if lower not in seen and lower != term.lower():
                seen.add(lower)
                result.add(syn)

    # Strategy 1: expand each NER entity
    for entity in entity_texts:
        entity_lower = entity.strip().lower()
        if entity_lower:
            _add_synonyms(entity_lower)

    # Strategy 2: scan query tokens for dictionary matches
    # Split on whitespace and punctuation, try 1-gram, 2-gram, 3-gram windows
    # Minimum 4 chars for single tokens to avoid abbreviation false-positives
    # (e.g. "all" matching "ALL" = acute lymphoblastic leukemia)
    tokens = re.findall(r"[a-zA-Z]+", query.lower())
    _COMMON_WORDS = {
        "what", "which", "how", "why", "when", "where", "who",
        "this", "that", "these", "those", "with", "from", "into",
        "about", "after", "before", "their", "there", "have", "does",
        "will", "been", "were", "they", "them", "than", "then",
        "also", "only", "some", "such", "more", "most", "very",
        "each", "both", "many", "much", "used", "using", "show",
    }

    for i, token in enumerate(tokens):
        # 1-gram: require >= 4 chars and not a common English word
        if len(token) >= 4 and token not in _COMMON_WORDS:
            if token in REVERSE_MAP:
                _add_synonyms(token)
        # 2-gram
        if i + 1 < len(tokens):
            bigram = f"{token} {tokens[i+1]}"
            if bigram in REVERSE_MAP:
                _add_synonyms(bigram)
        # 3-gram
        if i + 2 < len(tokens):
            trigram = f"{token} {tokens[i+1]} {tokens[i+2]}"
            if trigram in REVERSE_MAP:
                _add_synonyms(trigram)

    expanded = sorted(result)

    logger.debug(
        "expand_synonyms: query=%r → %d synonym terms found",
        query[:60], len(expanded),
    )
    if expanded:
        logger.debug("  synonyms: %s", expanded[:8])

    return expanded


# ---------------------------------------------------------------------------
# HyDE expansion  (Feature 3.7)
# ---------------------------------------------------------------------------

def hyde_expansion(
    query: str,
    query_type: str = "qa",
) -> str:
    """
    Hypothetical Document Embedding (HyDE) expansion.

    Asks Claude to generate a short hypothetical answer to the query as if
    it were written in an oncology textbook or clinical guideline. This
    hypothetical answer is then embedded and used as an additional query
    vector alongside the original query.

    Why HyDE works:
        The embedding of a hypothetical answer is closer in vector space to
        real relevant document chunks than the embedding of the short query
        alone. A question like "What is the dose of cisplatin?" embeds
        differently from a passage that says "Cisplatin is administered at
        75-100 mg/m2 IV every 3 weeks..." — HyDE bridges this gap.

    Query-type-aware prompts:
        - qa / lfqa        → generate a factual answer paragraph
        - mcqa             → generate an explanation of the correct option
        - fact_verification→ generate a verification statement
        - jeopardy         → generate the question being implied
        - fill_blank       → generate the completed sentence

    Falls back to the original query string if Claude API is unavailable.

    Args:
        query:      English query string.
        query_type: QueryType value from laqa_pipeline().

    Returns:
        Hypothetical answer string (100-200 words), or original query on failure.
    """
    # Query-type-specific prompt instructions
    _PROMPTS = {
        "qa": (
            "Write a concise factual answer (3-5 sentences) to this oncology question "
            "as it would appear in a medical textbook or clinical guideline. "
            "Include specific clinical details, dosages, or evidence where relevant."
        ),
        "lfqa": (
            "Write a detailed explanation (5-8 sentences) answering this oncology question "
            "as it would appear in a comprehensive medical reference. "
            "Cover mechanism, clinical application, and key evidence."
        ),
        "mcqa": (
            "Write a brief explanation (3-4 sentences) of the correct answer to this "
            "multiple choice oncology question, as would appear in a study guide."
        ),
        "fact_verification": (
            "Write a factual statement (2-4 sentences) that verifies or refutes this "
            "oncology claim, citing the clinical evidence or guideline basis."
        ),
        "jeopardy": (
            "Write the question that this oncology statement is the answer to, "
            "then provide a brief 2-3 sentence explanation."
        ),
        "fill_blank": (
            "Complete the blank in this oncology sentence and explain the answer "
            "in 2-3 sentences with clinical context."
        ),
    }

    instruction = _PROMPTS.get(query_type, _PROMPTS["qa"])

    try:
        import anthropic
        import config

        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

        prompt = (
            f"{instruction}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        hyde_text = response.content[0].text.strip()
        logger.info(
            "hyde_expansion: generated %d chars for query_type=%s",
            len(hyde_text), query_type,
        )
        logger.debug("HyDE text: %r", hyde_text[:120])
        return hyde_text

    except Exception as exc:
        logger.warning(
            "hyde_expansion: Claude API failed (%s) — returning original query",
            exc,
        )
        return query


# ---------------------------------------------------------------------------
# Multi-query generation  (Feature 3.8)
# ---------------------------------------------------------------------------

def multi_query_generation(
    query: str,
    n: int = 3,
) -> list[str]:
    """
    Generate N rephrased versions of the query using Claude.

    Each rephrased version captures a different semantic angle of the same
    medical question — different vocabulary, different framing — so that
    when all versions are embedded and searched, the union of results has
    higher recall than a single query alone.

    Example:
        Original: "What is the dose of cisplatin for head and neck cancer?"
        Version 1: "Cisplatin dosage in HNSCC treatment protocols"
        Version 2: "How much cisplatin is administered for squamous cell
                    carcinoma of the head and neck?"
        Version 3: "Standard platinum-based chemotherapy dosing for oral
                    cavity and oropharyngeal malignancies"

    Falls back to [query] (single-item list) if Claude API is unavailable,
    so the pipeline continues with at least the original query.

    Args:
        query: English query string.
        n:     Number of rephrased versions to generate (default 3).

    Returns:
        List of n rephrased query strings, or [query] on API failure.
    """
    try:
        import anthropic
        import config

        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

        prompt = (
            f"You are an expert oncology medical information retrieval assistant.\n"
            f"Generate exactly {n} different rephrased versions of the following "
            f"medical query. Each version should:\n"
            f"  - Preserve the original medical intent exactly\n"
            f"  - Use different vocabulary, synonyms, or clinical terminology\n"
            f"  - Be suitable for searching an oncology document corpus\n"
            f"  - Be a complete, standalone question or phrase\n\n"
            f"Return ONLY the {n} rephrased versions, one per line, numbered 1. 2. 3.\n"
            f"Do not include any explanation or the original query.\n\n"
            f"Original query: {query}"
        )

        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()

        # Parse numbered list — extract text after "1. ", "2. ", "3. "
        import re
        versions = []
        for line in raw.splitlines():
            line = line.strip()
            # Match "1. text", "1) text", "- text"
            match = re.match(r"^[\d]+[.)]\s+(.+)$", line)
            if match:
                versions.append(match.group(1).strip())
            elif line.startswith("- ") and len(line) > 3:
                versions.append(line[2:].strip())

        # Keep only non-empty versions up to n
        versions = [v for v in versions if v and len(v) > 5][:n]

        if not versions:
            logger.warning(
                "multi_query_generation: could not parse Claude response — "
                "returning original query"
            )
            return [query]

        logger.info(
            "multi_query_generation: generated %d versions for query=%r",
            len(versions), query[:60],
        )
        for i, v in enumerate(versions, 1):
            logger.debug("  v%d: %r", i, v)

        return versions

    except Exception as exc:
        logger.warning(
            "multi_query_generation: Claude API failed (%s) — returning [original query]",
            exc,
        )
        return [query]


# ---------------------------------------------------------------------------
# Full query expansion pipeline  (Feature 3.9)
# ---------------------------------------------------------------------------

def full_query_expansion(
    analysed_query: "AnalysedQuery",
    use_hyde: bool = True,
    use_multi_query: bool = True,
    use_synonyms: bool = True,
) -> "AnalysedQuery":
    """
    Run all query expansion steps and return an enriched AnalysedQuery.

    Expansion steps (all optional, all gracefully degrade):
      1. translate_to_english()    — convert non-English query to English
      2. expand_synonyms()         — add oncology synonym terms
      3. hyde_expansion()          — generate hypothetical answer (Claude)
      4. multi_query_generation()  — generate 3 rephrased versions (Claude)

    The returned AnalysedQuery has:
      - translated_query  : English version of the original query
      - entity_texts      : NER entities (re-extracted on translated text if needed)
      - expanded_queries  : flat list of all additional search strings:
                            [synonym_terms..., hyde_text, v1, v2, v3]

    Args:
        analysed_query:   Output of laqa_pipeline().
        use_hyde:         Whether to run HyDE expansion (needs Claude API).
        use_multi_query:  Whether to run multi-query generation (needs Claude API).
        use_synonyms:     Whether to run synonym expansion (always fast).

    Returns:
        Enriched AnalysedQuery with translated_query and expanded_queries filled.
    """
    from src.laqa import extract_medical_entities
    from src.models import Language
    import copy

    result = copy.deepcopy(analysed_query)
    query  = result.original_query
    lang   = result.detected_language

    logger.info(
        "full_query_expansion: query=%r lang=%s type=%s",
        query[:60], lang, result.query_type,
    )

    # ------------------------------------------------------------------
    # Step 1: Translate to English
    # ------------------------------------------------------------------
    translated = translate_to_english(query, lang)
    result.translated_query = translated
    logger.info("  [1/4] Translation: %r", translated[:80])

    # Re-run NER on translated text for non-English queries
    if lang not in (Language.ENGLISH.value, Language.HINGLISH.value):
        entities = extract_medical_entities(translated)
        result.medical_entities = entities
        result.entity_texts     = [e.text for e in entities]
        logger.info(
            "  [1/4] Re-extracted %d entities from translated text",
            len(entities),
        )

    # ------------------------------------------------------------------
    # Step 2: Synonym expansion
    # ------------------------------------------------------------------
    expanded: list[str] = []

    if use_synonyms:
        synonyms = expand_synonyms(translated, result.entity_texts)
        expanded.extend(synonyms)
        logger.info("  [2/4] Synonyms: %d terms", len(synonyms))
    else:
        logger.info("  [2/4] Synonyms: skipped")

    # ------------------------------------------------------------------
    # Step 3: HyDE expansion
    # ------------------------------------------------------------------
    if use_hyde:
        hyde_text = hyde_expansion(translated, result.query_type)
        # Only add if HyDE produced something different from the query
        if hyde_text and hyde_text.strip() != translated.strip():
            expanded.append(hyde_text)
            logger.info("  [3/4] HyDE: %d chars", len(hyde_text))
        else:
            logger.info("  [3/4] HyDE: fallback (same as query)")
    else:
        logger.info("  [3/4] HyDE: skipped")

    # ------------------------------------------------------------------
    # Step 4: Multi-query generation
    # ------------------------------------------------------------------
    if use_multi_query:
        versions = multi_query_generation(translated, n=3)
        # Only add versions that differ from the original
        for v in versions:
            if v.strip() != translated.strip():
                expanded.append(v)
        logger.info("  [4/4] Multi-query: %d versions added", len(versions))
    else:
        logger.info("  [4/4] Multi-query: skipped")

    result.expanded_queries = expanded
    logger.info(
        "full_query_expansion done: %d total expansion strings",
        len(expanded),
    )
    return result
