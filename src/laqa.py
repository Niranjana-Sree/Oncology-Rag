"""
src/laqa.py — Language-Aware Query Analysis (LAQA) for MedRAG.

Features implemented across 3.1–3.4:
  3.1  detect_language()        — identify language + Hinglish detection
  3.2  classify_query_type()    — zero-shot classification into 6 types
  3.3  extract_medical_entities() — scispaCy NER for medical terms
  3.4  laqa_pipeline()          — full AnalysedQuery assembly
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("medrag.laqa")

# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

# Unicode block ranges for Indic scripts
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")   # Hindi
_TAMIL_RE      = re.compile(r"[஀-௿]")   # Tamil
_TELUGU_RE     = re.compile(r"[ఀ-౿]")   # Telugu

# Hinglish keywords — common Hindi words written in Latin script
# These appear in romanised Hindi mixed with English
_HINGLISH_KEYWORDS: set[str] = {
    "kya", "hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "par",
    "aur", "ya", "nahi", "nahin", "kaise", "kyun", "kyunki", "lekin",
    "agar", "toh", "phir", "yeh", "woh", "iska", "uska", "inki", "unki",
    "kuch", "sab", "bahut", "thoda", "zyada", "abhi", "pehle", "baad",
    "bimari", "dawai", "dawa", "ilaj", "doctor", "marz", "dard", "cancer",
    "treatment", "kharabi", "theek", "symptoms", "bataiye", "batao",
}

_HINGLISH_MIN_HITS = 2   # need at least 2 Hinglish words to classify as Hinglish


def _unicode_script_detect(text: str) -> Optional[str]:
    """
    Detect language from Unicode script blocks.
    Returns a Language enum value string or None if no Indic script found.
    """
    from src.models import Language

    if _DEVANAGARI_RE.search(text):
        return Language.HINDI.value
    if _TAMIL_RE.search(text):
        return Language.TAMIL.value
    if _TELUGU_RE.search(text):
        return Language.TELUGU.value
    return None


def _hinglish_score(text: str) -> int:
    """Count how many Hinglish keywords appear in the lowercased text."""
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return len(words & _HINGLISH_KEYWORDS)


def _langdetect_language(text: str) -> Optional[str]:
    """
    Use the langdetect library for statistical language detection.
    Returns a Language enum value string or None on failure.
    """
    from src.models import Language
    try:
        from langdetect import detect, LangDetectException
        code = detect(text)
        mapping = {
            "en": Language.ENGLISH.value,
            "hi": Language.HINDI.value,
            "ta": Language.TAMIL.value,
            "te": Language.TELUGU.value,
        }
        return mapping.get(code)
    except Exception:
        return None


def detect_language(text: str) -> str:
    """
    Detect the language of a query string.

    Detection pipeline (in priority order):
      1. Unicode script detection — definitive for Devanagari, Tamil, Telugu
      2. Hinglish heuristic — romanised Hindi mixed with English
         (requires >= 2 Hinglish keyword hits)
      3. langdetect statistical model — handles English and other cases
      4. Default to English if all methods fail

    Args:
        text: Raw user query string.

    Returns:
        A Language enum value string: "en", "hi", "ta", "te", "hinglish", "unknown".
    """
    from src.models import Language

    if not text or not text.strip():
        return Language.UNKNOWN.value

    # Step 1: Unicode script — most reliable for native-script Indic text
    script_lang = _unicode_script_detect(text)
    if script_lang:
        logger.debug("detect_language: unicode script → %s", script_lang)
        return script_lang

    # Step 2: Hinglish — romanised Hindi, must check BEFORE langdetect
    # because langdetect often misclassifies Hinglish as English/Afrikaans
    hinglish_hits = _hinglish_score(text)
    if hinglish_hits >= _HINGLISH_MIN_HITS:
        logger.debug(
            "detect_language: Hinglish heuristic (%d keyword hits) → hinglish",
            hinglish_hits,
        )
        return Language.HINGLISH.value

    # Step 3: Statistical detection
    stat_lang = _langdetect_language(text)
    if stat_lang:
        logger.debug("detect_language: langdetect → %s", stat_lang)
        return stat_lang

    # Step 4: Default
    logger.debug("detect_language: default → en")
    return Language.ENGLISH.value


# ---------------------------------------------------------------------------
# Query type classification  (Feature 3.2)
# ---------------------------------------------------------------------------

# Singleton cache for the zero-shot classifier pipeline
_classifier_cache: dict = {}

# Candidate labels sent to BART — descriptive phrases work better than
# short labels for zero-shot NLI classification
_QUERY_TYPE_CANDIDATES: dict[str, str] = {
    "qa":               "a direct factual question asking for a specific answer",
    "mcqa":             "a multiple choice question with labeled options to choose from",
    "lfqa":             "a request for a detailed explanation or description of a concept",
    "jeopardy":         "a statement that describes something and implies a hidden question",
    "fact_verification":"checking whether a claim or statement is true or false",
    "fill_blank":       "a sentence with a blank space or missing word to be filled in",
}


def _get_classifier(model_name: str = "facebook/bart-large-mnli"):
    """Load and cache the zero-shot classification pipeline."""
    if model_name not in _classifier_cache:
        logger.info("Loading zero-shot classifier: %s", model_name)
        from transformers import pipeline
        clf = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1,            # CPU — avoids MPS mutex issues
        )
        _classifier_cache[model_name] = clf
        logger.info("Zero-shot classifier loaded.")
    return _classifier_cache[model_name]


def classify_query_type(
    query: str,
    model_name: str = "facebook/bart-large-mnli",
) -> tuple[str, float]:
    """
    Classify a query into one of 6 oncology Q&A types using zero-shot NLI.

    Uses facebook/bart-large-mnli with descriptive candidate labels.
    The model scores each label as a natural language inference hypothesis
    against the query as premise — no fine-tuning required.

    Query types:
      qa               — "What is the half-life of cisplatin?"
      mcqa             — "Which drug is first-line? A) X B) Y C) Z D) W"
      lfqa             — "Explain the mechanism of action of pembrolizumab."
      jeopardy         — "This PD-1 inhibitor is used in HNSCC."
      fact_verification— "Claim: Cisplatin is effective against MRSA."
      fill_blank       — "The antidote for heparin overdose is ___."

    Args:
        query:      English query string (translate first if needed).
        model_name: HuggingFace model ID for zero-shot classification.

    Returns:
        Tuple of (query_type_str, confidence_float) where query_type_str
        is one of the 6 QueryType enum values and confidence is in [0, 1].
    """
    from src.models import QueryType

    if not query or not query.strip():
        return QueryType.QA.value, 0.0

    # Rule-based pre-checks for high-confidence patterns — faster than model
    q = query.strip()

    # MCQA: contains option markers A) B) C) D) or (A) (B) (C) (D)
    if re.search(r"\b[A-D][).]\s+\w", q):
        logger.debug("classify_query_type: rule → mcqa (option markers detected)")
        return QueryType.MCQA.value, 0.99

    # Fill-in-blank: contains ___ or [blank] or (blank) — MUST come before QA check
    if re.search(r"___+|\[blank\]|\(blank\)", q, re.IGNORECASE):
        logger.debug("classify_query_type: rule → fill_blank (blank marker detected)")
        return QueryType.FILL_BLANK.value, 0.99

    # Fact verification: starts with "Claim:" or "True or False" or "Is it true"
    if re.search(
        r"^(claim\s*:|true or false|is it true|verify|fact\s*check)", q, re.IGNORECASE
    ):
        logger.debug("classify_query_type: rule → fact_verification")
        return QueryType.FACT_VERIFICATION.value, 0.99

    # Hinglish question patterns — "kya hai", "kaise", "kyun", "batao"
    if re.search(r"\b(kya|kaise|kyun|kyunki|batao|bataiye)\b", q, re.IGNORECASE):
        logger.debug("classify_query_type: rule → qa (Hinglish question word)")
        return QueryType.QA.value, 0.88

    # QA: starts with a question word — high confidence direct question
    if re.search(
        r"^(what|which|who|when|where|how|why|is|are|does|do|can|could|should|will)\b",
        q, re.IGNORECASE
    ):
        # But check for LFQA signals — "explain", "describe", "discuss"
        if re.search(
            r"^(explain|describe|discuss|elaborate|what is the mechanism|"
            r"what are the steps|how does .+ work|why does)", q, re.IGNORECASE
        ):
            logger.debug("classify_query_type: rule → lfqa (explanation request)")
            return QueryType.LFQA.value, 0.92
        logger.debug("classify_query_type: rule → qa (question word detected)")
        return QueryType.QA.value, 0.92

    # Jeopardy: starts with "This" or "These" describing a medical entity
    if re.search(r"^(this|these)\b", q, re.IGNORECASE):
        logger.debug("classify_query_type: rule → jeopardy (This/These opening)")
        return QueryType.JEOPARDY.value, 0.90

    # LFQA: imperative explanation requests
    if re.search(
        r"^(explain|describe|discuss|elaborate|outline|summarize|compare|"
        r"differentiate|define|list the)", q, re.IGNORECASE
    ):
        logger.debug("classify_query_type: rule → lfqa (imperative)")
        return QueryType.LFQA.value, 0.92

    # Zero-shot classification for remaining types
    try:
        clf = _get_classifier(model_name)
        candidate_labels = list(_QUERY_TYPE_CANDIDATES.values())
        result = clf(q, candidate_labels=candidate_labels, multi_label=False)

        # Map descriptive label back to query type key
        label_to_type = {v: k for k, v in _QUERY_TYPE_CANDIDATES.items()}
        top_label  = result["labels"][0]
        confidence = float(result["scores"][0])
        query_type = label_to_type.get(top_label, QueryType.QA.value)

        logger.debug(
            "classify_query_type: BART → %s (%.3f)  [label=%s]",
            query_type, confidence, top_label,
        )
        return query_type, confidence

    except Exception as exc:
        logger.warning(
            "classify_query_type: BART classification failed (%s) — defaulting to qa",
            exc,
        )
        return QueryType.QA.value, 0.0


# ---------------------------------------------------------------------------
# Medical NER  (Feature 3.3)
# ---------------------------------------------------------------------------

# Singleton cache for the scispaCy model
_spacy_cache: dict = {}

# Oncology-relevant entity labels from en_core_sci_md
# en_core_sci_md uses a single "ENTITY" label for all biomedical entities.
# We enrich the label using the medical synonym dictionary.
_SPACY_MODEL = "en_core_sci_md"

# Minimum entity character length — filters out noise like "a", "of"
_MIN_ENTITY_LEN = 3

# Stopwords that scispaCy sometimes tags as entities
_ENTITY_STOPWORDS: set[str] = {
    "the", "and", "or", "of", "in", "to", "a", "an", "is", "are",
    "was", "were", "be", "been", "that", "this", "these", "those",
    "with", "for", "on", "at", "by", "from", "as", "it", "its",
    "head", "neck", "body", "cell", "cells", "level", "levels",
    "rate", "type", "types", "form", "forms", "case", "cases",
}


def _get_spacy_model():
    """Load and cache the scispaCy model."""
    if _SPACY_MODEL not in _spacy_cache:
        logger.info("Loading scispaCy model: %s", _SPACY_MODEL)
        import spacy
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nlp = spacy.load(_SPACY_MODEL)
        _spacy_cache[_SPACY_MODEL] = nlp
        logger.info("scispaCy model loaded.")
    return _spacy_cache[_SPACY_MODEL]


def _enrich_entity_label(text: str) -> str:
    """
    Map a scispaCy entity to a more specific oncology label using the
    medical synonym dictionary.

    Returns one of: DRUG, DISEASE, TREATMENT, CLINICAL_TERM, ENTITY
    """
    from src.medical_synonyms import get_canonical

    lower = text.lower().strip()
    canonical = get_canonical(lower)

    if canonical is None:
        return "ENTITY"

    # Map canonical term categories to richer labels
    drug_canonicals = {
        "cisplatin", "carboplatin", "paclitaxel", "docetaxel",
        "pembrolizumab", "nivolumab", "bevacizumab", "trastuzumab",
        "fluorouracil", "doxorubicin", "cyclophosphamide", "methotrexate",
    }
    disease_canonicals = {
        "cancer", "breast cancer", "lung cancer", "colorectal cancer",
        "prostate cancer", "head and neck cancer", "buccal mucosa cancer",
        "laryngeal cancer", "lymphoma", "leukemia", "melanoma", "sarcoma",
        "carcinoma", "solid tumor",
    }
    treatment_canonicals = {
        "chemotherapy", "radiation therapy", "immunotherapy",
        "targeted therapy", "palliative care", "surgery",
        "bone marrow transplant", "hormone therapy",
    }
    clinical_canonicals = {
        "staging", "metastasis", "remission", "recurrence", "biopsy",
        "prognosis", "progression", "complete response", "tumor marker",
        "clinical trial", "adverse effects",
    }

    if canonical in drug_canonicals:
        return "DRUG"
    if canonical in disease_canonicals:
        return "DISEASE"
    if canonical in treatment_canonicals:
        return "TREATMENT"
    if canonical in clinical_canonicals:
        return "CLINICAL_TERM"
    return "ENTITY"


def extract_medical_entities(text: str) -> list["MedicalEntity"]:
    """
    Extract biomedical named entities from text using scispaCy en_core_sci_md.

    Post-processing steps:
      1. Filter entities shorter than _MIN_ENTITY_LEN characters.
      2. Filter common stopwords incorrectly tagged as entities.
      3. Enrich the generic "ENTITY" label with oncology-specific labels
         (DRUG, DISEASE, TREATMENT, CLINICAL_TERM) using the synonym dictionary.
      4. Deduplicate by entity text (case-insensitive).

    Args:
        text: English text to extract entities from (query or document snippet).

    Returns:
        List of MedicalEntity dataclass instances, sorted by start offset.
    """
    from src.models import MedicalEntity

    if not text or not text.strip():
        return []

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nlp = _get_spacy_model()
            doc = nlp(text)

        seen_texts: set[str] = set()
        entities: list[MedicalEntity] = []

        for ent in doc.ents:
            ent_text = ent.text.strip()
            lower    = ent_text.lower()

            # Filter: too short
            if len(ent_text) < _MIN_ENTITY_LEN:
                continue

            # Filter: stopwords
            if lower in _ENTITY_STOPWORDS:
                continue

            # Filter: purely numeric
            if re.match(r"^\d+\.?\d*$", ent_text):
                continue

            # Deduplicate (keep first occurrence)
            if lower in seen_texts:
                continue
            seen_texts.add(lower)

            # Enrich label
            label = _enrich_entity_label(ent_text)

            entities.append(
                MedicalEntity(
                    text=ent_text,
                    label=label,
                    start=ent.start_char,
                    end=ent.end_char,
                )
            )

        logger.debug(
            "extract_medical_entities: found %d entities in %d chars",
            len(entities), len(text),
        )
        return entities

    except Exception as exc:
        logger.warning(
            "extract_medical_entities: scispaCy failed (%s) — returning empty list",
            exc,
        )
        return []


# ---------------------------------------------------------------------------
# Full LAQA pipeline  (Feature 3.4)
# ---------------------------------------------------------------------------

def laqa_pipeline(
    query: str,
    force_language: Optional[str] = None,
    force_query_type: Optional[str] = None,
) -> "AnalysedQuery":
    """
    Full Language-Aware Query Analysis pipeline.

    Runs three sequential steps on the raw user query:
      1. detect_language()          — identify source language
      2. classify_query_type()      — classify into one of 6 Q&A types
      3. extract_medical_entities() — extract oncology NER from English text

    The translated_query field is populated by query_expansion.py (Feature 3.5).
    At this stage it is set to the original query if already English,
    or left as original if non-English (translation happens in next step).

    Args:
        query:            Raw user query in any supported language.
        force_language:   Override language detection (e.g. "en", "hi").
        force_query_type: Override query type classification (e.g. "mcqa").

    Returns:
        AnalysedQuery dataclass with all fields populated.
    """
    from src.models import AnalysedQuery, Language

    import time
    t0 = time.time()

    if not query or not query.strip():
        logger.warning("laqa_pipeline: received empty query")
        return AnalysedQuery(
            original_query="",
            translated_query="",
            detected_language=Language.UNKNOWN.value,
            query_type="qa",
            query_type_confidence=0.0,
        )

    logger.info("LAQA pipeline started for query: %r", query[:80])

    # --- Step 1: Language detection ---
    if force_language:
        detected_lang = force_language
        logger.info("  [1/3] Language: forced → %s", detected_lang)
    else:
        detected_lang = detect_language(query)
        logger.info("  [1/3] Language detected: %s", detected_lang)

    # For NER and classification we work on English text.
    # If query is already English or Hinglish (partially English), use as-is.
    # Full translation happens in query_expansion.py (Feature 3.5).
    is_english = detected_lang in (Language.ENGLISH.value, Language.HINGLISH.value)
    english_query = query if is_english else query  # placeholder — translated in 3.5

    # --- Step 2: Query type classification ---
    if force_query_type:
        query_type   = force_query_type
        confidence   = 1.0
        logger.info("  [2/3] Query type: forced → %s", query_type)
    elif detected_lang not in (Language.ENGLISH.value, Language.HINGLISH.value):
        # For non-Latin-script languages (Hindi, Tamil, Telugu), BART cannot
        # reliably classify — apply lightweight heuristics instead.
        # Rule-based patterns still work on A)/B)/C)/D) and ___ markers.
        if re.search(r"\b[A-D][).]\s+\w", query):
            query_type, confidence = "mcqa", 0.99
        elif re.search(r"___+|\[blank\]", query):
            query_type, confidence = "fill_blank", 0.99
        else:
            # Default non-English queries to QA — most common type
            query_type, confidence = "qa", 0.80
        logger.info(
            "  [2/3] Query type (non-English heuristic): %s (conf=%.2f)",
            query_type, confidence,
        )
    else:
        # English / Hinglish — full rule-based + BART pipeline
        query_type, confidence = classify_query_type(query)
        logger.info(
            "  [2/3] Query type: %s (conf=%.2f)", query_type, confidence
        )

    # --- Step 3: Medical NER (on English / Hinglish only — scispaCy is English) ---
    if is_english:
        entities = extract_medical_entities(english_query)
    else:
        # For non-English queries, NER will run after translation in pipeline.py
        entities = []
        logger.info(
            "  [3/3] NER skipped for language=%s (will run post-translation)",
            detected_lang,
        )

    if entities:
        logger.info(
            "  [3/3] Medical entities (%d): %s",
            len(entities),
            [f"{e.text}({e.label})" for e in entities[:5]],
        )

    entity_texts = [e.text for e in entities]

    elapsed = time.time() - t0
    logger.info("LAQA pipeline completed in %.2fs", elapsed)

    return AnalysedQuery(
        original_query=query,
        translated_query=english_query,   # updated by query_expansion.py if non-English
        detected_language=detected_lang,
        query_type=query_type,
        query_type_confidence=confidence,
        medical_entities=entities,
        entity_texts=entity_texts,
        expanded_queries=[],              # filled by query_expansion.py
    )
