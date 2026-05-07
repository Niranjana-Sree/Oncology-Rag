"""
Text chunking strategies for MedRAG.

Strategies implemented (added incrementally across features 1.3–1.6):
  1.3  sentence_split()   — sentence-boundary chunks with token overlap
  1.4  recursive_split()  — recursive character splitting with overlap
  1.5  semantic_chunk()   — group sentences by embedding similarity
  1.6  agentic_chunk()    — Claude decides chunk boundaries
       detect_doc_type()  — orchestrator that picks the right strategy
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

logger = logging.getLogger("medrag.chunker")


# ---------------------------------------------------------------------------
# Helpers shared across strategies
# ---------------------------------------------------------------------------

def _make_chunk_id(doc_id: str, index: int) -> str:
    """Stable chunk ID: doc_id + zero-padded index."""
    return f"{doc_id}__{index:04d}"


def _count_tokens(text: str) -> int:
    """
    Approximate token count using whitespace splitting.
    Close enough for chunking decisions without loading a tokenizer.
    """
    return len(text.split())


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation heuristics.
    Handles common medical abbreviations to avoid false splits.
    """
    # Protect common abbreviations that end with a period
    abbrevs = [
        "Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr",
        "et al", "e.g", "i.e", "vs", "approx", "mg", "mL",
        "Fig", "Tab", "Vol", "No", "Sec", "p", "pp",
        "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec",
    ]
    protected = text
    placeholders: dict[str, str] = {}
    for abbr in abbrevs:
        placeholder = f"__ABBR_{hashlib.md5(abbr.encode()).hexdigest()[:6]}__"
        protected = re.sub(rf"\b{re.escape(abbr)}\.", placeholder, protected)
        placeholders[placeholder] = f"{abbr}."

    # Split on sentence-ending punctuation followed by whitespace + capital
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'\(])", protected)

    # Restore abbreviations
    sentences: list[str] = []
    for part in parts:
        for ph, orig in placeholders.items():
            part = part.replace(ph, orig)
        stripped = part.strip()
        if stripped:
            sentences.append(stripped)

    return sentences if sentences else [text.strip()]


def _build_chunks_from_windows(
    sentences: list[str],
    doc_id: str,
    category: str,
    source_file: str,
    strategy: str,
    max_tokens: int,
    overlap_tokens: int,
    base_metadata: Optional[dict] = None,
) -> list["Chunk"]:
    """
    Slide a window over sentences to build Chunk objects respecting
    max_tokens with overlap_tokens of context carried forward.
    """
    from src.models import Chunk

    chunks: list[Chunk] = []
    chunk_index = 0
    i = 0

    while i < len(sentences):
        window: list[str] = []
        token_count = 0

        # Fill window up to max_tokens
        j = i
        while j < len(sentences):
            sent_tokens = _count_tokens(sentences[j])
            if token_count + sent_tokens > max_tokens and window:
                break
            window.append(sentences[j])
            token_count += sent_tokens
            j += 1

        if not window:
            # Single sentence exceeds max_tokens — include it anyway
            window = [sentences[i]]
            j = i + 1

        chunk_text = " ".join(window).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(doc_id, chunk_index),
                    doc_id=doc_id,
                    text=chunk_text,
                    category=category,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    strategy=strategy,
                    metadata={**(base_metadata or {}), "sentence_count": len(window)},
                )
            )
            chunk_index += 1

        # Advance by (window size - overlap)
        # Count how many sentences fit in overlap_tokens
        overlap_sents = 0
        overlap_count = 0
        for sent in reversed(window):
            overlap_count += _count_tokens(sent)
            if overlap_count >= overlap_tokens:
                break
            overlap_sents += 1

        advance = max(1, len(window) - overlap_sents)
        i += advance

    return chunks


# ---------------------------------------------------------------------------
# Strategy 1: sentence_split  (Feature 1.3)
# ---------------------------------------------------------------------------

def sentence_split(
    doc: "RawDocument",
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list["Chunk"]:
    """
    Split a document into chunks at sentence boundaries.

    Each chunk contains as many complete sentences as fit within max_tokens.
    The last overlap_tokens worth of sentences from the previous chunk are
    prepended to the next chunk to preserve context across boundaries.

    Args:
        doc:            RawDocument to chunk.
        max_tokens:     Maximum tokens per chunk (approximate).
        overlap_tokens: Tokens of overlap between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    from src.models import ChunkStrategy

    logger.debug(
        "sentence_split: doc_id=%s max_tokens=%d overlap=%d",
        doc.doc_id, max_tokens, overlap_tokens,
    )

    sentences = _split_into_sentences(doc.raw_text)
    logger.debug("  %d sentences detected", len(sentences))

    chunks = _build_chunks_from_windows(
        sentences=sentences,
        doc_id=doc.doc_id,
        category=doc.category,
        source_file=doc.file_path,
        strategy=ChunkStrategy.SENTENCE.value,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        base_metadata={"source_file": doc.file_path, "category": doc.category},
    )

    logger.info(
        "sentence_split: doc_id=%s → %d chunks (avg %.0f tokens)",
        doc.doc_id,
        len(chunks),
        sum(_count_tokens(c.text) for c in chunks) / max(len(chunks), 1),
    )
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: recursive_split  (Feature 1.4)
# ---------------------------------------------------------------------------

# Separator hierarchy: try each in order until chunks fit within max_tokens
_RECURSIVE_SEPARATORS: list[str] = [
    "\n\n\n",   # multi-blank-line section breaks
    "\n\n",     # paragraph breaks
    "\n",       # line breaks
    ". ",       # sentence endings
    "! ",
    "? ",
    "; ",       # clause boundaries
    ", ",       # phrase boundaries
    " ",        # word boundaries (last resort)
    "",         # character level (absolute last resort)
]


def _recursive_split_text(
    text: str,
    separators: list[str],
    max_tokens: int,
) -> list[str]:
    """
    Recursively split text using a hierarchy of separators until every
    piece fits within max_tokens. Returns a flat list of text fragments.
    """
    if _count_tokens(text) <= max_tokens:
        return [text] if text.strip() else []

    if not separators:
        # Absolute fallback: split by words
        words = text.split()
        pieces: list[str] = []
        for i in range(0, len(words), max_tokens):
            piece = " ".join(words[i: i + max_tokens])
            if piece.strip():
                pieces.append(piece)
        return pieces

    sep = separators[0]
    remaining_seps = separators[1:]

    if sep == "":
        # Character-level split
        chars = list(text)
        pieces = []
        for i in range(0, len(chars), max_tokens * 4):  # ~4 chars per token
            piece = "".join(chars[i: i + max_tokens * 4]).strip()
            if piece:
                pieces.append(piece)
        return pieces

    parts = text.split(sep)
    result: list[str] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if _count_tokens(part) <= max_tokens:
            result.append(part)
        else:
            # This part is still too large — recurse with next separator
            result.extend(_recursive_split_text(part, remaining_seps, max_tokens))

    return result


def recursive_split(
    doc: "RawDocument",
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list["Chunk"]:
    """
    Recursively split a document using a hierarchy of separators.

    Tries to split on double-newlines (paragraphs) first, then single
    newlines, then sentence punctuation, then words — whichever produces
    pieces that fit within max_tokens. After splitting, adjacent pieces are
    merged greedily and overlap is added between consecutive chunks.

    Particularly effective for structured documents (tables, lists, sections)
    where sentence boundaries are inconsistent.

    Args:
        doc:            RawDocument to chunk.
        max_tokens:     Maximum tokens per chunk (approximate).
        overlap_tokens: Tokens of overlap between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    from src.models import Chunk, ChunkStrategy

    logger.debug(
        "recursive_split: doc_id=%s max_tokens=%d overlap=%d",
        doc.doc_id, max_tokens, overlap_tokens,
    )

    # Step 1: recursively split into small fragments
    fragments = _recursive_split_text(doc.raw_text, _RECURSIVE_SEPARATORS, max_tokens)
    logger.debug("  %d fragments after recursive split", len(fragments))

    # Step 2: greedily merge fragments into chunks up to max_tokens
    chunks: list[Chunk] = []
    chunk_index = 0
    i = 0

    while i < len(fragments):
        window: list[str] = []
        token_count = 0

        j = i
        while j < len(fragments):
            frag_tokens = _count_tokens(fragments[j])
            if token_count + frag_tokens > max_tokens and window:
                break
            window.append(fragments[j])
            token_count += frag_tokens
            j += 1

        if not window:
            window = [fragments[i]]
            j = i + 1

        chunk_text = "\n".join(window).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(doc_id=doc.doc_id, index=chunk_index),
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    category=doc.category,
                    source_file=doc.file_path,
                    chunk_index=chunk_index,
                    strategy=ChunkStrategy.RECURSIVE.value,
                    metadata={
                        "source_file": doc.file_path,
                        "category": doc.category,
                        "fragment_count": len(window),
                    },
                )
            )
            chunk_index += 1

        # Step 3: compute overlap — how many trailing fragments fit in overlap_tokens
        overlap_frags = 0
        overlap_count = 0
        for frag in reversed(window):
            overlap_count += _count_tokens(frag)
            if overlap_count >= overlap_tokens:
                break
            overlap_frags += 1

        advance = max(1, len(window) - overlap_frags)
        i += advance

    logger.info(
        "recursive_split: doc_id=%s → %d chunks (avg %.0f tokens)",
        doc.doc_id,
        len(chunks),
        sum(_count_tokens(c.text) for c in chunks) / max(len(chunks), 1),
    )
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: semantic_chunk  (Feature 1.5)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_sentences(sentences: list[str], model_name: str) -> list[list[float]]:
    """
    Embed sentences using sentence-transformers (all-MiniLM-L6-v2).
    Returns L2-normalised float vectors for cosine similarity computation.
    Falls back to zero vectors only if the model cannot be loaded at all.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer(model_name, device="cpu")
        embeddings = model.encode(
            sentences,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        logger.debug(
            "semantic_chunk: embedded %d sentences via %s (dim=%d)",
            len(sentences), model_name, embeddings.shape[1],
        )
        return embeddings.tolist()

    except Exception as exc:
        logger.warning(
            "semantic_chunk: sentence-transformer embedding failed (%s) "
            "— falling back to zero vectors (no semantic grouping)", exc,
        )
        return [[0.0] * 384 for _ in sentences]


def _centroid(vecs: list[list[float]]) -> list[float]:
    """Mean vector of a list of embedding vectors."""
    if not vecs:
        return []
    dim = len(vecs[0])
    c = [0.0] * dim
    for v in vecs:
        for i, val in enumerate(v):
            c[i] += val
    n = len(vecs)
    return [x / n for x in c]


def semantic_chunk(
    doc: "RawDocument",
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    similarity_threshold: float = 0.45,
    embedding_model: Optional[str] = None,
) -> list["Chunk"]:
    """
    AI-driven dynamic chunking using real sentence embeddings.

    Algorithm:
      1. Split document into sentences.
      2. Embed every sentence with all-MiniLM-L6-v2 (384-dim, L2-normalised).
      3. Walk sentences sequentially; start a new chunk when cosine similarity
         between the current sentence embedding and the running centroid of the
         current chunk drops below similarity_threshold, OR the chunk exceeds
         max_tokens.
      4. Prepend overlap_tokens of trailing context into the next chunk.

    The centroid shifts as sentences are added — it represents the evolving
    topic of the current chunk. A similarity drop signals a topic change.

    Args:
        doc:                  RawDocument to chunk.
        max_tokens:           Maximum tokens per chunk (approximate).
        overlap_tokens:       Tokens of overlap between consecutive chunks.
        similarity_threshold: Cosine similarity below which a new chunk starts.
                              0.45 is calibrated for all-MiniLM-L6-v2 dense vectors.
                              Lower → fewer, larger chunks. Higher → more, smaller.
        embedding_model:      SentenceTransformer model ID. Defaults to
                              all-MiniLM-L6-v2 (already cached).

    Returns:
        List of Chunk objects tagged with strategy="semantic".
    """
    from src.models import Chunk, ChunkStrategy

    if embedding_model is None:
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    logger.debug(
        "semantic_chunk: doc_id=%s threshold=%.2f model=%s",
        doc.doc_id, similarity_threshold, embedding_model,
    )

    sentences = _split_into_sentences(doc.raw_text)
    if not sentences:
        return []

    logger.debug("  embedding %d sentences...", len(sentences))
    embeddings = _embed_sentences(sentences, embedding_model)

    chunks: list[Chunk] = []
    chunk_index = 0

    current_sents: list[str] = []
    current_embeds: list[list[float]] = []
    current_tokens: int = 0
    overlap_buffer: list[str] = []

    def _flush() -> None:
        nonlocal chunk_index, current_sents, current_embeds, current_tokens, overlap_buffer
        if not current_sents:
            return
        chunk_text = " ".join(current_sents).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(doc.doc_id, chunk_index),
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    category=doc.category,
                    source_file=doc.file_path,
                    chunk_index=chunk_index,
                    strategy=ChunkStrategy.SEMANTIC.value,
                    metadata={
                        "source_file": doc.file_path,
                        "category": doc.category,
                        "sentence_count": len(current_sents),
                        "similarity_threshold": similarity_threshold,
                    },
                )
            )
            chunk_index += 1

        # Build overlap buffer from trailing sentences
        overlap_buffer = []
        overlap_count = 0
        for sent in reversed(current_sents):
            tok = _count_tokens(sent)
            if overlap_count + tok > overlap_tokens:
                break
            overlap_buffer.insert(0, sent)
            overlap_count += tok

        current_sents.clear()
        current_embeds.clear()
        current_tokens = 0

    for sent, emb in zip(sentences, embeddings):
        sent_tokens = _count_tokens(sent)
        start_new = False

        if current_sents:
            sim = _cosine_similarity(emb, _centroid(current_embeds))
            if sim < similarity_threshold or current_tokens + sent_tokens > max_tokens:
                start_new = True

        if start_new:
            _flush()
            # Prepend overlap into the new chunk
            for osent in overlap_buffer:
                current_sents.append(osent)
                current_tokens += _count_tokens(osent)
            # Zero-embed placeholders for overlap sentences (don't skew centroid)
            current_embeds = [[0.0] * len(emb)] * len(overlap_buffer)

        current_sents.append(sent)
        current_embeds.append(emb)
        current_tokens += sent_tokens

    _flush()

    logger.info(
        "semantic_chunk: doc_id=%s → %d chunks (avg %.0f tokens, threshold=%.2f)",
        doc.doc_id,
        len(chunks),
        sum(_count_tokens(c.text) for c in chunks) / max(len(chunks), 1),
        similarity_threshold,
    )
    return chunks


# ---------------------------------------------------------------------------
# Strategy 4: agentic_chunk  (Feature 1.6)
# ---------------------------------------------------------------------------

# Signals used by detect_doc_type to fingerprint document structure
_RESEARCH_SIGNALS = re.compile(
    r"\b(abstract|introduction|methodology|results|conclusion|references|"
    r"doi:|pubmed|et al\.|figure \d|table \d|p\s*<\s*0\.\d)\b",
    re.IGNORECASE,
)

_GUIDELINE_SIGNALS = re.compile(
    r"\b(recommendation|guideline|grade [abcd]|evidence level|"
    r"strong recommendation|weak recommendation|nccn|asco|esmo|who|"
    r"clinical practice|consensus|staging|tnm)\b",
    re.IGNORECASE,
)

_DRUG_MONOGRAPH_SIGNALS = re.compile(
    r"\b(indication|contraindication|dosage|administration|"
    r"adverse effect|pharmacokinetic|mechanism of action|"
    r"drug interaction|overdose|storage|shelf.?life|"
    r"dose adjustment|renal impairment|hepatic impairment)\b",
    re.IGNORECASE,
)

_STRUCTURED_SIGNALS = re.compile(
    r"(?m)^\s*[-•*]\s+.{10,}|"          # bullet lists
    r"^\s*\d+\.\s+.{10,}|"              # numbered lists
    r"^\s*[A-Z][A-Z\s]{3,}:\s*$|"       # ALL-CAPS headers
    r"\|\s*.+\s*\|",                     # table rows
)


def _regex_fallback_doc_type(doc: "RawDocument") -> str:
    """
    Regex-based fallback for detect_doc_type() when Claude API is unavailable.
    Kept intact — never deleted. Used only when API call fails.
    """
    from src.models import ChunkStrategy, DocCategory

    sample = doc.raw_text[:8000]
    research_hits  = len(_RESEARCH_SIGNALS.findall(sample))
    guideline_hits = len(_GUIDELINE_SIGNALS.findall(sample))
    drug_hits      = len(_DRUG_MONOGRAPH_SIGNALS.findall(sample))
    struct_hits    = len(_STRUCTURED_SIGNALS.findall(sample))
    category       = doc.category.lower()

    if category == DocCategory.DRUG_MONOGRAPH.value or drug_hits >= 4:
        return ChunkStrategy.RECURSIVE.value
    elif category == DocCategory.RESEARCH_PAPER.value or research_hits >= 5:
        return ChunkStrategy.SENTENCE.value
    elif category == DocCategory.CLINICAL_GUIDELINE.value or guideline_hits >= 4:
        return ChunkStrategy.SEMANTIC.value
    elif struct_hits >= 6:
        return ChunkStrategy.RECURSIVE.value
    else:
        return ChunkStrategy.SENTENCE.value


def detect_doc_type(doc: "RawDocument") -> str:
    """
    AI-driven document classifier using Claude API.

    Sends the first 2000 characters of the document to Claude and asks it
    to classify into one of three oncology document types:
      - clinical_guideline → semantic_chunk()   (topically coherent blocks)
      - research_paper     → sentence_split()   (narrative prose)
      - general            → semantic_chunk()   (textbook reference material)

    Falls back to regex heuristics if the API call fails for any reason,
    so chunking never breaks even without network access.

    Args:
        doc: RawDocument to classify.

    Returns:
        A ChunkStrategy value string: "semantic", "sentence", or "recursive".
    """
    from src.models import ChunkStrategy
    from pathlib import Path

    filename = Path(doc.file_path).name
    sample   = doc.raw_text[:2000].strip()

    # --- Primary path: Claude API classification ---
    try:
        import anthropic
        import config

        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

        prompt = (
            "You are a medical document classifier. Read this text sample "
            "from an oncology document and classify it into exactly one "
            "of these categories:\n"
            "- clinical_guideline: treatment protocols, recommendations, "
            "staging criteria, NCCN/ASCO/ESMO/WHO guidelines\n"
            "- research_paper: clinical studies, trials, abstracts, "
            "methodology, results, statistical analysis\n"
            "- general: textbooks, handbooks, reference material, "
            "educational content, manuals\n\n"
            "Return ONLY the category name, nothing else.\n\n"
            f"Document sample:\n{sample}"
        )

        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )

        claude_category = response.content[0].text.strip().lower()

        # Map Claude's category → chunking strategy
        category_to_strategy = {
            "clinical_guideline": ChunkStrategy.SEMANTIC.value,
            "research_paper":     ChunkStrategy.SENTENCE.value,
            "general":            ChunkStrategy.SEMANTIC.value,
        }

        if claude_category not in category_to_strategy:
            logger.warning(
                "detect_doc_type: Claude returned unknown category '%s' for '%s' "
                "— falling back to regex",
                claude_category, filename,
            )
            strategy = _regex_fallback_doc_type(doc)
            print(f"Document: {filename} → Claude returned unknown: '{claude_category}' → Regex fallback: {strategy}")
            return strategy

        strategy = category_to_strategy[claude_category]
        print(f"Document: {filename} → Claude classified as: {claude_category} → Strategy: {strategy}")
        logger.info(
            "detect_doc_type: '%s' → Claude=%s → strategy=%s",
            filename, claude_category, strategy,
        )
        return strategy

    except Exception as exc:
        # --- Fallback path: regex heuristics ---
        logger.warning(
            "detect_doc_type: Claude API call failed for '%s' (%s) "
            "— using regex fallback",
            filename, exc,
        )
        strategy = _regex_fallback_doc_type(doc)
        print(f"Document: {filename} → Claude unavailable ({type(exc).__name__}) → Regex fallback: {strategy}")
        return strategy


def agentic_chunk(
    doc: "RawDocument",
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list["Chunk"]:
    """
    Agentic chunking orchestrator: automatically detects the document type
    and delegates to the most appropriate chunking strategy.

    Detection logic (via detect_doc_type):
      - Drug monographs        → recursive_split  (structured sections)
      - Research papers        → sentence_split   (narrative prose)
      - Clinical guidelines    → semantic_chunk   (topical coherence)
      - Highly structured docs → recursive_split
      - Unknown / general      → sentence_split

    After chunking with the chosen strategy, each chunk's strategy field
    is overwritten to "agentic" so downstream code knows this path was used,
    while the original strategy is stored in metadata["delegated_strategy"].

    Args:
        doc:            RawDocument to chunk.
        max_tokens:     Maximum tokens per chunk.
        overlap_tokens: Token overlap between consecutive chunks.

    Returns:
        List of Chunk objects tagged with strategy="agentic".
    """
    from src.models import ChunkStrategy

    chosen = detect_doc_type(doc)
    logger.info(
        "agentic_chunk: doc_id=%s → delegating to %s", doc.doc_id, chosen
    )

    # Delegate to the chosen strategy
    if chosen == ChunkStrategy.RECURSIVE.value:
        chunks = recursive_split(doc, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    elif chosen == ChunkStrategy.SEMANTIC.value:
        chunks = semantic_chunk(doc, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    else:
        chunks = sentence_split(doc, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    # Tag all chunks as "agentic" and record the delegated strategy
    for chunk in chunks:
        chunk.metadata["delegated_strategy"] = chunk.strategy
        chunk.strategy = ChunkStrategy.AGENTIC.value

    logger.info(
        "agentic_chunk: doc_id=%s → %d chunks via %s",
        doc.doc_id, len(chunks), chosen,
    )
    return chunks
