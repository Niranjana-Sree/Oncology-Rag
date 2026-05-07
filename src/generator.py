"""
src/generator.py — LLM answer generation for MedRAG.

Features implemented across 5.2–5.4:
  5.2  generate_with_claude()      — Anthropic API fallback generator
  5.3  post_generation_checks()    — safety, length, grounding validation
  5.4  generate_with_medgemma()    — local MedGemma primary generator

Public API
----------
generate_with_claude(prompt_dict, max_tokens) → GeneratedAnswer
post_generation_checks(answer, chunks, query) → GeneratedAnswer (validated)
generate_with_medgemma(prompt_dict, max_tokens) → GeneratedAnswer
generate(prompt_dict, chunks, query)           → GeneratedAnswer (primary + fallback)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import config
from src.models import GeneratedAnswer, RetrievedChunk

logger = logging.getLogger("medrag.generator")

# ---------------------------------------------------------------------------
# Safety patterns — responses containing these are flagged
# ---------------------------------------------------------------------------
_SAFETY_PATTERNS = [
    "i cannot provide",
    "i'm unable to",
    "as an ai",
    "i don't have access",
    "consult a doctor",       # too generic — we add our own medical disclaimer
    "i cannot diagnose",
    "seek immediate medical",
]

_MIN_ANSWER_LENGTH = 30    # chars — shorter answers are likely refusals
_MAX_ANSWER_LENGTH = 4000  # chars — cap runaway generation


# ---------------------------------------------------------------------------
# Feature 5.2 — generate_with_claude()
# ---------------------------------------------------------------------------

def generate_with_claude(
    prompt_dict: dict,
    max_tokens: int = 1024,
) -> GeneratedAnswer:
    """
    Generate an answer using the Anthropic Claude API.

    Used as the primary generator (MedGemma is not yet available in venv)
    and as fallback when MedGemma fails.

    Args:
        prompt_dict: Dict from build_prompt() with "system" and "user" keys.
        max_tokens:  Maximum tokens to generate.

    Returns:
        GeneratedAnswer with answer_text, token counts, and latency.

    Raises:
        RuntimeError if Claude API is unavailable and no fallback exists.
    """
    import anthropic

    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set — cannot use Claude generator")

    system_prompt = prompt_dict.get("system", "")
    user_message  = prompt_dict.get("user", "")

    if not user_message.strip():
        raise ValueError("generate_with_claude: empty user message")

    logger.info(
        "generate_with_claude: model=%s max_tokens=%d prompt_len=%d",
        config.CLAUDE_MODEL, max_tokens, len(user_message),
    )

    t0 = time.time()
    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        answer_text      = response.content[0].text.strip()
        prompt_tokens    = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        latency_ms       = (time.time() - t0) * 1000

        logger.info(
            "generate_with_claude: done — %d chars, %d+%d tokens, %.0fms",
            len(answer_text), prompt_tokens, completion_tokens, latency_ms,
        )

        return GeneratedAnswer(
            answer_text=answer_text,
            model_used="claude-" + config.CLAUDE_MODEL,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            generation_latency_ms=latency_ms,
        )

    except Exception as exc:
        latency_ms = (time.time() - t0) * 1000
        logger.error("generate_with_claude: API call failed (%s)", exc)
        raise RuntimeError(f"Claude generation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Feature 5.3 — post_generation_checks()
# ---------------------------------------------------------------------------

def post_generation_checks(
    generated: GeneratedAnswer,
    chunks: list[RetrievedChunk],
    query: str,
) -> GeneratedAnswer:
    """
    Validate a generated answer on three dimensions and update safety flags.

    Three checks:
      1. Length check      — answer must be >= _MIN_ANSWER_LENGTH chars
                             and <= _MAX_ANSWER_LENGTH chars
      2. Safety check      — answer must not contain refusal/off-topic patterns
      3. Grounding check   — answer should share at least some vocabulary
                             with the retrieved context (loose check)

    The function never raises — it always returns a GeneratedAnswer.
    If a check fails, passed_safety_check is set to False and the
    relevant flag is added to safety_flags.

    Args:
        generated: GeneratedAnswer from generate_with_claude() or generate_with_medgemma().
        chunks:    Retrieved chunks used to build the prompt (for grounding check).
        query:     Original query string (for context).

    Returns:
        Updated GeneratedAnswer with passed_safety_check and safety_flags set.
    """
    import copy
    import re

    result = copy.copy(generated)
    result.safety_flags = []
    flags: list[str] = []

    answer = result.answer_text.strip()
    answer_lower = answer.lower()

    # ------------------------------------------------------------------
    # Check 1: Length
    # ------------------------------------------------------------------
    if len(answer) < _MIN_ANSWER_LENGTH:
        flags.append(f"TOO_SHORT:{len(answer)}_chars")
        logger.warning(
            "post_generation_checks: answer too short (%d chars < %d)",
            len(answer), _MIN_ANSWER_LENGTH,
        )

    if len(answer) > _MAX_ANSWER_LENGTH:
        # Truncate and flag — don't fail, just trim
        result.answer_text = answer[:_MAX_ANSWER_LENGTH] + "\n\n[Answer truncated]"
        flags.append(f"TRUNCATED:{len(answer)}_chars")
        logger.warning(
            "post_generation_checks: answer truncated (%d > %d chars)",
            len(answer), _MAX_ANSWER_LENGTH,
        )
        answer_lower = result.answer_text.lower()

    # ------------------------------------------------------------------
    # Check 2: Safety patterns (refusals / off-topic responses)
    # ------------------------------------------------------------------
    for pattern in _SAFETY_PATTERNS:
        if pattern in answer_lower:
            flags.append(f"SAFETY_PATTERN:{pattern!r}")
            logger.warning(
                "post_generation_checks: safety pattern found: %r", pattern
            )

    # ------------------------------------------------------------------
    # Check 3: Grounding — answer should share vocabulary with context
    # ------------------------------------------------------------------
    if chunks:
        # Extract significant words from context (>= 5 chars)
        context_text = " ".join(c.text.lower() for c in chunks[:5])
        context_words = set(re.findall(r"[a-z]{5,}", context_text))

        # Extract significant words from answer
        answer_words = set(re.findall(r"[a-z]{5,}", answer_lower))

        # Overlap ratio
        if context_words:
            overlap = len(answer_words & context_words)
            overlap_ratio = overlap / max(len(answer_words), 1)

            if overlap_ratio < 0.05:   # less than 5% vocabulary overlap
                flags.append(f"LOW_GROUNDING:overlap_ratio={overlap_ratio:.3f}")
                logger.warning(
                    "post_generation_checks: low grounding (overlap=%.3f)", overlap_ratio
                )
            else:
                logger.debug(
                    "post_generation_checks: grounding ok (overlap=%.3f, %d shared words)",
                    overlap_ratio, overlap,
                )

    # ------------------------------------------------------------------
    # Set final safety status
    # ------------------------------------------------------------------
    result.safety_flags = flags
    # Only fail on TOO_SHORT or SAFETY_PATTERN — grounding and truncation
    # are warnings, not hard failures
    hard_failures = [f for f in flags if f.startswith("TOO_SHORT") or f.startswith("SAFETY_PATTERN")]
    result.passed_safety_check = len(hard_failures) == 0

    logger.info(
        "post_generation_checks: passed=%s  flags=%s",
        result.passed_safety_check, flags or "none",
    )
    return result


# ---------------------------------------------------------------------------
# Feature 5.4 — generate_with_medgemma()
# ---------------------------------------------------------------------------

# Module-level cache for MedGemma model + tokenizer
_medgemma_cache: dict = {}


def _load_medgemma():
    """
    Load MedGemma tokenizer and model with automatic quantization selection.

    Memory strategy (auto-selected):
      - CUDA available           → float16, device_map=auto (~8GB VRAM)
      - bitsandbytes available   → 4-bit quantization (~2.5GB RAM, CPU/MPS)
      - fallback                 → float32 CPU (slow but works if enough RAM)

    Returns (tokenizer, model) or raises RuntimeError.
    """
    if "model" in _medgemma_cache:
        return _medgemma_cache["tokenizer"], _medgemma_cache["model"]

    model_path = config.MEDGEMMA_MODEL_PATH
    device     = config.MEDGEMMA_DEVICE

    logger.info("Loading MedGemma from: %s  device=%s", model_path, device)

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import importlib

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        has_bitsandbytes = importlib.util.find_spec("bitsandbytes") is not None
        has_cuda = torch.cuda.is_available()

        if has_cuda:
            # Full precision on GPU
            logger.info("MedGemma: loading in float16 on CUDA")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        elif has_bitsandbytes:
            # 4-bit quantization — fits in ~2.5GB RAM (works on Mac 8GB)
            logger.info("MedGemma: loading in 4-bit quantization (bitsandbytes)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

        else:
            # Full float32 on CPU — slow, needs ~16GB RAM
            logger.warning("MedGemma: loading in float32 on CPU (no GPU or bitsandbytes)")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            model = model.to("cpu")

        model.eval()
        _medgemma_cache["tokenizer"] = tokenizer
        _medgemma_cache["model"]     = model

        # Log memory usage
        try:
            if has_cuda:
                mem_gb = torch.cuda.memory_allocated() / 1e9
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                mem_gb = torch.mps.current_allocated_memory() / 1e9
            else:
                mem_gb = 0
            logger.info("MedGemma loaded — allocated %.2f GB", mem_gb)
        except Exception:
            logger.info("MedGemma loaded successfully")

        return tokenizer, model

    except Exception as exc:
        logger.error("Failed to load MedGemma: %s", exc)
        raise RuntimeError(f"MedGemma load failed: {exc}") from exc


def generate_with_medgemma(
    prompt_dict: dict,
    max_tokens: int = 1024,
) -> GeneratedAnswer:
    """
    Generate an answer using the local MedGemma 4B model.

    MedGemma is a medical-domain fine-tune of Google's Gemma model.
    It runs locally (no API cost) but requires:
      - ~8 GB RAM/VRAM for 4B parameter model
      - MEDGEMMA_MODEL_PATH set in .env (HuggingFace model ID or local path)
      - HuggingFace token if using gated model (google/medgemma-4b-it)

    Prompt format: MedGemma uses the Gemma instruction-tuned chat format
    with <start_of_turn> / <end_of_turn> tokens.

    Falls back to Claude if:
      - Model path not set
      - Insufficient memory
      - Model load fails for any reason

    Args:
        prompt_dict: Dict from build_prompt() with "system" and "user" keys.
        max_tokens:  Maximum new tokens to generate.

    Returns:
        GeneratedAnswer with model_used="medgemma-4b" on success,
        or delegates to generate_with_claude() on failure.
    """
    import time

    system_prompt = prompt_dict.get("system", "")
    user_message  = prompt_dict.get("user", "")

    if not config.MEDGEMMA_MODEL_PATH:
        logger.warning("MEDGEMMA_MODEL_PATH not set — falling back to Claude")
        result = generate_with_claude(prompt_dict, max_tokens=max_tokens)
        result.model_used = "claude-fallback"
        return result

    try:
        import torch

        tokenizer, model = _load_medgemma()
        device = config.MEDGEMMA_DEVICE

        # Format prompt in Gemma instruction-tuned format
        # System message prepended to the user turn
        full_prompt = (
            f"<start_of_turn>user\n"
            f"{system_prompt}\n\n"
            f"{user_message}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        logger.info(
            "generate_with_medgemma: prompt_len=%d  max_tokens=%d  device=%s",
            len(full_prompt), max_tokens, device,
        )

        t0 = time.time()
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        prompt_token_count = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,          # greedy decoding for clinical precision
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (skip input tokens)
        new_tokens = outputs[0][prompt_token_count:]
        answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completion_tokens = len(new_tokens)
        latency_ms = (time.time() - t0) * 1000

        logger.info(
            "generate_with_medgemma: done — %d chars, %d+%d tokens, %.0fms",
            len(answer_text), prompt_token_count, completion_tokens, latency_ms,
        )

        return GeneratedAnswer(
            answer_text=answer_text,
            model_used="medgemma-4b",
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_tokens,
            generation_latency_ms=latency_ms,
        )

    except Exception as exc:
        logger.warning(
            "generate_with_medgemma: failed (%s) — falling back to Claude", exc
        )
        result = generate_with_claude(prompt_dict, max_tokens=max_tokens)
        result.model_used = "claude-fallback"
        return result


# ---------------------------------------------------------------------------
# Unified generator — primary (MedGemma) + fallback (Claude)
# ---------------------------------------------------------------------------

def generate(
    prompt_dict: dict,
    chunks: list[RetrievedChunk],
    query: str,
    max_tokens: int = 1024,
    prefer_local: bool = True,
) -> GeneratedAnswer:
    """
    Unified generation entry point with automatic fallback.

    Primary  : MedGemma 4B (local, no API cost) — if available
    Fallback : Claude API — if MedGemma unavailable or fails

    After generation, runs post_generation_checks() on the answer.

    Args:
        prompt_dict:  Dict from build_prompt().
        chunks:       Retrieved chunks (for grounding check).
        query:        Original query string.
        max_tokens:   Maximum tokens to generate.
        prefer_local: If True, try MedGemma first. If False, use Claude directly.

    Returns:
        Validated GeneratedAnswer.
    """
    if prefer_local and config.MEDGEMMA_MODEL_PATH:
        raw = generate_with_medgemma(prompt_dict, max_tokens=max_tokens)
    else:
        raw = generate_with_claude(prompt_dict, max_tokens=max_tokens)

    return post_generation_checks(raw, chunks, query)


def clean_answer(text: str) -> str:
    """Strip markdown formatting from generated answer text."""
    import re
    if not text:
        return text
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
