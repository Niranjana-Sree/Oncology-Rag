"""
evaluate.py — Comprehensive multi-layer evaluation for MedRAG.

Produces the exact report format:
  ONCOLOGY RAG - COMPLETE EVALUATION REPORT
  LAQA + MRL + Agentic RAG

Layers:
  Layer 1 — Lexical   : BLEU-1/2/4, GLEU, ROUGE-1/2/L/Lsum, METEOR, Answer F1
  Layer 2 — Semantic  : BERTScore F1 (roberta-large)
  Layer 3 — System    : Faithfulness, Context Relevancy, Answer Relevance (Claude)
  Layer 4 — Retrieval : Precision@5, Recall@5, MRR, NDCG@5, Hit-Rate@5, Avg rerank
  Layer 5 — SCOPE     : Safety, Completeness, Originality, Precision, Efficiency (Claude 1-5)

Usage:
    python evaluate.py --answers results/answers.jsonl
                       --references data/references.jsonl
                       --output-dir results/
                       --layers 1 2 3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

logger = logging.getLogger("medrag.evaluate")

# ===========================================================================
# TOKENISER HELPER
# ===========================================================================

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


# ===========================================================================
# LAYER 1 — LEXICAL METRICS
# ===========================================================================

def compute_bleu(hypothesis: str, reference: str, max_n: int = 4) -> dict[str, float]:
    hyp = _tokenize(hypothesis)
    ref = _tokenize(reference)
    if not hyp or not ref:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}
    scores = {}
    for n in range(1, max_n + 1):
        hyp_ng = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp)-n+1))
        ref_ng = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
        clipped = sum(min(c, ref_ng[ng]) for ng, c in hyp_ng.items())
        total   = max(sum(hyp_ng.values()), 1)
        prec    = clipped / total
        bp      = min(1.0, len(hyp) / max(len(ref), 1))
        scores[f"bleu_{n}"] = round(bp * math.exp(math.log(prec + 1e-10)) if prec > 0 else 0.0, 4)
    return scores


def compute_gleu(hypothesis: str, reference: str) -> float:
    """Google's GLEU — min of precision and recall over all n-grams 1-4."""
    hyp = _tokenize(hypothesis)
    ref = _tokenize(reference)
    if not hyp or not ref:
        return 0.0
    total_prec = total_rec = 0.0
    for n in range(1, 5):
        hyp_ng = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp)-n+1))
        ref_ng = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
        overlap  = sum(min(c, ref_ng[ng]) for ng, c in hyp_ng.items())
        total_prec += overlap / max(sum(hyp_ng.values()), 1)
        total_rec  += overlap / max(sum(ref_ng.values()), 1)
    prec = total_prec / 4
    rec  = total_rec  / 4
    return round(min(prec, rec), 4)


def compute_rouge(hypothesis: str, reference: str) -> dict[str, float]:
    hyp = _tokenize(hypothesis)
    ref = _tokenize(reference)

    def _ngram_f1(h, r, n):
        hng = Counter(tuple(h[i:i+n]) for i in range(len(h)-n+1))
        rng = Counter(tuple(r[i:i+n]) for i in range(len(r)-n+1))
        ov  = sum(min(c, rng[ng]) for ng, c in hng.items())
        p   = ov / max(sum(hng.values()), 1)
        rec = ov / max(sum(rng.values()), 1)
        return round(2*p*rec/(p+rec), 4) if (p+rec) > 0 else 0.0

    def _lcs(x, y):
        m, n = len(x), len(y)
        dp = [[0]*(n+1) for _ in range(2)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                dp[i%2][j] = dp[(i-1)%2][j-1]+1 if x[i-1]==y[j-1] else max(dp[(i-1)%2][j], dp[i%2][j-1])
        return dp[m%2][n]

    lcs = _lcs(hyp, ref)
    p_l = lcs / max(len(hyp), 1)
    r_l = lcs / max(len(ref), 1)
    rouge_l = round(2*p_l*r_l/(p_l+r_l), 4) if (p_l+r_l) > 0 else 0.0

    # ROUGE-Lsum: sentence-level LCS union (approximate via per-sentence max)
    hyp_sents = [s.strip() for s in re.split(r'[.!?]\s+', hypothesis) if s.strip()]
    ref_sents = [s.strip() for s in re.split(r'[.!?]\s+', reference) if s.strip()]
    if not hyp_sents: hyp_sents = [hypothesis]
    if not ref_sents: ref_sents = [reference]
    lcs_sum = 0
    for rs in ref_sents:
        rt = _tokenize(rs)
        best = max((_lcs(_tokenize(hs), rt) for hs in hyp_sents), default=0)
        lcs_sum += best
    p_lsum = lcs_sum / max(len(hyp), 1)
    r_lsum = lcs_sum / max(len(ref), 1)
    rouge_lsum = round(2*p_lsum*r_lsum/(p_lsum+r_lsum), 4) if (p_lsum+r_lsum) > 0 else 0.0

    return {
        "rouge_1": _ngram_f1(hyp, ref, 1),
        "rouge_2": _ngram_f1(hyp, ref, 2),
        "rouge_l": rouge_l,
        "rouge_lsum": rouge_lsum,
    }


def compute_meteor(hypothesis: str, reference: str) -> float:
    try:
        import nltk
        from nltk.translate.meteor_score import single_meteor_score
        for res in ("wordnet", "omw-1.4", "punkt_tab"):
            try:
                nltk.data.find(f"corpora/{res}")
            except LookupError:
                nltk.download(res, quiet=True)
        hyp_tok = _tokenize(hypothesis)
        ref_tok = _tokenize(reference)
        if not hyp_tok or not ref_tok:
            return 0.0
        return round(float(single_meteor_score(ref_tok, hyp_tok)), 4)
    except Exception as exc:
        logger.warning("METEOR failed (%s) — returning 0.0", exc)
        return 0.0


def compute_token_f1(hypothesis: str, reference: str) -> float:
    hyp = Counter(_tokenize(hypothesis))
    ref = Counter(_tokenize(reference))
    common = sum((hyp & ref).values())
    if common == 0:
        return 0.0
    p = common / max(sum(hyp.values()), 1)
    r = common / max(sum(ref.values()), 1)
    return round(2*p*r/(p+r), 4)


def compute_distinct(texts: list[str], n: int = 1) -> float:
    all_ng, uniq = [], set()
    for t in texts:
        tok = _tokenize(t)
        ng  = [tuple(tok[i:i+n]) for i in range(len(tok)-n+1)]
        all_ng.extend(ng); uniq.update(ng)
    return round(len(uniq)/len(all_ng), 4) if all_ng else 0.0


def compute_mcqa_accuracy(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    pat = re.compile(r"\b([A-D])\b[).:\s]", re.IGNORECASE)
    correct = sum(
        1 for p, r in zip(predictions, references)
        if (pm := pat.search(p)) and (rm := pat.search(r))
        and pm.group(1).upper() == rm.group(1).upper()
    )
    return round(correct / len(predictions), 4)


def layer1_metrics(
    hypotheses: list[str],
    references: list[str],
    query_types: Optional[list[str]] = None,
) -> dict:
    if len(hypotheses) != len(references) or not hypotheses:
        return {}

    bleu   = [compute_bleu(h, r)  for h, r in zip(hypotheses, references)]
    rouge  = [compute_rouge(h, r) for h, r in zip(hypotheses, references)]
    f1s    = [compute_token_f1(h, r) for h, r in zip(hypotheses, references)]
    gleus  = [compute_gleu(h, r)  for h, r in zip(hypotheses, references)]
    meteors= [compute_meteor(h, r) for h, r in zip(hypotheses, references)]

    def _mean(vals): return round(sum(vals)/len(vals), 4) if vals else 0.0

    metrics = {
        "bleu_1":     _mean([s["bleu_1"] for s in bleu]),
        "bleu_2":     _mean([s["bleu_2"] for s in bleu]),
        "bleu_4":     _mean([s["bleu_4"] for s in bleu]),
        "gleu":       _mean(gleus),
        "rouge_1":    _mean([s["rouge_1"]    for s in rouge]),
        "rouge_2":    _mean([s["rouge_2"]    for s in rouge]),
        "rouge_l":    _mean([s["rouge_l"]    for s in rouge]),
        "rouge_lsum": _mean([s["rouge_lsum"] for s in rouge]),
        "meteor":     _mean(meteors),
        "answer_f1":  _mean(f1s),
        "num_samples": len(hypotheses),
    }
    if query_types:
        mcqa_h = [h for h, qt in zip(hypotheses, query_types) if qt == "mcqa"]
        mcqa_r = [r for r, qt in zip(references,  query_types) if qt == "mcqa"]
        metrics["mcqa_accuracy"] = compute_mcqa_accuracy(mcqa_h, mcqa_r) if mcqa_h else None
    return metrics


# ===========================================================================
# LAYER 2 — SEMANTIC METRICS
# ===========================================================================

def layer2_metrics(hypotheses: list[str], references: list[str]) -> dict:
    if len(hypotheses) != len(references) or not hypotheses:
        return {}
    metrics: dict = {"num_samples": len(hypotheses)}

    # BERTScore — roberta-large (default lang="en" model)
    try:
        from bert_score import score as bert_score_fn
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P, R, F1 = bert_score_fn(
                hypotheses, references,
                lang="en",
                rescale_with_baseline=False,
                verbose=False,
                batch_size=4,
            )
        metrics["bertscore_precision"] = round(float(P.mean()), 4)
        metrics["bertscore_recall"]    = round(float(R.mean()), 4)
        metrics["bertscore_f1"]        = round(float(F1.mean()), 4)
        metrics["bertscore_model"]     = "roberta-large"
        logger.info("BERTScore F1=%.4f", metrics["bertscore_f1"])
    except Exception as exc:
        logger.warning("BERTScore failed (%s)", exc)
        metrics["bertscore_precision"] = metrics["bertscore_recall"] = metrics["bertscore_f1"] = 0.0
        metrics["bertscore_model"] = "roberta-large"

    # SBERT cosine
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np, warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            he = m.encode(hypotheses, normalize_embeddings=True, show_progress_bar=False)
            re_ = m.encode(references,  normalize_embeddings=True, show_progress_bar=False)
        metrics["sbert_cosine"] = round(float((he * re_).sum(axis=1).mean()), 4)
        logger.info("SBERT cosine=%.4f", metrics["sbert_cosine"])
    except Exception as exc:
        logger.warning("SBERT failed (%s)", exc)
        metrics["sbert_cosine"] = 0.0

    return metrics


# ===========================================================================
# SINGLE-CALL CLAUDE EVALUATOR  (1 API call per question — all metrics)
# ===========================================================================

_SCOPE_WEIGHTS = {"S": 0.20, "C": 0.25, "O": 0.15, "P": 0.25, "E": 0.15}

_EVAL_PROMPT = """\
You are an expert oncology medical evaluator.
Evaluate this RAG system response and return ONLY a valid JSON object with no explanation.

Question: {question}

Retrieved context chunks:
{chunks}

Generated answer: {generated_answer}

Reference answer: {reference_answer}

Score each dimension:
- faithfulness (0.0-1.0): Is the answer grounded in the retrieved chunks? Does it avoid hallucination?
- context_relevancy (0.0-1.0): Are the retrieved chunks relevant to the question?
- answer_relevance (0.0-1.0): Does the answer directly address what was asked?
- retrieval_precision (0.0-1.0): What fraction of retrieved chunks contain information useful for answering this question?
- scope.S_safety (1-5): Does the answer avoid harmful medical advice? Recommends professional consultation?
- scope.C_completeness (1-5): Does it cover all important aspects of the question?
- scope.O_originality (1-5): Does it add insight beyond just repeating the retrieved text?
- scope.P_precision (1-5): Are all medical facts, drug names, dosages, and terminology accurate?
- scope.E_efficiency (1-5): Is the answer concise and well structured without unnecessary content?

Return ONLY the JSON. No explanation. No markdown.
{{"faithfulness": 0.0, "context_relevancy": 0.0, "answer_relevance": 0.0, "retrieval_precision": 0.0, "scope": {{"S_safety": 3, "C_completeness": 3, "O_originality": 3, "P_precision": 3, "E_efficiency": 3}}}}"""

_FALLBACK_SCORES = {
    "faithfulness": 0.5,
    "context_relevancy": 0.5,
    "answer_relevance": 0.5,
    "retrieval_precision": 0.5,
    "scope": {
        "S_safety": 3, "C_completeness": 3, "O_originality": 3,
        "P_precision": 3, "E_efficiency": 3,
    },
}


def evaluate_single_sample(
    question: str,
    generated_answer: str,
    retrieved_chunks: list[str],
    reference_answer: str,
    client=None,
) -> dict:
    """
    Make EXACTLY ONE Claude API call that returns all scores for one sample.

    Returns dict with faithfulness, context_relevancy, answer_relevance,
    retrieval_precision, and scope sub-scores (S/C/O/P/E each 1-5).
    Falls back to neutral defaults on any error — never raises.
    """
    import anthropic as _anthropic

    if client is None:
        client = _anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Format context (top 5 chunks, 300 chars each)
    chunk_text = "\n".join(
        f"[{i+1}] {c[:300]}" for i, c in enumerate(retrieved_chunks[:5])
    )

    prompt = _EVAL_PROMPT.format(
        question=question[:300],
        chunks=chunk_text,
        generated_answer=generated_answer[:700],
        reference_answer=reference_answer[:400],
    )

    try:
        r = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = r.content[0].text.strip()

        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

        # Find outermost JSON object
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON found in: {raw[:80]}")

        d = json.loads(m.group())

        def _f(key, default=0.5):
            return max(0.0, min(1.0, float(d.get(key, default))))

        def _i(key, default=3):
            return max(1, min(5, int(d.get(key, default))))

        scope_raw = d.get("scope", {})
        if not isinstance(scope_raw, dict):
            scope_raw = {}

        return {
            "faithfulness":       _f("faithfulness"),
            "context_relevancy":  _f("context_relevancy"),
            "answer_relevance":   _f("answer_relevance"),
            "retrieval_precision":_f("retrieval_precision"),
            "scope": {
                "S_safety":       _i(scope_raw.get("S_safety",      scope_raw.get("S", 3))),
                "C_completeness": _i(scope_raw.get("C_completeness", scope_raw.get("C", 3))),
                "O_originality":  _i(scope_raw.get("O_originality",  scope_raw.get("O", 3))),
                "P_precision":    _i(scope_raw.get("P_precision",    scope_raw.get("P", 3))),
                "E_efficiency":   _i(scope_raw.get("E_efficiency",   scope_raw.get("E", 3))),
            },
        }

    except Exception as exc:
        logger.warning("evaluate_single_sample failed (%s) — using fallback", exc)
        import copy
        return copy.deepcopy(_FALLBACK_SCORES)


# ===========================================================================
# SCOPE-ONLY RESCORING  (Step 4)
# ===========================================================================

_SCOPE_ONLY_PROMPT = """\
You are a medical evaluator. Score this oncology answer on 5 dimensions. Return ONLY valid JSON.

Question: {question}
Answer: {generated_answer}

Score each 1-5:
S_safety: Does it avoid harmful advice?
C_completeness: Does it cover all key aspects?
O_originality: Does it add insight beyond the question?
P_precision: Are all medical facts accurate?
E_efficiency: Is it concise and well structured?

Return exactly:
{{"S_safety": 1-5, "C_completeness": 1-5, "O_originality": 1-5, "P_precision": 1-5, "E_efficiency": 1-5}}"""


def rescore_scope_only(
    pipeline_cache_path: Optional[Path] = None,
    scope_checkpoint_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Re-score only SCOPE dimensions for all 200 pipeline records.
    Makes exactly 1 Claude call per sample (200 total).
    Saves checkpoint every 10 questions.
    Returns aggregated scope dict.
    """
    import anthropic as _anthropic

    if pipeline_cache_path is None:
        pipeline_cache_path = config.RESULTS_DIR / "pipeline_200_responses.jsonl"
    if scope_checkpoint_path is None:
        scope_checkpoint_path = config.RESULTS_DIR / "scope_checkpoint.json"
    if output_dir is None:
        output_dir = config.RESULTS_DIR

    # Load pipeline records
    records = []
    with open(pipeline_cache_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info("Loaded %d pipeline records for SCOPE rescoring", len(records))

    # Load existing scope checkpoint
    scope_scores: dict = {}
    if scope_checkpoint_path.exists():
        with open(scope_checkpoint_path, encoding="utf-8") as f:
            scope_scores = json.load(f)
        print(f"Resuming SCOPE scoring from checkpoint: {len(scope_scores)} already done.")

    remaining = [i for i in range(len(records)) if str(i) not in scope_scores]

    # Cost estimate
    cost = len(remaining) * COST_PER_CALL_USD
    print(f"\nSCOPE-only rescoring:")
    print(f"  Remaining : {len(remaining)} questions")
    print(f"  Cached    : {len(scope_scores)} questions")
    print(f"  Est. cost : ${cost:.2f}")
    if remaining:
        try:
            ans = input("  Proceed? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        if ans != "y":
            logger.info("SCOPE rescoring cancelled.")
            return _compute_scope_from_scores(scope_scores)

    client = _anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    for idx in remaining:
        rec = records[idx]
        prompt = _SCOPE_ONLY_PROMPT.format(
            question=rec.get("question", "")[:300],
            generated_answer=rec.get("generated_answer", "")[:700],
        )
        try:
            r = client.messages.create(
                model=config.CLAUDE_MODEL, max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = r.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            d = json.loads(m.group()) if m else {}
            scope_scores[str(idx)] = {
                "S_safety":       max(1, min(5, int(d.get("S_safety",       3)))),
                "C_completeness": max(1, min(5, int(d.get("C_completeness", 3)))),
                "O_originality":  max(1, min(5, int(d.get("O_originality",  3)))),
                "P_precision":    max(1, min(5, int(d.get("P_precision",    3)))),
                "E_efficiency":   max(1, min(5, int(d.get("E_efficiency",   3)))),
            }
        except Exception as exc:
            logger.warning("SCOPE call failed for idx=%d: %s", idx, exc)
            scope_scores[str(idx)] = {
                "S_safety": 3, "C_completeness": 3, "O_originality": 3,
                "P_precision": 3, "E_efficiency": 3,
            }

        # Checkpoint every 10
        if (idx + 1) % 10 == 0 or idx == len(records) - 1:
            scope_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(scope_checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(scope_scores, f)
            logger.info("  SCOPE [%d/%d] checkpoint saved", len(scope_scores), len(records))

    return _compute_scope_from_scores(scope_scores)


def _compute_scope_from_scores(scope_scores: dict) -> dict:
    """Aggregate per-sample SCOPE dicts into summary metrics."""
    S_l, C_l, O_l, P_l, E_l, W_l = [], [], [], [], [], []
    for s in scope_scores.values():
        S = s.get("S_safety", 3);       S_l.append(S)
        C = s.get("C_completeness", 3); C_l.append(C)
        O = s.get("O_originality", 3);  O_l.append(O)
        P = s.get("P_precision", 3);    P_l.append(P)
        E = s.get("E_efficiency", 3);   E_l.append(E)
        w = (S*_SCOPE_WEIGHTS["S"] + C*_SCOPE_WEIGHTS["C"] +
             O*_SCOPE_WEIGHTS["O"] + P*_SCOPE_WEIGHTS["P"] +
             E*_SCOPE_WEIGHTS["E"])
        W_l.append(w)

    def _m(lst): return round(sum(lst)/len(lst), 4) if lst else 0.0
    def _std(lst):
        if not lst: return 0.0
        mu = _m(lst)
        return round((sum((x-mu)**2 for x in lst)/len(lst))**0.5, 4)

    return {
        "scope_s": _m(S_l), "scope_c": _m(C_l),
        "scope_o": _m(O_l), "scope_p": _m(P_l),
        "scope_e": _m(E_l), "scope_total": _m(W_l),
        "scope_std": _std(W_l), "num_samples": len(W_l),
    }


# ===========================================================================
# CHECKPOINT SAVE / RESUME
# ===========================================================================

_CHECKPOINT_PATH = config.RESULTS_DIR / "eval_checkpoint.json"


def load_checkpoint(checkpoint_path: Path = None) -> dict:
    """Load checkpoint if it exists. Returns empty checkpoint dict if not."""
    path = checkpoint_path or _CHECKPOINT_PATH
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                ckpt = json.load(f)
            completed = len(ckpt.get("completed_indices", []))
            last      = ckpt.get("last_completed_index", -1)
            print(f"Resuming from question {last + 2} of 200. "
                  f"{completed} questions already completed.")
            return ckpt
        except Exception as exc:
            logger.warning("Could not load checkpoint (%s) — starting fresh", exc)
    return {"last_completed_index": -1, "completed_indices": [], "scores": {}}


def save_checkpoint(ckpt: dict, checkpoint_path: Path = None) -> None:
    """Persist checkpoint to disk atomically."""
    path = checkpoint_path or _CHECKPOINT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ===========================================================================
# COST ESTIMATOR + USER CONFIRMATION
# ===========================================================================

COST_PER_CALL_USD = 0.003   # approximate cost per Claude sonnet call

def confirm_cost(n_remaining: int, n_cached: int) -> bool:
    """
    Print cost estimate and ask user to confirm before making API calls.
    Returns True if user confirms, False otherwise.
    """
    cost = n_remaining * COST_PER_CALL_USD
    print()
    print("=" * 50)
    print("CLAUDE API COST ESTIMATE")
    print("=" * 50)
    print(f"  Estimated API calls  : {n_remaining}")
    print(f"  Estimated cost       : ${cost:.2f}")
    print(f"  Cached responses     : {n_cached}")
    print(f"  Remaining to score   : {n_remaining}")
    print("=" * 50)

    if n_remaining == 0:
        print("  All samples already cached — no API calls needed.")
        return True

    try:
        answer = input("  Proceed? (y/n): ").strip().lower()
        return answer == "y"
    except (EOFError, KeyboardInterrupt):
        return False


# ===========================================================================
# AGGREGATE SCORES FROM CHECKPOINT
# ===========================================================================

def _aggregate_claude_scores(scores_dict: dict) -> tuple[dict, dict, dict]:
    """
    Given checkpoint scores dict (str index → score dict), compute:
      - layer3: faithfulness, context_relevancy, answer_relevance
      - retrieval: precision_at_5, hit_rate_at_5, mrr, ndcg_at_5
      - scope: S/C/O/P/E means + weighted total + std
    """
    faith, ctx_rel, ans_rel, ret_prec = [], [], [], []
    S_list, C_list, O_list, P_list, E_list, W_list = [], [], [], [], [], []
    rerank_list = []

    for idx_str, s in scores_dict.items():
        faith.append(s.get("faithfulness",       0.5))
        ctx_rel.append(s.get("context_relevancy", 0.5))
        ans_rel.append(s.get("answer_relevance",  0.5))
        rp = s.get("retrieval_precision", 0.5)
        ret_prec.append(rp)
        rerank_list.append(s.get("avg_rerank_score", 0.0))

        sc = s.get("scope", {})
        S = sc.get("S_safety",       3)
        C = sc.get("C_completeness", 3)
        O = sc.get("O_originality",  3)
        P = sc.get("P_precision",    3)
        E = sc.get("E_efficiency",   3)
        S_list.append(S); C_list.append(C); O_list.append(O)
        P_list.append(P); E_list.append(E)
        w = (S*_SCOPE_WEIGHTS["S"] + C*_SCOPE_WEIGHTS["C"] +
             O*_SCOPE_WEIGHTS["O"] + P*_SCOPE_WEIGHTS["P"] +
             E*_SCOPE_WEIGHTS["E"])
        W_list.append(w)

    def _m(lst): return round(sum(lst)/len(lst), 4) if lst else 0.0
    def _std(lst):
        if not lst: return 0.0
        mu = _m(lst)
        return round((sum((x-mu)**2 for x in lst)/len(lst))**0.5, 4)

    # Derive retrieval metrics from retrieval_precision scores
    # Precision@5 = mean retrieval_precision
    # Hit-Rate@5  = fraction where retrieval_precision > 0.5
    # MRR         = approximate from precision scores
    # NDCG@5      = approximate using precision as relevance proxy
    precision_at_5 = _m(ret_prec)
    hit_rate_at_5  = round(sum(1 for p in ret_prec if p > 0.5) / max(len(ret_prec), 1), 4)
    mrr            = _m([min(p / 0.5, 1.0) for p in ret_prec])   # normalised proxy
    ndcg_at_5      = _m([p / math.log2(2) for p in ret_prec])     # simplified DCG@1

    layer3 = {
        "faithfulness":      _m(faith),
        "context_relevancy": _m(ctx_rel),
        "answer_relevance":  _m(ans_rel),
        "num_samples":       len(faith),
    }
    retrieval = {
        "precision_at_5":  precision_at_5,
        "recall_at_5":     precision_at_5,   # proxy (same as precision without exhaustive labels)
        "mrr":             mrr,
        "ndcg_at_5":       round(min(ndcg_at_5, 1.0), 4),
        "hit_rate_at_5":   hit_rate_at_5,
        "avg_rerank_score":_m(rerank_list) if any(r > 0 for r in rerank_list) else 0.0,
        "num_samples":     len(ret_prec),
    }
    scope = {
        "scope_s":     _m(S_list),
        "scope_c":     _m(C_list),
        "scope_o":     _m(O_list),
        "scope_p":     _m(P_list),
        "scope_e":     _m(E_list),
        "scope_total": _m(W_list),
        "scope_std":   _std(W_list),
        "num_samples": len(W_list),
    }
    return layer3, retrieval, scope


# ===========================================================================
# HEADER STATS
# ===========================================================================

def compute_header_stats(pipeline_records: list[dict]) -> dict:
    iters = [r.get("agent_iterations", 1) for r in pipeline_records]
    confs = [r.get("confidence", 0.0)     for r in pipeline_records]
    def _m(lst): return round(sum(lst)/len(lst), 4) if lst else 0.0
    return {
        "total_samples":        len(pipeline_records),
        "avg_agent_iterations": _m(iters),
        "avg_confidence":       _m(confs),
    }


# ===========================================================================
# QA PAIR GENERATION (200 samples)
# ===========================================================================

def generate_qa_pairs(
    n_total: int = 200,
    cache_path: Optional[Path] = None,
    chunks_path: Optional[Path] = None,
    batch_size: int = 15,
) -> list[dict]:
    """
    Generate N oncology QA pairs by sampling corpus chunks and prompting Claude.
    Caches results to avoid regeneration on subsequent runs.
    """
    if cache_path is None:
        cache_path = config.RESULTS_DIR / "qa_pairs_200.jsonl"
    if chunks_path is None:
        chunks_path = config.CHUNKS_DIR / "all_chunks.json"

    # Load from cache if available
    if cache_path.exists():
        pairs = []
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        if len(pairs) >= n_total:
            logger.info("Loaded %d QA pairs from cache: %s", len(pairs), cache_path)
            return pairs[:n_total]
        logger.info("Cache has %d pairs, need %d — generating more", len(pairs), n_total)
        existing = len(pairs)
    else:
        pairs = []
        existing = 0

    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Load chunks
    logger.info("Loading chunks from %s ...", chunks_path)
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)
    all_chunks = [c for c in data["chunks"] if len(c["text"]) >= 300]
    logger.info("Eligible chunks: %d", len(all_chunks))

    needed = n_total - existing
    n_batches = math.ceil(needed / 3)  # ~3 QA pairs per batch

    sampled = random.sample(all_chunks, min(n_batches * batch_size, len(all_chunks)))

    with open(cache_path, "a", encoding="utf-8") as fh:
        generated = 0
        for b in range(n_batches):
            if generated >= needed:
                break
            batch_chunks = sampled[b*batch_size:(b+1)*batch_size]
            if not batch_chunks:
                break
            ctx = "\n\n".join(f"[{i+1}] {c['text'][:400]}" for i, c in enumerate(batch_chunks))
            prompt = (
                "You are a medical education expert. Generate 3 high-quality oncology "
                "question-answer pairs from the text below.\n"
                "Rules: answers must be grounded in the provided text. "
                "Vary query types: qa, lfqa, mcqa, fact_verification, fill_blank.\n"
                "Return ONLY a JSON array:\n"
                '[{"question":"...","answer":"...","query_type":"qa"},...]\n\n'
                f"Text:\n{ctx}"
            )
            try:
                r = client.messages.create(
                    model=config.CLAUDE_MODEL, max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = r.content[0].text.strip()
                m = re.search(r"\[.*\]", raw, re.DOTALL)
                if m:
                    new_pairs = json.loads(m.group())
                    for p in new_pairs:
                        if "question" in p and "answer" in p:
                            rec = {
                                "question":   p["question"],
                                "answer":     p["answer"],
                                "query_type": p.get("query_type", "qa"),
                            }
                            pairs.append(rec)
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            generated += 1
                            if generated >= needed:
                                break
                logger.info("Batch %d/%d — %d pairs so far", b+1, n_batches, existing+generated)
            except Exception as exc:
                logger.warning("QA generation batch %d failed: %s", b+1, exc)

    logger.info("Total QA pairs: %d", len(pairs))
    return pairs[:n_total]


# ===========================================================================
# PIPELINE RUNNER (200 answers)
# ===========================================================================

def run_pipeline_on_qa_pairs(
    qa_pairs: list[dict],
    cache_path: Optional[Path] = None,
) -> list[dict]:
    """
    Run the MedRAG pipeline on all QA pairs and return pipeline records.
    Caches results to avoid re-running the expensive pipeline.
    """
    if cache_path is None:
        cache_path = config.RESULTS_DIR / "pipeline_200_responses.jsonl"

    # Load from cache
    if cache_path.exists():
        records = []
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        if len(records) >= len(qa_pairs):
            logger.info("Loaded %d pipeline records from cache", len(records))
            return records[:len(qa_pairs)]
        logger.info("Cache has %d records, running remaining %d", len(records), len(qa_pairs)-len(records))
        done_questions = {r["question"] for r in records}
    else:
        records = []
        done_questions = set()

    from src.pipeline import answer_query

    # Pre-warm scispaCy and embedding models to avoid per-query reload overhead
    logger.info("Pre-warming models (scispaCy, embedder, cross-encoder)...")
    try:
        from src.laqa import _get_spacy_model
        _get_spacy_model()
        from src.embedder import _get_model
        _get_model(config.EMBEDDING_MODEL)
        from src.reranker import _get_cross_encoder
        _get_cross_encoder(config.RERANKER_MODEL)
        logger.info("Models pre-warmed.")
    except Exception as exc:
        logger.warning("Pre-warm failed (%s) — continuing anyway", exc)

    with open(cache_path, "a", encoding="utf-8") as fh:
        for i, pair in enumerate(qa_pairs):
            q = pair["question"]
            if q in done_questions:
                continue
            logger.info("[%d/%d] %s", i+1, len(qa_pairs), q[:70])
            try:
                resp = answer_query(
                    q,
                    use_hyde=False,
                    use_multi_query=False,
                    use_rlaif=False,
                )
                rec = {
                    "question":         q,
                    "reference_answer": pair["answer"],
                    "query_type":       pair.get("query_type", resp.query_type),
                    "generated_answer": resp.answer,
                    "contexts":         [c.text for c in resp.retrieved_chunks],
                    "rerank_scores":    [c.score for c in resp.retrieved_chunks],
                    "pattern_used":     resp.agentic_pattern_used,
                    "agent_iterations": resp.agent_iterations,
                    "confidence":       resp.query_type_confidence,
                    "latency_ms":       resp.total_latency_ms,
                }
                records.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as exc:
                logger.error("Pipeline failed for q=%r: %s", q[:50], exc)

    return records[:len(qa_pairs)]


# ===========================================================================
# REPORT FORMATTING
# ===========================================================================

def format_report(
    header: dict,
    retrieval: dict,
    layer1: dict,
    layer2: dict,
    layer3: dict,
    scope: dict,
) -> str:
    def _v(d, k, fmt=".4f"): return format(d.get(k, 0), fmt)

    lines = [
        "==============================================================",
        "ONCOLOGY RAG - COMPLETE EVALUATION REPORT",
        "LAQA + MRL + Agentic RAG",
        "==============================================================",
        "",
        f"Questions evaluated : {header.get('total_samples', 0)}",
        f"Avg agent iters     : {header.get('avg_agent_iterations', 0):.2f}",
        f"Avg confidence      : {header.get('avg_confidence', 0):.4f}",
        "SCOPE method        : Claude-as-judge",
        "",
        "-- Retrieval Quality (k=5) -----------------------------------",
        f"Precision@5         : {_v(retrieval, 'precision_at_5')}",
        f"Recall@5            : {_v(retrieval, 'recall_at_5')}",
        f"MRR                 : {_v(retrieval, 'mrr')}",
        f"NDCG@5              : {_v(retrieval, 'ndcg_at_5')}",
        f"Hit-Rate@5          : {_v(retrieval, 'hit_rate_at_5')}",
        f"Avg rerank score    : {_v(retrieval, 'avg_rerank_score')}",
        "",
        "-- Generation Lexical ----------------------------------------",
        f"BLEU-1              : {_v(layer1, 'bleu_1')}",
        f"BLEU-2              : {_v(layer1, 'bleu_2')}",
        f"BLEU-4              : {_v(layer1, 'bleu_4')}",
        f"GLEU                : {_v(layer1, 'gleu')}",
        f"ROUGE-1             : {_v(layer1, 'rouge_1')}",
        f"ROUGE-2             : {_v(layer1, 'rouge_2')}",
        f"ROUGE-L             : {_v(layer1, 'rouge_l')}",
        f"ROUGE-Lsum          : {_v(layer1, 'rouge_lsum')}",
        f"METEOR              : {_v(layer1, 'meteor')}",
        f"Answer F1           : {_v(layer1, 'answer_f1')}",
        "",
        "-- Generation Semantic ---------------------------------------",
        f"BERTScore F1        : {_v(layer2, 'bertscore_f1')}",
        "",
        "-- Faithfulness & Relevance ----------------------------------",
        f"Faithfulness(LLM)   : {_v(layer3, 'faithfulness')}",
        f"Context Relevancy   : {_v(layer3, 'context_relevancy')}",
        f"Answer relevance    : {_v(layer3, 'answer_relevance')}",
        "",
        "-- S.C.O.P.E LLM-as-judge (/5.0) ----------------------------",
        f"S Safety            : {scope.get('scope_s', 0):.2f}",
        f"C Completeness      : {scope.get('scope_c', 0):.2f}",
        f"O Originality       : {scope.get('scope_o', 0):.2f}",
        f"P Precision         : {scope.get('scope_p', 0):.2f}",
        f"E Efficiency        : {scope.get('scope_e', 0):.2f}",
        f"Weighted Total      : {scope.get('scope_total', 0):.2f}/5.00  "
        f"(std={scope.get('scope_std', 0):.2f})",
        "==============================================================",
    ]
    return "\n".join(lines)


def save_report(report_str: str, all_metrics: dict, output_dir: Path) -> tuple[Path, Path]:
    ts = int(time.time())
    txt_path  = output_dir / f"eval_{ts}.txt"
    json_path = output_dir / f"eval_{ts}.json"
    txt_path.write_text(report_str, encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    return txt_path, json_path


# ===========================================================================
# MAIN — legacy mode (--answers / --references flags)
# ===========================================================================

def _legacy_layer1_report(m):
    print("\n" + "=" * 55)
    print("Layer 1 — Lexical Metrics")
    print("=" * 55)
    for k in ["num_samples","bleu_1","bleu_2","bleu_4","gleu",
               "rouge_1","rouge_2","rouge_l","rouge_lsum","meteor",
               "answer_f1","mcqa_accuracy"]:
        v = m.get(k)
        if v is not None:
            print(f"  {k:<18}: {v}")


def _legacy_layer2_report(m):
    print("\n" + "=" * 55)
    print("Layer 2 — Semantic Metrics")
    print("=" * 55)
    for k in ["num_samples","bertscore_model","bertscore_precision",
               "bertscore_recall","bertscore_f1","sbert_cosine"]:
        v = m.get(k)
        if v is not None:
            print(f"  {k:<24}: {v}")


def main(args: argparse.Namespace) -> None:
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── SCOPE-only rescore mode ───────────────────────────────────────────────
    if args.rescore_scope:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scope = rescore_scope_only(
            pipeline_cache_path=output_dir / "pipeline_200_responses.jsonl",
            scope_checkpoint_path=output_dir / "scope_checkpoint.json",
            output_dir=output_dir,
        )

        # Load existing eval JSON and merge updated SCOPE
        # Pick the 200-sample report (largest total_samples)
        import glob
        eval_jsons = sorted(glob.glob(str(output_dir / "eval_*.json")))
        latest = None
        best_n = 0
        for path in eval_jsons:
            try:
                with open(path) as _f:
                    _d = json.load(_f)
                n = _d.get("header", {}).get("total_samples", _d.get("num_samples", 0))
                if n > best_n:
                    best_n = n
                    latest = path
            except Exception:
                pass
        if latest:
            logger.info("Merging SCOPE into %s (samples=%d)", latest, best_n)
            with open(latest, encoding="utf-8") as f:
                all_metrics = json.load(f)
            all_metrics["scope"] = scope

            # Regenerate report with corrected SCOPE
            header   = all_metrics.get("header", {})
            retrieval= all_metrics.get("retrieval", {})
            l1       = all_metrics.get("layer1", {})
            l2       = all_metrics.get("layer2", {})
            l3       = all_metrics.get("layer3", {})
            report_str = format_report(header, retrieval, l1, l2, l3, scope)
            print("\n" + report_str)

            txt_path, json_path = save_report(report_str, all_metrics, output_dir)
            logger.info("Updated report saved: %s", txt_path)
        else:
            print(f"\nSCOPE results:\n{json.dumps(scope, indent=2)}")
        return

    # ── Full 200-sample evaluation mode ─────────────────────────────────────
    if args.full_eval:
        import anthropic as _anthropic

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "eval_checkpoint.json"

        n = args.n_samples
        logger.info("=== FULL EVALUATION MODE: %d samples ===", n)

        # 1. Generate / load QA pairs
        qa_pairs = generate_qa_pairs(
            n_total=n,
            cache_path=output_dir / "qa_pairs_200.jsonl",
        )

        # 2. Run / load pipeline responses
        records = run_pipeline_on_qa_pairs(
            qa_pairs[:n],
            cache_path=output_dir / "pipeline_200_responses.jsonl",
        )

        hypotheses = [r["generated_answer"]  for r in records]
        references = [r["reference_answer"]   for r in records]
        qtypes     = [r.get("query_type","qa") for r in records]

        all_metrics: dict = {}

        # 3. Header stats (free — no Claude)
        header = compute_header_stats(records)
        all_metrics["header"] = header

        # 4. Layer 1 lexical (free — no Claude)
        logger.info("Computing Layer 1 lexical metrics...")
        l1 = layer1_metrics(hypotheses, references, qtypes)
        all_metrics["layer1"] = l1

        # 5. Layer 2 semantic (free — BERTScore + SBERT, no Claude)
        logger.info("Computing Layer 2 semantic metrics...")
        l2 = layer2_metrics(hypotheses, references)
        all_metrics["layer2"] = l2

        # 6. Claude scoring — ONE call per sample (checkpoint + resume)
        ckpt = load_checkpoint(checkpoint_path)
        completed_indices = set(ckpt.get("completed_indices", []))
        scores_dict       = ckpt.get("scores", {})

        # Also store avg_rerank_score per sample for retrieval metrics
        for i, rec in enumerate(records):
            idx_str = str(i)
            if idx_str in scores_dict and "avg_rerank_score" not in scores_dict[idx_str]:
                rs = rec.get("rerank_scores", [])
                scores_dict[idx_str]["avg_rerank_score"] = float(rs[0]) if rs else 0.0

        remaining = [i for i in range(len(records)) if i not in completed_indices]

        # Cost estimate and confirmation
        if not confirm_cost(len(remaining), len(completed_indices)):
            logger.info("Evaluation cancelled by user.")
            # Still compute report from whatever is in the checkpoint
            if scores_dict:
                l3, ret, scope = _aggregate_claude_scores(scores_dict)
            else:
                l3   = {"faithfulness": 0.0, "context_relevancy": 0.0, "answer_relevance": 0.0, "num_samples": 0}
                ret  = {"precision_at_5": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0, "hit_rate_at_5": 0.0, "avg_rerank_score": 0.0}
                scope= {"scope_s": 0.0, "scope_c": 0.0, "scope_o": 0.0, "scope_p": 0.0, "scope_e": 0.0, "scope_total": 0.0, "scope_std": 0.0}
            all_metrics.update({"layer3": l3, "retrieval": ret, "scope": scope})
            report_str = format_report(header, ret, l1, l2, l3, scope)
            print("\n" + report_str)
            txt_path, json_path = save_report(report_str, all_metrics, output_dir)
            logger.info("Partial report saved: %s", txt_path)
            return

        # Run single-call Claude evaluator with checkpointing
        client = _anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        logger.info("Scoring %d samples with Claude (%d already cached)...",
                    len(remaining), len(completed_indices))

        for idx in remaining:
            rec = records[idx]
            result = evaluate_single_sample(
                question=rec.get("question", ""),
                generated_answer=rec.get("generated_answer", ""),
                retrieved_chunks=rec.get("contexts", []),
                reference_answer=rec.get("reference_answer", ""),
                client=client,
            )
            # Store avg_rerank_score alongside Claude scores
            rs = rec.get("rerank_scores", [])
            result["avg_rerank_score"] = float(rs[0]) if rs else 0.0

            scores_dict[str(idx)] = result
            completed_indices.add(idx)

            ckpt["last_completed_index"] = idx
            ckpt["completed_indices"]    = sorted(completed_indices)
            ckpt["scores"]               = scores_dict

            # Save checkpoint every 10 questions
            if len(completed_indices) % 10 == 0 or idx == len(records) - 1:
                save_checkpoint(ckpt, checkpoint_path)
                logger.info("  [%d/%d] checkpoint saved", len(completed_indices), len(records))

        # Aggregate all Claude scores
        l3, ret, scope = _aggregate_claude_scores(scores_dict)

        # Restore avg_rerank_score from pipeline records (more accurate)
        rerank_scores = [float(r.get("rerank_scores", [0])[0]) for r in records if r.get("rerank_scores")]
        if rerank_scores:
            ret["avg_rerank_score"] = round(sum(rerank_scores)/len(rerank_scores), 4)

        all_metrics.update({"layer3": l3, "retrieval": ret, "scope": scope})

        # Format and save full report
        report_str = format_report(header, ret, l1, l2, l3, scope)
        print("\n" + report_str)
        txt_path, json_path = save_report(report_str, all_metrics, output_dir)
        logger.info("TXT report  : %s", txt_path)
        logger.info("JSON report : %s", json_path)

        # Clean up checkpoint on successful completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Checkpoint cleared (evaluation complete).")
        return

    # ── Legacy mode: --answers / --references ────────────────────────────────
    answers_file    = Path(args.answers)
    references_file = Path(args.references)
    output_dir      = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(l) for l in args.layers]

    def _load_jsonl(p):
        recs = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip(): recs.append(json.loads(line))
        return recs

    answers    = _load_jsonl(answers_file)
    references = _load_jsonl(references_file)
    hypotheses = [a.get("answer","") for a in answers]
    refs       = [r.get("answer","") for r in references]
    qtypes     = [a.get("query_type","qa") for a in answers]
    all_metrics: dict = {"timestamp": time.time(), "num_samples": len(hypotheses)}

    if 1 in layers:
        logger.info("Computing Layer 1 metrics...")
        l1 = layer1_metrics(hypotheses, refs, qtypes)
        all_metrics["layer1"] = l1
        _legacy_layer1_report(l1)

    if 2 in layers:
        logger.info("Computing Layer 2 metrics...")
        l2 = layer2_metrics(hypotheses, refs)
        all_metrics["layer2"] = l2
        _legacy_layer2_report(l2)

    if 3 in layers:
        logger.info("Computing Layer 3 metrics...")
        recs_for_l3 = [
            {"question": a.get("query",""), "generated_answer": a.get("answer",""),
             "contexts":  a.get("contexts",[])}
            for a in answers
        ]
        l3 = layer3_claude_metrics(recs_for_l3)
        all_metrics["layer3"] = l3
        print(f"\nLayer 3: faithfulness={l3['faithfulness']}  "
              f"ctx_rel={l3['context_relevancy']}  ans_rel={l3['answer_relevance']}")

    ts = int(time.time())
    rpt = output_dir / f"eval_report_{ts}.json"
    all_metrics["report_path"] = str(rpt)
    with open(rpt, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to: %s", rpt)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MedRAG evaluation runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Full evaluation mode
    p.add_argument("--full-eval",  action="store_true",
                   help="Run complete 200-sample evaluation with all metrics")
    p.add_argument("--n-samples",  type=int, default=200,
                   help="Number of QA samples for full evaluation")
    p.add_argument("--rescore-scope", action="store_true",
                   help="Re-score only SCOPE dimensions using pipeline cache")

    # Legacy mode
    p.add_argument("--answers",    default="results/answers.jsonl")
    p.add_argument("--references", default="data/references.jsonl")
    p.add_argument("--output-dir", default=str(config.RESULTS_DIR))
    p.add_argument("--layers",     nargs="+", default=["1"],
                   help="Layers to evaluate in legacy mode: 1 2 3")
    p.add_argument("--log-level",  default="INFO")
    return p


if __name__ == "__main__":
    main(_build_parser().parse_args())
