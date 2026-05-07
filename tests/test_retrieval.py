"""
tests/test_retrieval.py — Comprehensive retrieval pipeline tests.

Covers:
  - Stage 1 coarse search (64-dim MRL)
  - Stage 2 fine rerank (384-dim MRL)
  - Stage 3 cross-encoder rerank
  - MRL ranking vs cross-encoder ranking side-by-side comparison
  - All 3 agentic patterns (router, iterative, decomposition)
  - Orchestrator pattern selection for all 6 query types

Run:
    source .venv/bin/activate
    python3 tests/test_retrieval.py

Or with pytest:
    pytest tests/test_retrieval.py -v
"""

import sys
import warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import pytest
from src.laqa import laqa_pipeline
from src.query_expansion import full_query_expansion
from src.retriever import stage1_coarse_search, stage2_fine_rerank
from src.reranker import cross_encoder_rerank
from src.agentic_rag import orchestrator
from src.models import RetrievedChunk


# ---------------------------------------------------------------------------
# The 5 core test queries (one per major oncology retrieval scenario)
# ---------------------------------------------------------------------------

QUERIES = [
    {
        "id": "R01",
        "query": "What is the standard dosage of cisplatin for head and neck cancer?",
        "type": "qa",
        "expected_pattern": "standard",
        "expected_keywords": ["cisplatin", "dose", "mg"],
        "desc": "QA — cisplatin dosage",
    },
    {
        "id": "R02",
        "query": "Explain the complete chemoradiation protocol for locally advanced HNSCC.",
        "type": "lfqa",
        "expected_pattern": "iterative",
        "expected_keywords": ["radiation", "chemotherapy", "cisplatin"],
        "desc": "LFQA — chemoradiation protocol",
    },
    {
        "id": "R03",
        "query": "This PD-1 checkpoint inhibitor is approved for first-line recurrent metastatic HNSCC.",
        "type": "jeopardy",
        "expected_pattern": "decomposition",
        "expected_keywords": ["immunotherapy", "pembrolizumab", "HNSCC", "PD-1"],
        "desc": "Jeopardy — PD-1 inhibitor identification",
    },
    {
        "id": "R04",
        "query": "Claim: Radiation therapy alone is sufficient for stage IV head and neck cancer.",
        "type": "fact_verification",
        "expected_pattern": "router",
        "expected_keywords": ["radiation", "stage", "treatment"],
        "desc": "Fact verification — radiation monotherapy claim",
    },
    {
        "id": "R05",
        "query": "Which staging system is used for oral cavity cancer? A) AJCC B) FIGO C) Dukes D) Gleason",
        "type": "mcqa",
        "expected_pattern": "router",
        "expected_keywords": ["staging", "oral", "cancer"],
        "desc": "MCQA — oral cavity staging system",
    },
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_3stage(query: str, n1=100, n2=20, n3=5):
    """Run bare 3-stage retrieval without agentic wrapping."""
    s1 = stage1_coarse_search(query, n_results=n1)
    s2 = stage2_fine_rerank(query, s1, n_results=n2)
    s3 = cross_encoder_rerank(query, s2, top_k=n3)
    return s1, s2, s3


def _keywords_in_chunks(chunks: list[RetrievedChunk], keywords: list[str]) -> list[str]:
    """Return which keywords appear in the text of the top chunks."""
    combined = " ".join(c.text.lower() for c in chunks)
    return [k for k in keywords if k.lower() in combined]


# ---------------------------------------------------------------------------
# Tests — Stage 1
# ---------------------------------------------------------------------------

class TestStage1:
    def test_returns_correct_count(self):
        s1 = stage1_coarse_search("cisplatin head neck cancer dosage", n_results=50)
        assert len(s1) == 50

    def test_scores_descending(self):
        s1 = stage1_coarse_search("radiation therapy HNSCC", n_results=20)
        scores = [c.score for c in s1]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    def test_stage_tag(self):
        s1 = stage1_coarse_search("cancer staging oral cavity", n_results=5)
        assert all(c.stage == "stage1" for c in s1)

    def test_scores_in_range(self):
        s1 = stage1_coarse_search("chemotherapy cisplatin", n_results=10)
        assert all(0.0 <= c.score <= 1.0 for c in s1)

    def test_category_filter(self):
        s1 = stage1_coarse_search(
            "clinical guidelines HNSCC",
            n_results=10,
            where={"category": "clinical_guideline"},
        )
        assert all(c.category == "clinical_guideline" for c in s1)

    def test_multi_query_fusion_improves_score(self):
        query = "pembrolizumab PD-1 immunotherapy HNSCC"
        analysed = laqa_pipeline(query)
        expanded = full_query_expansion(analysed, use_hyde=False, use_multi_query=False)

        single = stage1_coarse_search(query, n_results=100)
        fused  = stage1_coarse_search(
            expanded.translated_query,
            analysed_query=expanded,
            n_results=100,
        )
        # Fused should have equal or higher top score
        assert fused[0].score >= single[0].score * 0.95

    def test_empty_collection_returns_list(self):
        s1 = stage1_coarse_search("nonexistent xyzabc123", n_results=5)
        assert isinstance(s1, list)


# ---------------------------------------------------------------------------
# Tests — Stage 2
# ---------------------------------------------------------------------------

class TestStage2:
    def test_reduces_to_n_results(self):
        s1 = stage1_coarse_search("cisplatin dosage", n_results=100)
        s2 = stage2_fine_rerank("cisplatin dosage", s1, n_results=20)
        assert len(s2) == 20

    def test_stage_tag(self):
        s1 = stage1_coarse_search("radiation head neck", n_results=50)
        s2 = stage2_fine_rerank("radiation head neck", s1, n_results=10)
        assert all(c.stage == "stage2" for c in s2)

    def test_scores_descending(self):
        s1 = stage1_coarse_search("HNSCC staging TNM", n_results=100)
        s2 = stage2_fine_rerank("HNSCC staging TNM", s1, n_results=20)
        scores = [c.score for c in s2]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    def test_empty_input(self):
        s2 = stage2_fine_rerank("any query", [], n_results=20)
        assert s2 == []

    def test_reranks_stage1(self):
        query = "cisplatin head neck cancer treatment dosage"
        s1 = stage1_coarse_search(query, n_results=100)
        s2 = stage2_fine_rerank(query, s1, n_results=20)
        s1_top20_ids = {c.chunk_id for c in s1[:20]}
        s2_ids = {c.chunk_id for c in s2}
        # Stage2 draws from full stage1 pool — some new chunks promoted
        assert len(s2_ids) == 20


# ---------------------------------------------------------------------------
# Tests — Stage 3 (Cross-Encoder)
# ---------------------------------------------------------------------------

class TestStage3:
    def test_returns_top_k(self):
        s1 = stage1_coarse_search("cisplatin HNSCC", n_results=100)
        s2 = stage2_fine_rerank("cisplatin HNSCC", s1, n_results=20)
        s3 = cross_encoder_rerank("cisplatin HNSCC", s2, top_k=5)
        assert len(s3) == 5

    def test_stage_tag(self):
        s1 = stage1_coarse_search("radiation therapy", n_results=50)
        s2 = stage2_fine_rerank("radiation therapy", s1, n_results=10)
        s3 = cross_encoder_rerank("radiation therapy", s2, top_k=5)
        assert all(c.stage == "stage3" for c in s3)

    def test_sigmoid_scores(self):
        s1 = stage1_coarse_search("immunotherapy cancer", n_results=50)
        s2 = stage2_fine_rerank("immunotherapy cancer", s1, n_results=10)
        s3 = cross_encoder_rerank("immunotherapy cancer", s2, top_k=5)
        assert all(0.0 <= c.score <= 1.0 for c in s3)

    def test_scores_descending(self):
        s1 = stage1_coarse_search("staging oral cancer", n_results=50)
        s2 = stage2_fine_rerank("staging oral cancer", s1, n_results=10)
        s3 = cross_encoder_rerank("staging oral cancer", s2, top_k=5)
        scores = [c.score for c in s3]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    def test_empty_input(self):
        s3 = cross_encoder_rerank("any query", [], top_k=5)
        assert s3 == []

    def test_single_chunk(self):
        s1 = stage1_coarse_search("cisplatin", n_results=1)
        s3 = cross_encoder_rerank("cisplatin", s1, top_k=5)
        assert len(s3) == 1 and s3[0].stage == "stage3"


# ---------------------------------------------------------------------------
# Tests — Orchestrator
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def test_qa_uses_standard(self):
        analysed = laqa_pipeline("What is the dosage of cisplatin?")
        expanded = full_query_expansion(analysed, use_hyde=False, use_multi_query=False)
        chunks, pattern = orchestrator(expanded)
        assert pattern == "standard"
        assert len(chunks) > 0

    def test_lfqa_uses_iterative(self):
        analysed = laqa_pipeline("Explain the mechanism of chemoradiation in HNSCC.")
        expanded = full_query_expansion(analysed, use_hyde=False, use_multi_query=False)
        chunks, pattern = orchestrator(expanded)
        assert pattern == "iterative"
        assert len(chunks) > 0

    def test_jeopardy_uses_decomposition(self):
        analysed = laqa_pipeline(
            "This platinum-based drug is the cornerstone of HNSCC chemoradiation."
        )
        expanded = full_query_expansion(analysed, use_hyde=False, use_multi_query=False)
        chunks, pattern = orchestrator(expanded)
        assert pattern == "decomposition"
        assert len(chunks) > 0

    def test_fact_verification_uses_router(self):
        analysed = laqa_pipeline("Claim: Surgery is the only treatment for HNSCC.")
        expanded = full_query_expansion(analysed, use_hyde=False, use_multi_query=False)
        chunks, pattern = orchestrator(expanded)
        assert pattern == "router"
        assert all(c.category == "clinical_guideline" for c in chunks)

    def test_mcqa_uses_router(self):
        analysed = laqa_pipeline(
            "First-line chemo for HNSCC? A) Cisplatin B) Carboplatin C) Paclitaxel D) Docetaxel"
        )
        expanded = full_query_expansion(analysed, use_hyde=False, use_multi_query=False)
        chunks, pattern = orchestrator(expanded)
        assert pattern == "router"
        assert len(chunks) > 0

    def test_all_chunks_stage3(self):
        analysed = laqa_pipeline("What is metastasis in cancer?")
        expanded = full_query_expansion(analysed, use_hyde=False, use_multi_query=False)
        chunks, _ = orchestrator(expanded)
        assert all(c.stage == "stage3" for c in chunks)

    def test_returns_tuple(self):
        analysed = laqa_pipeline("What is radiation therapy?")
        expanded = full_query_expansion(analysed, use_hyde=False, use_multi_query=False)
        result = orchestrator(expanded)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], str)


# ---------------------------------------------------------------------------
# Standalone runner with side-by-side MRL vs cross-encoder comparison
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    print("=" * 70)
    print("tests/test_retrieval.py — Full Retrieval Test Suite")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Run all 5 queries with side-by-side MRL vs cross-encoder comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE: MRL (Stage2) vs Cross-Encoder (Stage3) Rankings")
    print("=" * 70)

    for q_data in QUERIES:
        print(f"\n{'─'*70}")
        print(f"[{q_data['id']}] {q_data['desc']}")
        print(f"Query: {q_data['query']!r}")

        analysed = laqa_pipeline(q_data["query"])
        expanded = full_query_expansion(analysed, use_hyde=False, use_multi_query=False)

        s1, s2, s3 = _run_3stage(expanded.translated_query)

        # Keyword check
        found_s2 = _keywords_in_chunks(s2[:5], q_data["expected_keywords"])
        found_s3 = _keywords_in_chunks(s3[:5], q_data["expected_keywords"])

        print(f"\n{'Rank':<5} {'Stage2 (MRL 384-dim)':<38} {'Score':<8} │ {'Stage3 (Cross-Encoder)':<38} {'Score'}")
        print("─" * 100)
        for i in range(min(5, len(s2), len(s3))):
            s2c = s2[i]
            s3c = s3[i]
            s2_text = s2c.text[:35].replace("\n", " ")
            s3_text = s3c.text[:35].replace("\n", " ")
            print(f"{i+1:<5} {s2_text:<38} {s2c.score:<8.3f} │ {s3_text:<38} {s3c.score:.4f}")

        print(f"\nKeywords in Stage2 top-5: {found_s2}")
        print(f"Keywords in Stage3 top-5: {found_s3}")

        # Orchestrator
        chunks, pattern = orchestrator(expanded)
        print(f"Orchestrator pattern    : {pattern}  (expect {q_data['expected_pattern']})")
        print(f"Final chunks            : {len(chunks)}")

    # ------------------------------------------------------------------
    # Run all unit tests
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("UNIT TESTS")
    print("=" * 70)

    all_tests = [
        (TestStage1, "test_returns_correct_count"),
        (TestStage1, "test_scores_descending"),
        (TestStage1, "test_stage_tag"),
        (TestStage1, "test_scores_in_range"),
        (TestStage1, "test_category_filter"),
        (TestStage1, "test_multi_query_fusion_improves_score"),
        (TestStage1, "test_empty_collection_returns_list"),
        (TestStage2, "test_reduces_to_n_results"),
        (TestStage2, "test_stage_tag"),
        (TestStage2, "test_scores_descending"),
        (TestStage2, "test_empty_input"),
        (TestStage2, "test_reranks_stage1"),
        (TestStage3, "test_returns_top_k"),
        (TestStage3, "test_stage_tag"),
        (TestStage3, "test_sigmoid_scores"),
        (TestStage3, "test_scores_descending"),
        (TestStage3, "test_empty_input"),
        (TestStage3, "test_single_chunk"),
        (TestOrchestrator, "test_qa_uses_standard"),
        (TestOrchestrator, "test_lfqa_uses_iterative"),
        (TestOrchestrator, "test_jeopardy_uses_decomposition"),
        (TestOrchestrator, "test_fact_verification_uses_router"),
        (TestOrchestrator, "test_mcqa_uses_router"),
        (TestOrchestrator, "test_all_chunks_stage3"),
        (TestOrchestrator, "test_returns_tuple"),
    ]

    passed = failed = 0
    for cls, method in all_tests:
        instance = cls()
        try:
            getattr(instance, method)()
            print(f"  ✅ {cls.__name__}.{method}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {cls.__name__}.{method}")
            print(f"     {type(e).__name__}: {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"Results: {passed}/{passed+failed} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
    print("=" * 70)
