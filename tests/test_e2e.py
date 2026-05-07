"""
tests/test_e2e.py — End-to-end API integration tests for MedRAG.

Sends 10 queries through the running FastAPI server and validates:
  - HTTP status codes
  - Response schema correctness
  - Query type detection accuracy
  - Language detection accuracy
  - Agentic pattern selection
  - Answer quality (non-empty, min length, safety)
  - Source citation presence
  - RL score validity

Prerequisites:
    Start the API server first:
        source .venv/bin/activate
        uvicorn api:app --host 0.0.0.0 --port 8000

Run:
    source .venv/bin/activate
    python3 tests/test_e2e.py [--base-url http://localhost:8000]

Or with pytest (requires running server):
    pytest tests/test_e2e.py -v
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

BASE_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# 10 end-to-end test cases
# ---------------------------------------------------------------------------

E2E_TESTS = [
    # ── English QA ─────────────────────────────────────────────────────────
    {
        "id":           "E01",
        "desc":         "English QA — cisplatin dosage",
        "payload":      {
            "query":      "What is the standard dosage of cisplatin for head and neck cancer?",
            "language":   "auto",
            "query_type": "auto",
            "top_k":      3,
        },
        "expect_lang":    "en",
        "expect_type":    "qa",
        "expect_pattern": "standard",
        "expect_keywords":["cisplatin", "mg"],
        "min_answer_len": 100,
    },
    # ── English MCQA ────────────────────────────────────────────────────────
    {
        "id":           "E02",
        "desc":         "English MCQA — first-line HNSCC drug",
        "payload":      {
            "query":      "Which drug is first-line for locally advanced HNSCC? A) Cisplatin B) Carboplatin C) Paclitaxel D) Cetuximab",
            "language":   "auto",
            "query_type": "auto",
            "top_k":      3,
        },
        "expect_lang":    "en",
        "expect_type":    "mcqa",
        "expect_pattern": "router",
        "expect_keywords":["cisplatin", "A)"],
        "min_answer_len": 50,
    },
    # ── English LFQA ────────────────────────────────────────────────────────
    {
        "id":           "E03",
        "desc":         "English LFQA — pembrolizumab mechanism",
        "payload":      {
            "query":      "Explain the mechanism of action of pembrolizumab in treating HNSCC.",
            "language":   "auto",
            "query_type": "auto",
            "top_k":      5,
        },
        "expect_lang":    "en",
        "expect_type":    "lfqa",
        "expect_pattern": "iterative",
        "expect_keywords":["PD-1", "pembrolizumab", "immune"],
        "min_answer_len": 200,
    },
    # ── Fact Verification ──────────────────────────────────────────────────
    {
        "id":           "E04",
        "desc":         "Fact verification → router (clinical_guideline)",
        "payload":      {
            "query":      "Claim: Radiation therapy alone is sufficient for stage IV HNSCC.",
            "language":   "auto",
            "query_type": "auto",
            "top_k":      3,
        },
        "expect_lang":    "en",
        "expect_type":    "fact_verification",
        "expect_pattern": "router",
        "expect_keywords":["FALSE", "radiation", "chemotherapy"],
        "min_answer_len": 100,
        "expect_category":"clinical_guideline",
    },
    # ── Fill-in-blank ──────────────────────────────────────────────────────
    {
        "id":           "E05",
        "desc":         "Fill in blank — chemoradiation agent",
        "payload":      {
            "query":      "The standard chemotherapy agent combined with radiation for locally advanced HNSCC is ___.",
            "language":   "auto",
            "query_type": "auto",
            "top_k":      3,
        },
        "expect_lang":    "en",
        "expect_type":    "fill_blank",
        "expect_pattern": "router",
        "expect_keywords":["cisplatin"],
        "min_answer_len": 50,
    },
    # ── Jeopardy ───────────────────────────────────────────────────────────
    {
        "id":           "E06",
        "desc":         "Jeopardy → decomposition pattern",
        "payload":      {
            "query":      "This platinum-based chemotherapy is the backbone of concurrent chemoradiation for head and neck cancer.",
            "language":   "auto",
            "query_type": "auto",
            "top_k":      3,
        },
        "expect_lang":    "en",
        "expect_type":    "jeopardy",
        "expect_pattern": "decomposition",
        "expect_keywords":["cisplatin"],
        "min_answer_len": 50,
    },
    # ── Hindi ──────────────────────────────────────────────────────────────
    {
        "id":           "E07",
        "desc":         "Hindi QA — cisplatin dosage",
        "payload":      {
            "query":      "सिसप्लैटिन की खुराक क्या है?",
            "language":   "auto",
            "query_type": "auto",
            "top_k":      3,
        },
        "expect_lang":    "hi",
        "expect_type":    "qa",
        "expect_pattern": None,   # any pattern
        "expect_keywords":["cisplatin", "mg"],
        "min_answer_len": 50,
    },
    # ── Tamil ──────────────────────────────────────────────────────────────
    {
        "id":           "E08",
        "desc":         "Tamil QA — head and neck cancer",
        "payload":      {
            "query":      "தலை மற்றும் கழுத்து புற்றுநோய் சிகிச்சை என்ன?",
            "language":   "auto",
            "query_type": "auto",
            "top_k":      3,
        },
        "expect_lang":    "ta",
        "expect_type":    "qa",
        "expect_pattern": None,
        "expect_keywords":["cancer", "treatment", "radiation"],
        "min_answer_len": 50,
    },
    # ── Hinglish ───────────────────────────────────────────────────────────
    {
        "id":           "E09",
        "desc":         "Hinglish QA — cancer treatment",
        "payload":      {
            "query":      "Cancer ka ilaj kya hai doctor?",
            "language":   "auto",
            "query_type": "auto",
            "top_k":      3,
        },
        "expect_lang":    "hinglish",
        "expect_type":    "qa",
        "expect_pattern": None,
        "expect_keywords":["cancer", "treatment"],
        "min_answer_len": 50,
    },
    # ── Forced pattern override ─────────────────────────────────────────────
    {
        "id":           "E10",
        "desc":         "Force iterative pattern override",
        "payload":      {
            "query":          "What are the staging criteria and treatment protocols for oral cavity cancer?",
            "language":       "en",   # force English to avoid langdetect quirks
            "query_type":     "auto",
            "top_k":          5,
            "agentic_pattern":"iterative",
        },
        "expect_lang":    "en",
        "expect_type":    None,   # any type
        "expect_pattern": "iterative",
        "expect_keywords":["staging", "cancer"],
        "min_answer_len": 100,
    },
]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _send_query(payload: dict, base_url: str, timeout: int = 120) -> dict:
    """Send a POST /api/query request and return parsed JSON."""
    import urllib.request
    import urllib.error

    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        f"{base_url}/api/query",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _check_health(base_url: str) -> bool:
    """Return True if the API server is reachable and healthy."""
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except Exception:
        return False


def run_e2e_tests(base_url: str = BASE_URL) -> tuple[int, int]:
    """
    Run all 10 E2E tests and print pass/fail for each.
    Returns (passed, failed).
    """
    print("=" * 70)
    print("MedRAG End-to-End Test Suite")
    print(f"API: {base_url}")
    print("=" * 70)

    # Verify server is up
    print("\nChecking API health...")
    if not _check_health(base_url):
        print(f"❌ API server not reachable at {base_url}")
        print("   Start the server: uvicorn api:app --host 0.0.0.0 --port 8000")
        return 0, len(E2E_TESTS)

    import urllib.request
    with urllib.request.urlopen(f"{base_url}/api/health") as r:
        health = json.loads(r.read())
    print(f"✅ API online — chunks: {health['chunks_128_count']:,}  "
          f"uptime: {health['uptime_seconds']:.0f}s")

    passed = 0
    failed = 0
    results = []

    for i, test in enumerate(E2E_TESTS, 1):
        print(f"\n[{i:02d}/{len(E2E_TESTS)}] {test['desc']}")
        print(f"      Query: {test['payload']['query'][:70]!r}")

        t0 = time.time()
        checks: list[tuple[bool, str]] = []

        try:
            response = _send_query(test["payload"], base_url, timeout=180)
            latency  = time.time() - t0

            # Check 1: HTTP success (no exception = 200)
            checks.append((True, "HTTP 200 OK"))

            # Check 2: Required fields present
            required = ["answer", "query_type", "language", "sources",
                        "agentic_pattern_used", "latency_ms", "model_used"]
            missing  = [f for f in required if f not in response]
            checks.append((not missing, f"Schema fields present (missing={missing})"))

            # Check 3: Language detection
            if test["expect_lang"]:
                lang_ok = response.get("language") == test["expect_lang"]
                checks.append((lang_ok,
                    f"Language={response.get('language')}  (expect {test['expect_lang']})"))

            # Check 4: Query type detection
            if test["expect_type"]:
                type_ok = response.get("query_type") == test["expect_type"]
                checks.append((type_ok,
                    f"QueryType={response.get('query_type')}  (expect {test['expect_type']})"))

            # Check 5: Agentic pattern
            if test["expect_pattern"]:
                pat_ok = response.get("agentic_pattern_used") == test["expect_pattern"]
                checks.append((pat_ok,
                    f"Pattern={response.get('agentic_pattern_used')}  (expect {test['expect_pattern']})"))

            # Check 6: Answer length
            answer     = response.get("answer", "")
            length_ok  = len(answer) >= test["min_answer_len"]
            checks.append((length_ok,
                f"Answer length={len(answer)}  (min={test['min_answer_len']})"))

            # Check 7: Keywords in answer
            answer_lower = answer.lower()
            found_kws = [kw for kw in test["expect_keywords"]
                         if kw.lower() in answer_lower]
            kw_ok = len(found_kws) >= max(1, len(test["expect_keywords"]) // 2)
            checks.append((kw_ok,
                f"Keywords found={found_kws}/{test['expect_keywords']}"))

            # Check 8: Sources present
            sources = response.get("sources", [])
            src_ok  = len(sources) > 0
            checks.append((src_ok, f"Sources={len(sources)} chunks returned"))

            # Check 9: Source category if specified
            if test.get("expect_category"):
                cat_ok = all(
                    s["category"] == test["expect_category"] for s in sources
                )
                checks.append((cat_ok,
                    f"Category={[s['category'] for s in sources]}  "
                    f"(expect all {test['expect_category']})"))

            # Check 10: RL score valid
            rl = response.get("rl_score")
            rl_ok = rl is None or (isinstance(rl, (int, float)) and 0.0 <= rl <= 1.0)
            checks.append((rl_ok, f"RL score={rl} valid"))

            all_ok = all(ok for ok, _ in checks)

            # Print result
            status = "✅ PASS" if all_ok else "❌ FAIL"
            rl_str = f"{rl:.3f}" if rl is not None else "N/A"
            print(f"      {status}  latency={latency:.1f}s  "
                  f"rl={rl_str}  "
                  f"model={response.get('model_used','?')}")

            for ok, msg in checks:
                if not ok:
                    print(f"        ✗ {msg}")
                else:
                    print(f"        ✓ {msg}")

            if all_ok:
                passed += 1
            else:
                failed += 1

            results.append({
                "id":      test["id"],
                "desc":    test["desc"],
                "passed":  all_ok,
                "latency": round(latency, 1),
                "checks":  [(ok, msg) for ok, msg in checks],
            })

        except Exception as exc:
            latency = time.time() - t0
            print(f"      ❌ FAIL  Exception: {exc}")
            failed += 1
            results.append({
                "id":     test["id"],
                "desc":   test["desc"],
                "passed": False,
                "error":  str(exc),
            })

    # Final summary
    print()
    print("=" * 70)
    print("END-TO-END TEST SUMMARY")
    print("=" * 70)
    for r in results:
        status = "✅" if r["passed"] else "❌"
        latency_str = f"{r.get('latency', '?')}s" if "latency" in r else "ERROR"
        print(f"  {status} [{r['id']}] {r['desc']}  ({latency_str})")

    print()
    print(f"Results: {passed}/{len(E2E_TESTS)} passed, {failed} failed")
    if failed == 0:
        print("ALL E2E TESTS PASSED ✅")
    else:
        print("SOME E2E TESTS FAILED ❌")

    # Check API metrics after tests
    print()
    try:
        import urllib.request
        with urllib.request.urlopen(f"{base_url}/api/metrics") as r:
            metrics = json.loads(r.read())
        print("API Metrics after test run:")
        print(f"  Total queries      : {metrics['total_queries']}")
        print(f"  Avg latency        : {metrics['avg_latency_ms']:.0f}ms")
        print(f"  Avg RL score       : {metrics['avg_rl_score']:.3f}")
        print(f"  Query type dist    : {metrics['query_type_distribution']}")
        print(f"  Language dist      : {metrics['language_distribution']}")
    except Exception as exc:
        print(f"Could not fetch metrics: {exc}")

    print("=" * 70)
    return passed, failed


# ---------------------------------------------------------------------------
# pytest compatibility
# ---------------------------------------------------------------------------

def test_e2e_all_pass():
    """pytest entry point — fails if any E2E test fails."""
    passed, failed = run_e2e_tests(BASE_URL)
    assert failed == 0, f"{failed} E2E test(s) failed"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedRAG E2E test suite")
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="API base URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    BASE_URL = args.base_url

    passed, failed = run_e2e_tests(BASE_URL)
    sys.exit(0 if failed == 0 else 1)
