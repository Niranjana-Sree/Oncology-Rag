"""
tests/test_laqa.py — Comprehensive LAQA pipeline tests.

Covers:
  - All 6 query types (qa, mcqa, lfqa, jeopardy, fact_verification, fill_blank)
  - All 5 languages (en, hi, ta, te, hinglish)
  - Full laqa_pipeline() + full_query_expansion() integration
  - Edge cases (empty query, very short query, mixed script)

Run:
    source .venv/bin/activate
    python3 tests/test_laqa.py

Or with pytest:
    pytest tests/test_laqa.py -v
"""

import sys
import warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import pytest
from src.laqa import detect_language, classify_query_type, extract_medical_entities, laqa_pipeline
from src.query_expansion import translate_to_english, expand_synonyms, full_query_expansion


# ---------------------------------------------------------------------------
# Fixtures — the 10 core test queries
# ---------------------------------------------------------------------------

QUERIES = [
    # 1. English QA — drug dosage
    {
        "id": "Q01",
        "query": "What is the standard dosage of cisplatin for head and neck cancer?",
        "lang": "en",
        "type": "qa",
        "must_entities": ["cisplatin"],
        "must_synonyms": ["CDDP", "HNSCC"],
        "desc": "English QA — cisplatin dosage",
    },
    # 2. English MCQA
    {
        "id": "Q02",
        "query": "Which drug is first-line for locally advanced HNSCC? A) Cisplatin B) Carboplatin C) Paclitaxel D) Cetuximab",
        "lang": "en",
        "type": "mcqa",
        "must_entities": ["HNSCC"],
        "must_synonyms": ["CDDP"],
        "desc": "English MCQA — first-line HNSCC",
    },
    # 3. English LFQA
    {
        "id": "Q03",
        "query": "Explain the mechanism of action of pembrolizumab in treating head and neck squamous cell carcinoma.",
        "lang": "en",
        "type": "lfqa",
        "must_entities": ["pembrolizumab"],
        "must_synonyms": ["Keytruda", "PD-1 inhibitor"],
        "desc": "English LFQA — pembrolizumab mechanism",
    },
    # 4. English Jeopardy
    {
        "id": "Q04",
        "query": "This platinum-based chemotherapy is the cornerstone of treatment for locally advanced head and neck cancer.",
        "lang": "en",
        "type": "jeopardy",
        "must_entities": ["chemotherapy"],
        "must_synonyms": ["chemo", "cytotoxic therapy"],
        "desc": "English Jeopardy — platinum chemotherapy",
    },
    # 5. English Fact Verification
    {
        "id": "Q05",
        "query": "Claim: Pembrolizumab is approved as first-line treatment for recurrent metastatic HNSCC.",
        "lang": "en",
        "type": "fact_verification",
        "must_entities": ["Pembrolizumab"],
        "must_synonyms": ["Keytruda"],
        "desc": "English Fact Verification — pembrolizumab approval",
    },
    # 6. English Fill-in-blank
    {
        "id": "Q06",
        "query": "The standard chemotherapy agent combined with radiation for locally advanced HNSCC is ___.",
        "lang": "en",
        "type": "fill_blank",
        "must_entities": ["chemotherapy", "radiation"],
        "must_synonyms": ["chemo", "radiotherapy"],
        "desc": "English Fill-in-blank — HNSCC chemoradiation",
    },
    # 7. Hindi query
    {
        "id": "Q07",
        "query": "सिसप्लैटिन की खुराक क्या है?",
        "lang": "hi",
        "type": "qa",
        "must_entities": [],      # NER skipped for non-English
        "must_synonyms": [],      # synonyms run on translated text
        "desc": "Hindi QA — cisplatin dosage",
        "translated_must_contain": ["cisplatin", "dosage"],
    },
    # 8. Tamil query
    {
        "id": "Q08",
        "query": "தலை மற்றும் கழுத்து புற்றுநோய் என்றால் என்ன?",
        "lang": "ta",
        "type": "qa",
        "must_entities": [],
        "must_synonyms": [],
        "desc": "Tamil QA — head and neck cancer",
        "translated_must_contain": ["head", "neck", "cancer"],
    },
    # 9. Telugu query
    {
        "id": "Q09",
        "query": "కాన్సర్ చికిత్స ఏమిటి?",
        "lang": "te",
        "type": "qa",
        "must_entities": [],
        "must_synonyms": [],
        "desc": "Telugu QA — cancer treatment",
        "translated_must_contain": ["cancer", "treatment"],
    },
    # 10. Hinglish query
    {
        "id": "Q10",
        "query": "Cancer ka ilaj kya hai doctor?",
        "lang": "hinglish",
        "type": "qa",
        "must_entities": [],
        "must_synonyms": [],
        "desc": "Hinglish QA — cancer treatment",
        "translated_must_contain": ["cancer"],
    },
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check(condition: bool, msg: str) -> tuple[bool, str]:
    return condition, msg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_english(self):
        assert detect_language("What is the dosage of cisplatin?") == "en"

    def test_hindi_devanagari(self):
        assert detect_language("सिसप्लैटिन की खुराक क्या है?") == "hi"

    def test_tamil(self):
        assert detect_language("தலை மற்றும் கழுத்து புற்றுநோய் என்றால் என்ன?") == "ta"

    def test_telugu(self):
        assert detect_language("కాన్సర్ చికిత్స ఏమిటి?") == "te"

    def test_hinglish(self):
        assert detect_language("Cancer ka ilaj kya hai doctor?") == "hinglish"

    def test_empty(self):
        assert detect_language("") == "unknown"

    def test_hinglish_medical(self):
        assert detect_language("Mujhe bimari ke symptoms bataiye") == "hinglish"


class TestClassifyQueryType:
    def test_qa(self):
        qt, conf = classify_query_type("What is the dosage of cisplatin?")
        assert qt == "qa"
        assert conf >= 0.80

    def test_mcqa(self):
        qt, conf = classify_query_type(
            "Which is first-line? A) Cisplatin B) Carboplatin C) Paclitaxel D) Docetaxel"
        )
        assert qt == "mcqa"
        assert conf == 0.99

    def test_lfqa(self):
        qt, conf = classify_query_type(
            "Explain the mechanism of action of pembrolizumab in HNSCC."
        )
        assert qt == "lfqa"
        assert conf >= 0.80

    def test_jeopardy(self):
        qt, conf = classify_query_type(
            "This PD-1 inhibitor is approved for recurrent metastatic HNSCC."
        )
        assert qt == "jeopardy"
        assert conf >= 0.80

    def test_fact_verification(self):
        qt, conf = classify_query_type("Claim: Cisplatin is nephrotoxic.")
        assert qt == "fact_verification"
        assert conf == 0.99

    def test_fill_blank(self):
        qt, conf = classify_query_type(
            "The antidote for heparin overdose is ___."
        )
        assert qt == "fill_blank"
        assert conf == 0.99

    def test_empty(self):
        qt, conf = classify_query_type("")
        assert qt == "qa"
        assert conf == 0.0


class TestExtractMedicalEntities:
    def test_drug_entity(self):
        entities = extract_medical_entities("Cisplatin is used for head and neck cancer.")
        texts = [e.text.lower() for e in entities]
        assert any("cisplatin" in t for t in texts)

    def test_drug_label(self):
        entities = extract_medical_entities("Pembrolizumab treats HNSCC.")
        labels = {e.label for e in entities}
        assert "DRUG" in labels

    def test_disease_label(self):
        entities = extract_medical_entities(
            "Metastatic colorectal cancer requires systemic chemotherapy."
        )
        labels = {e.label for e in entities}
        assert "DISEASE" in labels or "ENTITY" in labels

    def test_treatment_label(self):
        entities = extract_medical_entities(
            "Radiation therapy is combined with cisplatin."
        )
        labels = {e.label for e in entities}
        assert "TREATMENT" in labels or "ENTITY" in labels

    def test_empty(self):
        entities = extract_medical_entities("")
        assert entities == []

    def test_no_noise_entities(self):
        entities = extract_medical_entities("The patient is well.")
        # "The", "patient", "is", "well" should mostly be filtered
        entity_texts = {e.text.lower() for e in entities}
        assert "the" not in entity_texts
        assert "is" not in entity_texts


class TestLaqaPipeline:
    def test_full_english_qa(self):
        result = laqa_pipeline(
            "What is the standard dosage of cisplatin for head and neck cancer?"
        )
        assert result.detected_language == "en"
        assert result.query_type == "qa"
        assert result.query_type_confidence >= 0.80
        assert result.original_query != ""
        assert result.translated_query != ""
        assert isinstance(result.medical_entities, list)
        assert isinstance(result.entity_texts, list)
        assert result.expanded_queries == []  # filled by full_query_expansion

    def test_hindi_query(self):
        result = laqa_pipeline("सिसप्लैटिन की खुराक क्या है?")
        assert result.detected_language == "hi"
        assert result.query_type == "qa"
        assert result.medical_entities == []  # NER skipped for non-English

    def test_hinglish_query(self):
        result = laqa_pipeline("Cancer ka ilaj kya hai doctor?")
        assert result.detected_language == "hinglish"
        assert result.query_type == "qa"

    def test_mcqa_detection(self):
        result = laqa_pipeline(
            "Which drug is first-line? A) Cisplatin B) Carboplatin C) Paclitaxel D) Cetuximab"
        )
        assert result.query_type == "mcqa"

    def test_empty_query(self):
        result = laqa_pipeline("")
        assert result.detected_language == "unknown"
        assert result.original_query == ""

    def test_force_overrides(self):
        result = laqa_pipeline(
            "Some query",
            force_language="hi",
            force_query_type="mcqa",
        )
        assert result.detected_language == "hi"
        assert result.query_type == "mcqa"
        assert result.query_type_confidence == 1.0


class TestFullQueryExpansion:
    def test_english_synonym_expansion(self):
        analysed = laqa_pipeline(
            "What is the dosage of cisplatin for head and neck cancer?"
        )
        expanded = full_query_expansion(analysed, use_hyde=False, use_multi_query=False)
        assert expanded.translated_query != ""
        assert len(expanded.expanded_queries) > 0
        all_syns = " ".join(expanded.expanded_queries).lower()
        assert "cddp" in all_syns or "cisplatinum" in all_syns

    def test_hindi_translation_and_expansion(self):
        analysed = laqa_pipeline("सिसप्लैटिन की खुराक क्या है?")
        expanded = full_query_expansion(analysed, use_hyde=False, use_multi_query=False)
        assert "cisplatin" in expanded.translated_query.lower()
        assert len(expanded.entity_texts) > 0  # re-extracted after translation

    def test_return_type(self):
        from src.models import AnalysedQuery
        analysed = laqa_pipeline("What is metastasis?")
        expanded = full_query_expansion(analysed, use_hyde=False, use_multi_query=False)
        assert isinstance(expanded, AnalysedQuery)
        assert isinstance(expanded.expanded_queries, list)

    def test_empty_query_safe(self):
        analysed = laqa_pipeline("")
        expanded = full_query_expansion(analysed, use_hyde=False, use_multi_query=False)
        assert isinstance(expanded.expanded_queries, list)


# ---------------------------------------------------------------------------
# Standalone runner (no pytest needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    all_tests = [
        # (test_class, method_name)
        (TestDetectLanguage, "test_english"),
        (TestDetectLanguage, "test_hindi_devanagari"),
        (TestDetectLanguage, "test_tamil"),
        (TestDetectLanguage, "test_telugu"),
        (TestDetectLanguage, "test_hinglish"),
        (TestDetectLanguage, "test_empty"),
        (TestDetectLanguage, "test_hinglish_medical"),
        (TestClassifyQueryType, "test_qa"),
        (TestClassifyQueryType, "test_mcqa"),
        (TestClassifyQueryType, "test_lfqa"),
        (TestClassifyQueryType, "test_jeopardy"),
        (TestClassifyQueryType, "test_fact_verification"),
        (TestClassifyQueryType, "test_fill_blank"),
        (TestClassifyQueryType, "test_empty"),
        (TestExtractMedicalEntities, "test_drug_entity"),
        (TestExtractMedicalEntities, "test_drug_label"),
        (TestExtractMedicalEntities, "test_disease_label"),
        (TestExtractMedicalEntities, "test_treatment_label"),
        (TestExtractMedicalEntities, "test_empty"),
        (TestExtractMedicalEntities, "test_no_noise_entities"),
        (TestLaqaPipeline, "test_full_english_qa"),
        (TestLaqaPipeline, "test_hindi_query"),
        (TestLaqaPipeline, "test_hinglish_query"),
        (TestLaqaPipeline, "test_mcqa_detection"),
        (TestLaqaPipeline, "test_empty_query"),
        (TestLaqaPipeline, "test_force_overrides"),
        (TestFullQueryExpansion, "test_english_synonym_expansion"),
        (TestFullQueryExpansion, "test_hindi_translation_and_expansion"),
        (TestFullQueryExpansion, "test_return_type"),
        (TestFullQueryExpansion, "test_empty_query_safe"),
    ]

    print("=" * 65)
    print("tests/test_laqa.py — Full LAQA Test Suite")
    print("=" * 65)

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
    print("=" * 65)
    print(f"Results: {passed}/{passed+failed} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
    print("=" * 65)
