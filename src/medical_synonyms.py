"""
Oncology-focused medical synonym dictionary for MedRAG.
Used by query_expansion.py to broaden retrieval coverage.

Structure:
    SYNONYMS: dict[canonical_term -> list[synonyms]]
    REVERSE_MAP: dict[any_term -> canonical_term]  (auto-built at import)

All keys and values are lowercase for case-insensitive matching.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("medrag.medical_synonyms")

# ---------------------------------------------------------------------------
# Master synonym dictionary
# canonical term → list of equivalent / related terms
# ---------------------------------------------------------------------------

SYNONYMS: dict[str, list[str]] = {

    # -----------------------------------------------------------------------
    # Cancer types — general
    # -----------------------------------------------------------------------
    "cancer": [
        "malignancy", "malignant neoplasm", "tumor", "tumour",
        "carcinoma", "neoplasm", "oncological disease", "malignant disease",
    ],
    "solid tumor": [
        "solid tumour", "solid malignancy", "solid cancer",
    ],
    "carcinoma": [
        "epithelial cancer", "epithelial malignancy",
    ],

    # -----------------------------------------------------------------------
    # Cancer types — specific
    # -----------------------------------------------------------------------
    "breast cancer": [
        "breast carcinoma", "mammary carcinoma", "breast malignancy",
        "breast neoplasm", "BC", "carcinoma of the breast",
    ],
    "lung cancer": [
        "lung carcinoma", "pulmonary carcinoma", "pulmonary malignancy",
        "bronchogenic carcinoma", "NSCLC", "SCLC",
        "non-small cell lung cancer", "small cell lung cancer",
    ],
    "colorectal cancer": [
        "colon cancer", "rectal cancer", "bowel cancer",
        "colorectal carcinoma", "CRC", "colon carcinoma",
        "carcinoma of the colon", "large bowel cancer",
    ],
    "prostate cancer": [
        "prostate carcinoma", "prostatic carcinoma",
        "prostatic malignancy", "PCa", "carcinoma of the prostate",
    ],
    "head and neck cancer": [
        "head and neck malignancy", "head and neck squamous cell carcinoma",
        "HNSCC", "oropharyngeal cancer", "oral cavity cancer",
        "hypopharyngeal cancer", "nasopharyngeal cancer",
        "head neck carcinoma",
    ],
    "buccal mucosa cancer": [
        "buccal carcinoma", "buccal mucosal carcinoma",
        "oral mucosal cancer", "cheek cancer",
        "carcinoma of buccal mucosa", "buccal squamous cell carcinoma",
    ],
    "laryngeal cancer": [
        "larynx cancer", "carcinoma of the larynx",
        "laryngeal carcinoma", "laryngeal squamous cell carcinoma",
        "glottic cancer", "supraglottic cancer",
    ],
    "lymphoma": [
        "lymphatic cancer", "lymphoid malignancy",
        "Hodgkin lymphoma", "non-Hodgkin lymphoma", "NHL", "HL",
        "B-cell lymphoma", "T-cell lymphoma", "diffuse large B-cell lymphoma",
        "DLBCL", "follicular lymphoma",
    ],
    "leukemia": [
        "leukaemia", "blood cancer", "hematologic malignancy",
        "acute myeloid leukemia", "AML", "chronic myeloid leukemia", "CML",
        "acute lymphoblastic leukemia", "ALL", "chronic lymphocytic leukemia",
        "CLL", "bone marrow cancer",
    ],
    "melanoma": [
        "malignant melanoma", "skin melanoma", "cutaneous melanoma",
        "metastatic melanoma", "skin cancer melanoma",
    ],
    "sarcoma": [
        "soft tissue sarcoma", "bone sarcoma", "osteosarcoma",
        "Ewing sarcoma", "rhabdomyosarcoma", "leiomyosarcoma",
        "liposarcoma", "spindle cell sarcoma",
    ],

    # -----------------------------------------------------------------------
    # Treatment — modalities
    # -----------------------------------------------------------------------
    "chemotherapy": [
        "chemo", "cytotoxic therapy", "antineoplastic therapy",
        "cytotoxic chemotherapy", "systemic chemotherapy",
        "cancer drug therapy", "cytotoxic treatment",
    ],
    "radiation therapy": [
        "radiotherapy", "RT", "XRT", "irradiation",
        "radiation treatment", "external beam radiation",
        "EBRT", "radiation oncology", "therapeutic radiation",
        "locoregional radiation",
    ],
    "immunotherapy": [
        "biological therapy", "checkpoint inhibitor therapy",
        "immune checkpoint blockade", "cancer immunotherapy",
        "immuno-oncology", "IO therapy", "biologic therapy",
    ],
    "targeted therapy": [
        "molecularly targeted therapy", "targeted cancer therapy",
        "molecular targeted therapy", "precision oncology",
        "personalized cancer therapy",
    ],
    "palliative care": [
        "supportive care", "comfort care", "palliative treatment",
        "symptom management", "end-of-life care",
        "palliative oncology", "best supportive care", "BSC",
    ],
    "surgery": [
        "surgical resection", "tumor resection", "excision",
        "oncological surgery", "cancer surgery", "curative resection",
        "debulking surgery",
    ],
    "bone marrow transplant": [
        "stem cell transplant", "SCT", "HSCT",
        "hematopoietic stem cell transplantation",
        "allogeneic transplant", "autologous transplant",
    ],
    "hormone therapy": [
        "endocrine therapy", "hormonal therapy",
        "androgen deprivation therapy", "ADT",
        "anti-estrogen therapy", "aromatase inhibitor therapy",
    ],

    # -----------------------------------------------------------------------
    # Drug classes and specific agents
    # -----------------------------------------------------------------------
    "cisplatin": [
        "platinum-based chemotherapy", "CDDP",
        "cis-diamminedichloroplatinum", "cisplatinum",
    ],
    "carboplatin": [
        "platinum compound", "CBDCA", "paraplatin",
    ],
    "paclitaxel": [
        "taxane", "Taxol", "taxane chemotherapy",
        "paclitaxel injection", "PTX",
    ],
    "docetaxel": [
        "taxotere", "taxane", "docetaxel injection",
    ],
    "pembrolizumab": [
        "PD-1 inhibitor", "Keytruda", "anti-PD-1",
        "PD-1 checkpoint inhibitor", "anti-PD-1 therapy",
    ],
    "nivolumab": [
        "Opdivo", "anti-PD-1", "PD-1 inhibitor",
        "nivolumab injection",
    ],
    "bevacizumab": [
        "anti-VEGF", "Avastin", "VEGF inhibitor",
        "anti-angiogenic therapy", "bevacizumab injection",
    ],
    "trastuzumab": [
        "Herceptin", "HER2 inhibitor", "anti-HER2",
        "HER2-targeted therapy",
    ],
    "fluorouracil": [
        "5-FU", "5-fluorouracil", "fluorouracil injection",
    ],
    "doxorubicin": [
        "adriamycin", "anthracycline", "doxorubicin hydrochloride",
    ],
    "cyclophosphamide": [
        "cytoxan", "alkylating agent", "CTX",
    ],
    "methotrexate": [
        "MTX", "antimetabolite", "methotrexate injection",
    ],

    # -----------------------------------------------------------------------
    # Clinical and staging terms
    # -----------------------------------------------------------------------
    "staging": [
        "TNM classification", "cancer staging", "tumor staging",
        "disease staging", "clinical staging", "pathological staging",
        "AJCC staging", "UICC staging",
    ],
    "metastasis": [
        "spread", "secondary cancer", "mets", "distant metastasis",
        "metastatic disease", "metastatic spread", "cancer spread",
        "secondary tumor", "secondary tumour", "dissemination",
    ],
    "remission": [
        "complete response", "CR", "partial response", "PR",
        "disease response", "tumor response", "treatment response",
        "complete remission", "cancer remission",
    ],
    "recurrence": [
        "relapse", "disease recurrence", "cancer relapse",
        "tumor recurrence", "local recurrence", "locoregional recurrence",
        "disease relapse", "refractory disease",
    ],
    "biopsy": [
        "tissue sampling", "histopathology", "tissue biopsy",
        "core needle biopsy", "fine needle aspiration", "FNA",
        "excisional biopsy", "incisional biopsy", "tissue diagnosis",
    ],
    "oncologist": [
        "cancer specialist", "medical oncologist", "oncology physician",
        "radiation oncologist", "surgical oncologist",
        "cancer doctor", "oncology specialist",
    ],
    "prognosis": [
        "disease outcome", "survival outcome", "cancer prognosis",
        "overall survival", "OS", "progression-free survival", "PFS",
        "disease-free survival", "DFS", "5-year survival",
    ],
    "tumor marker": [
        "cancer biomarker", "serum marker", "PSA", "CEA", "CA-125",
        "AFP", "biomarker", "diagnostic marker",
    ],
    "clinical trial": [
        "cancer trial", "oncology trial", "phase I trial", "phase II trial",
        "phase III trial", "randomized controlled trial", "RCT",
        "experimental treatment",
    ],
    "adverse effects": [
        "side effects", "toxicity", "treatment toxicity",
        "chemotherapy side effects", "adverse reactions",
        "drug toxicity", "treatment side effects",
    ],
    "complete response": [
        "CR", "complete remission", "no evidence of disease",
        "NED", "complete regression",
    ],
    "progression": [
        "disease progression", "tumor progression",
        "cancer progression", "PD", "progressive disease",
        "treatment failure", "refractory",
    ],
}

# ---------------------------------------------------------------------------
# Build reverse lookup: any term → canonical term  (auto-generated)
# ---------------------------------------------------------------------------

REVERSE_MAP: dict[str, str] = {}

for _canonical, _synonyms in SYNONYMS.items():
    REVERSE_MAP[_canonical.lower()] = _canonical
    for _syn in _synonyms:
        REVERSE_MAP[_syn.lower()] = _canonical

logger.debug(
    "Medical synonyms loaded: %d canonical terms, %d total mappings",
    len(SYNONYMS),
    len(REVERSE_MAP),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_synonyms(term: str) -> list[str]:
    """
    Return all synonyms for a given term (case-insensitive).
    Also works if term is itself a synonym — looks up the canonical first.

    Returns an empty list if the term is not in the dictionary.
    """
    key = term.strip().lower()
    canonical = REVERSE_MAP.get(key)
    if canonical is None:
        return []
    all_terms = [canonical] + SYNONYMS[canonical]
    # Exclude the original query term from the result
    return [t for t in all_terms if t.lower() != key]


def get_canonical(term: str) -> str | None:
    """
    Return the canonical term for any synonym (case-insensitive).
    Returns None if the term is not recognised.
    """
    return REVERSE_MAP.get(term.strip().lower())


def expand_terms(terms: list[str]) -> list[str]:
    """
    Given a list of terms (e.g. extracted NER entities or query tokens),
    return the original terms plus all their synonyms, deduplicated.
    Preserves original order; synonyms appended after originals.
    """
    seen: set[str] = set()
    result: list[str] = []

    for term in terms:
        lower = term.strip().lower()
        if lower not in seen:
            seen.add(lower)
            result.append(term)

        for syn in get_synonyms(term):
            if syn.lower() not in seen:
                seen.add(syn.lower())
                result.append(syn)

    return result


def list_all_canonical_terms() -> list[str]:
    """Return all canonical terms in alphabetical order."""
    return sorted(SYNONYMS.keys())
