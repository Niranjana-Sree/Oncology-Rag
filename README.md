# MedRAG — Multilingual Medical Question & Answer System

> A production-ready Retrieval-Augmented Generation system for medical Q&A, supporting 6 query types across 5 languages with agentic retrieval, cross-encoder reranking, and continuous RL-based improvement.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Query Types](#query-types)
- [Supported Languages](#supported-languages)
- [Pipeline Deep Dive](#pipeline-deep-dive)
- [Evaluation](#evaluation)
- [Running Evaluation](#running-evaluation)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

**MedRAG** is a research-grade, production-deployable Medical Question & Answer system designed for healthcare professionals, medical students, and clinical decision support applications. It combines state-of-the-art retrieval techniques with a specialized medical language model to deliver accurate, grounded, and verifiable answers from a curated corpus of drug monographs, clinical guidelines, and research papers.

### Why RAG for Medical Q&A?

Large language models hallucinate. In a medical context, hallucination is not a UX problem — it is a patient safety risk. RAG grounds every generated answer in retrieved source documents, enabling:

- **Verifiability**: answers cite specific chunks from trusted documents
- **Updateability**: add new guidelines without retraining the model
- **Auditability**: every answer traces back to a retrievable source
- **Domain accuracy**: the retrieval corpus acts as a hard knowledge boundary

### Who Is This For?

| Audience | Use Case |
|---|---|
| Medical students | Study Q&A, exam preparation (USMLE-style MCQA) |
| Clinicians | Fast drug interaction checks, dosage lookups |
| Researchers | Fact verification against latest guidelines |
| Health-tech teams | Embeddable medical Q&A API |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER QUERY (any language)                      │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     LAQA — Language-Aware Query Analysis                │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │ Lang Detect  │  │ Query Type   │  │ Medical    │  │  Query       │  │
│  │ + Translate  │→ │ Classify     │→ │ NER        │→ │  Expansion   │  │
│  │ (deep-trans) │  │ (BART-MNLI)  │  │ (scispaCy) │  │ (synonyms)   │  │
│  └──────────────┘  └──────────────┘  └────────────┘  └──────────────┘  │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               AGENTIC RAG ORCHESTRATOR                                  │
│                                                                         │
│   ┌──────────────┐    ┌─────────────────┐    ┌────────────────────┐    │
│   │    ROUTER    │    │    ITERATIVE    │    │   DECOMPOSITION    │    │
│   │   Pattern    │    │    Pattern      │    │     Pattern        │    │
│   │ (route query │    │ (retrieve →     │    │ (break complex     │    │
│   │  to corpus   │    │  evaluate →     │    │  query into sub-   │    │
│   │  partition)  │    │  re-retrieve)   │    │  queries, merge)   │    │
│   └──────────────┘    └─────────────────┘    └────────────────────┘    │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    3-STAGE RETRIEVAL PIPELINE                           │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Stage 1 — MRL 128-dim Coarse Scan                                │ │
│  │  ChromaDB (128-dim collection) → top 100 candidates from ~100k    │ │
│  └───────────────────────────────┬────────────────────────────────────┘ │
│                                  │                                      │
│  ┌───────────────────────────────▼────────────────────────────────────┐ │
│  │  Stage 2 — MRL 768-dim Fine Rerank                                │ │
│  │  ChromaDB (768-dim collection) → top 100 → top 20                 │ │
│  └───────────────────────────────┬────────────────────────────────────┘ │
│                                  │                                      │
│  ┌───────────────────────────────▼────────────────────────────────────┐ │
│  │  Stage 3 — Cross-Encoder Rerank                                   │ │
│  │  ms-marco-MiniLM-L-6-v2 → top 20 → top 5 chunks                  │ │
│  └───────────────────────────────┬────────────────────────────────────┘ │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       PROMPT BUILDER                                    │
│   Query-type-aware prompt template + top-5 context chunks injected     │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        GENERATOR                                        │
│   Primary: MedGemma 4B (local, via Ollama/transformers)                │
│   Fallback: Claude API (Anthropic)                                     │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                        ┌───────────┴───────────┐
                        │                       │
                        ▼                       ▼
        ┌───────────────────────┐   ┌───────────────────────┐
        │      RL SCORER        │   │   FASTAPI RESPONSE    │
        │  RLAIF (Claude API)   │   │  answer + sources +   │
        │  GRPO signal for      │   │  query_type + lang +  │
        │  future fine-tuning   │   │  eval_scores          │
        └───────────────────────┘   └───────────────────────┘
```

---

## Features

- **Multilingual input**: accepts queries in English, Hindi, Tamil, Telugu, and Hinglish; auto-detects language and translates before retrieval
- **6 query types**: QA, Multiple-Choice QA, Long-Form QA, Jeopardy-style, Fact Verification, and Fill-in-the-Blank — each with a tailored prompt template and retrieval strategy
- **MRL (Matryoshka Representation Learning) embeddings**: dual-granularity retrieval using 128-dim for coarse speed and 768-dim for semantic precision from a single `nomic-embed-text-v1.5` model
- **Agentic RAG**: three orchestration patterns (Router, Iterative, Decomposition) selected dynamically based on query complexity
- **Cross-encoder reranking**: final scoring with `ms-marco-MiniLM-L-6-v2` for maximum relevance precision
- **Medical NER**: scispaCy `en_core_sci_md` extracts entities (diseases, drugs, procedures) to enrich queries and filter retrieval
- **MedGemma 4B** as primary generator with Claude API as a robust fallback
- **RLAIF + GRPO**: continuous quality improvement using AI feedback scores and group relative policy optimization signals
- **Three-layer evaluation**: lexical (BLEU, ROUGE), semantic (BERTScore, SBERT), and system (RAGAS, SCOPE, LLM-as-Judge)
- **FastAPI REST API**: production-ready endpoints with health checks and metrics
- **ChromaDB persistence**: two vector collections (128-dim, 768-dim) with metadata filtering by document category

---

## Tech Stack

| Component | Tool / Library | Purpose |
|---|---|---|
| Language | Python 3.10+ | Core runtime |
| Embedding model | `nomic-ai/nomic-embed-text-v1.5` | MRL dual-dim embeddings |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Stage 3 cross-encoder scoring |
| Medical NLP | `scispaCy` + `en_core_sci_md` | Medical named entity recognition |
| Vector database | ChromaDB | Persistent vector storage (128-dim + 768-dim) |
| Primary LLM | MedGemma 4B | Medical answer generation (local) |
| Fallback LLM | Claude API (Anthropic) | Answer generation + RLAIF scoring |
| Translation | `deep-translator` + Claude | Query translation; Hinglish via Claude |
| Query classification | `facebook/bart-large-mnli` | Zero-shot query type classification |
| Evaluation | `ragas`, `bert-score`, `rouge-score`, `nltk` | Multi-layer answer evaluation |
| API framework | FastAPI + Uvicorn | REST API serving |
| RL optimization | Anthropic Claude API | RLAIF reward signals for GRPO |
| Document parsing | PyPDF2, python-docx, BeautifulSoup4 | Multi-format document ingestion |
| Configuration | `python-dotenv` | Environment variable management |

---

## Project Structure

```
medical_qa/
├── .env                        # Environment variables (API keys, model paths)
├── requirements.txt            # Python dependencies
├── config.py                   # Centralized configuration (paths, thresholds, model IDs)
├── build_index.py              # Entry point: embed chunks and populate ChromaDB
├── run_chunking.py             # Entry point: load documents and produce chunks/
├── evaluate.py                 # Entry point: run full evaluation suite on results/
├── api.py                      # FastAPI application: routes, middleware, startup
│
├── data/
│   ├── drug_monographs/        # FDA-style drug monograph PDFs and text files
│   ├── clinical_guidelines/    # WHO, NIH, NICE clinical guideline documents
│   ├── research_papers/        # PubMed / arXiv medical research PDFs
│   └── general/                # General medical reference documents
│
├── src/
│   ├── models.py               # Pydantic request/response models and dataclasses
│   ├── document_loader.py      # Multi-format document ingestion (PDF, DOCX, TXT, HTML)
│   ├── chunker.py              # Sentence-aware, overlap-preserving text chunking
│   ├── embedder.py             # MRL embedding logic for 128-dim and 768-dim vectors
│   ├── vectordb.py             # ChromaDB client, collection management, upsert/query
│   ├── medical_synonyms.py     # Medical synonym dictionaries for query expansion
│   ├── laqa.py                 # Language-Aware Query Analysis: detect, translate, classify, NER
│   ├── query_expansion.py      # Synonym expansion + hypothetical document expansion
│   ├── agentic_rag.py          # Agentic orchestrator: Router, Iterative, Decomposition patterns
│   ├── retriever.py            # 3-stage retrieval: Stage 1 (128-dim), Stage 2 (768-dim)
│   ├── reranker.py             # Stage 3 cross-encoder reranking with ms-marco model
│   ├── prompt_builder.py       # Query-type-specific prompt templates and context injection
│   ├── generator.py            # LLM generation: MedGemma primary, Claude fallback
│   ├── rl_scorer.py            # RLAIF scoring via Claude + GRPO signal computation
│   └── pipeline.py             # End-to-end pipeline orchestrator tying all stages together
│
├── chunks/                     # Serialized chunk objects produced by run_chunking.py
├── medical_vectordb/           # ChromaDB persistent storage directory
├── results/                    # Evaluation outputs: JSONL answers, metric CSVs, reports
└── tests/
    ├── test_laqa.py            # Unit tests for language detection, classification, NER
    ├── test_retrieval.py       # Unit tests for 3-stage retrieval and reranker
    └── test_e2e.py             # End-to-end integration tests for the full pipeline
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Git
- 16 GB RAM recommended (MedGemma 4B requires ~8 GB VRAM or CPU offload)
- An Anthropic API key (for Claude fallback and RLAIF scoring)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-org/medrag.git
cd medrag
```

### Step 2 — Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### Step 3 — Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Download the scispaCy Medical NER Model

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz
```

Verify the installation:

```bash
python -c "import spacy; nlp = spacy.load('en_core_sci_md'); print('scispaCy model loaded successfully')"
```

### Step 5 — Set Up the `.env` File

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your values (see [Configuration](#configuration) for all variables).

### Step 6 — Verify the Installation

```bash
python -c "
from src.embedder import Embedder
from src.laqa import LAQA
e = Embedder()
print('Embedder ready — dim 128:', e.embed(['test'], dim=128).shape)
print('Embedder ready — dim 768:', e.embed(['test'], dim=768).shape)
l = LAQA()
result = l.analyze('What is the dosage of Metformin?')
print('LAQA result:', result)
"
```

---

## Usage

### Step 1 — Prepare Your Documents

Place your source documents into the appropriate subdirectory under `data/`:

```
data/drug_monographs/   → drug PDFs, monograph text files
data/clinical_guidelines/ → WHO/NIH/NICE PDFs
data/research_papers/   → PubMed articles, arXiv PDFs
data/general/           → other medical reference material
```

Supported formats: `.pdf`, `.docx`, `.txt`, `.html`

### Step 2 — Run Document Chunking

```bash
python run_chunking.py --data-dir data/ --output-dir chunks/ --chunk-size 512 --overlap 64
```

This produces serialized chunk files in `chunks/` with metadata (source, category, page number).

### Step 3 — Build the Vector Index

```bash
python build_index.py --chunks-dir chunks/ --vectordb-dir medical_vectordb/
```

This embeds all chunks at both 128-dim and 768-dim and populates the two ChromaDB collections. Progress is logged to stdout.

### Step 4 — Start the API Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API is now available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Step 5 — Send a Query

**Using curl:**

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the contraindications of Metformin in diabetic patients?",
    "language": "auto",
    "query_type": "auto",
    "top_k": 5
  }'
```

**Using Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/query",
    json={
        "query": "मेटफॉर्मिन की खुराक क्या है?",  # Hindi query
        "language": "auto",
        "query_type": "auto",
        "top_k": 5,
    },
)

data = response.json()
print("Answer:", data["answer"])
print("Query type detected:", data["query_type"])
print("Language detected:", data["language"])
print("Sources:")
for src in data["sources"]:
    print(f"  - {src['document']} (score: {src['score']:.3f})")
```

---

## Query Types

| Query Type | Description | Example | Retrieval Strategy |
|---|---|---|---|
| **QA** | Direct factual question | "What is the half-life of Aspirin?" | Standard 3-stage retrieval, top 5 |
| **MCQA** | Multiple-choice question | "Which drug is first-line for Type 2 diabetes? A) Insulin B) Metformin C) Glipizide D) Sitagliptin" | Retrieval + option-aware reranking |
| **LFQA** | Long-form explanatory answer | "Explain the mechanism of action of ACE inhibitors in heart failure." | Iterative agentic pattern, top 8 |
| **Jeopardy** | Answer-to-question format | "This SSRI is most commonly prescribed for OCD in children." | Decomposition pattern |
| **Fact Verification** | True/False/Uncertain claim check | "Claim: Penicillin is effective against MRSA." | Router pattern → guideline corpus |
| **Fill-in-Blank** | Cloze-style completion | "The antidote for Heparin overdose is ___." | Standard retrieval, top 3 |

---

## Supported Languages

| Language | Script | Detection Method | Translation Method |
|---|---|---|---|
| English | Latin | `langdetect` primary signal | No translation needed |
| Hindi | Devanagari | Unicode block detection + `langdetect` | `deep-translator` (GoogleTranslator) |
| Tamil | Tamil script | Unicode block detection | `deep-translator` (GoogleTranslator) |
| Telugu | Telugu script | Unicode block detection | `deep-translator` (GoogleTranslator) |
| Hinglish | Latin (mixed) | Keyword heuristics + Claude classifier | Claude API (context-aware transliteration) |

All queries are normalized to English before retrieval. The original query language is stored in the response for UI rendering purposes.

---

## Pipeline Deep Dive

### LAQA — Language-Aware Query Analysis

LAQA is the entry gate for every query. It performs four sequential operations. First, language detection combines Unicode script analysis (for Devanagari, Tamil, and Telugu) with `langdetect` statistical scoring to identify the source language; Hinglish (romanized Hindi mixed with English) is detected via a keyword heuristic and confirmed by Claude. Second, translation converts non-English queries to English using `deep-translator`'s GoogleTranslator for structured languages and Claude for Hinglish, which requires semantic context to translate correctly. Third, query type classification uses `facebook/bart-large-mnli` in zero-shot mode with six candidate labels (QA, MCQA, LFQA, Jeopardy, FactVerification, FillInBlank); the top scoring label drives downstream prompt selection and agentic pattern choice. Fourth, medical NER with scispaCy `en_core_sci_md` extracts disease names, drug names, dosages, and clinical procedures, which are added as metadata filters for ChromaDB retrieval and seed synonym expansion.

### Chunking Strategy

Documents are loaded from `data/` by `document_loader.py` using format-specific parsers (PyPDF2 for PDFs, python-docx for Word files, BeautifulSoup4 for HTML). The `chunker.py` module applies sentence-boundary-aware chunking with a configurable window size (default 512 tokens) and overlap (default 64 tokens). Each chunk is stored with rich metadata: source document path, document category (drug_monograph, clinical_guideline, research_paper, general), page number, section heading, and a unique chunk ID. This metadata enables ChromaDB `where` filters that restrict retrieval to the most relevant corpus partition.

### MRL Embeddings — Dual-Granularity Retrieval

The `nomic-ai/nomic-embed-text-v1.5` model implements Matryoshka Representation Learning, which trains a single embedding model to produce coherent representations at multiple dimensionalities. MedRAG exploits this with two ChromaDB collections: a 128-dimensional collection for Stage 1 coarse scanning (fast cosine search over ~100k chunks, returning top 100 candidates) and a 768-dimensional collection for Stage 2 fine reranking (re-scoring the top 100 from Stage 1 to produce top 20). Using 128-dim for the initial pass reduces compute by 6× compared to a single 768-dim scan, without meaningful recall loss.

### Agentic RAG — Three Orchestration Patterns

The `agentic_rag.py` orchestrator selects among three retrieval patterns based on query type and complexity score. The **Router pattern** is triggered for Fact Verification queries: it routes the query exclusively to the clinical_guidelines ChromaDB partition, which has higher authority for clinical claims. The **Iterative pattern** is used for LFQA queries: it retrieves an initial set, scores answer completeness using Claude, and issues follow-up retrievals if coverage gaps are detected — typically running 2–3 rounds. The **Decomposition pattern** handles complex multi-part queries: Claude decomposes the query into atomic sub-questions, each sub-question is retrieved independently, and answers are merged into a coherent long-form response. All other query types default to the standard 3-stage pipeline.

### Cross-Encoder Reranking

After Stage 2 produces 20 candidate chunks, `reranker.py` passes each (query, chunk) pair through `cross-encoder/ms-marco-MiniLM-L-6-v2`. Unlike bi-encoder retrieval (where query and document are embedded independently), a cross-encoder jointly encodes the pair, producing a single relevance score that captures fine-grained semantic interaction. The top 5 chunks by cross-encoder score are forwarded to the prompt builder. This stage is the primary quality gate — Stage 1 and 2 optimize for recall, Stage 3 optimizes for precision.

### MedGemma Generation

`generator.py` formats the top-5 chunks and query into a structured prompt via `prompt_builder.py`, then passes it to MedGemma 4B loaded locally. MedGemma is a medical-domain fine-tune of Google's Gemma 2B/4B and has stronger calibration on clinical terminology than general-purpose models. If the local MedGemma instance is unavailable (OOM, model load failure, timeout), the generator transparently falls back to the Claude API with the same prompt, ensuring zero downtime for the API. The response includes the raw generated text, source chunk IDs, and a confidence signal from the generator.

### RL Optimization — RLAIF + GRPO

Every generated answer is scored by `rl_scorer.py` using RLAIF (Reinforcement Learning from AI Feedback): Claude evaluates the answer on four axes — medical accuracy, faithfulness to retrieved context, completeness, and safety — returning a scalar reward in [0, 1]. These scores are logged to `results/rl_scores.jsonl` and used to compute GRPO (Group Relative Policy Optimization) training signals for periodic fine-tuning of MedGemma. GRPO computes the relative advantage of each answer within a group of candidates generated for the same query, providing a stable gradient signal without a separate value model.

---

## Evaluation

MedRAG uses a three-layer evaluation framework covering lexical overlap, semantic similarity, and system-level quality.

### Layer 1 — Lexical Metrics

| Metric | Target Score | Description |
|---|---|---|
| BLEU-1 | ≥ 0.45 | Unigram precision of generated vs. reference answer |
| BLEU-4 | ≥ 0.20 | 4-gram precision (stricter fluency measure) |
| ROUGE-1 | ≥ 0.50 | Unigram recall of reference tokens in generated answer |
| ROUGE-L | ≥ 0.40 | Longest common subsequence F1 |
| Token F1 | ≥ 0.55 | Exact token overlap F1 (common in QA benchmarks) |
| MCQA Accuracy | ≥ 0.75 | Exact match on answer option (A/B/C/D) |
| DISTINCT-1 | ≥ 0.60 | Lexical diversity of generated answers |

### Layer 2 — Semantic Metrics

| Metric | Target Score | Description |
|---|---|---|
| BERTScore F1 | ≥ 0.85 | Token-level contextual similarity via BERT |
| SBERT Cosine | ≥ 0.80 | Sentence-level semantic similarity via SBERT |

### Layer 3 — System Metrics

| Metric | Target Score | Description |
|---|---|---|
| RAGAS Faithfulness | ≥ 0.80 | Fraction of answer claims entailed by retrieved context |
| RAGAS Answer Relevance | ≥ 0.75 | Relevance of answer to the input question |
| RAGAS Context Recall | ≥ 0.70 | Coverage of ground-truth by retrieved context |
| RAGAS Context Precision | ≥ 0.72 | Fraction of retrieved context that is relevant |
| SCOPE | ≥ 0.75 | Scientific claim overlap and precision for medical text |
| LLM-as-Judge (Claude) | ≥ 4.0 / 5.0 | Holistic quality rating by Claude on accuracy + safety |

---

## Running Evaluation

Ensure you have a `results/answers.jsonl` file from running queries through the pipeline, then:

```bash
python evaluate.py \
  --answers results/answers.jsonl \
  --references data/references.jsonl \
  --output-dir results/ \
  --layers 1 2 3
```

Sample output:

```
============================================================
  MedRAG Evaluation Report — 2024-01-15
============================================================

Layer 1 — Lexical
  BLEU-1        :  0.472
  BLEU-4        :  0.213
  ROUGE-1       :  0.531
  ROUGE-L       :  0.418
  Token F1      :  0.574
  MCQA Accuracy :  0.783
  DISTINCT-1    :  0.634

Layer 2 — Semantic
  BERTScore F1  :  0.867
  SBERT Cosine  :  0.824

Layer 3 — System
  RAGAS Faithfulness     :  0.831
  RAGAS Answer Relevance :  0.764
  RAGAS Context Recall   :  0.712
  RAGAS Context Precision:  0.738
  SCOPE                  :  0.776
  LLM-as-Judge           :  4.1 / 5.0

Results saved to: results/eval_report_20240115.json
============================================================
```

---

## API Reference

### POST `/api/query`

Submit a medical question and receive a grounded answer.

**Request body:**

```json
{
  "query": "string (required) — the medical question in any supported language",
  "language": "string (optional, default: 'auto') — 'en', 'hi', 'ta', 'te', 'hinglish', or 'auto'",
  "query_type": "string (optional, default: 'auto') — 'qa', 'mcqa', 'lfqa', 'jeopardy', 'fact_verification', 'fill_blank', or 'auto'",
  "top_k": "integer (optional, default: 5, max: 10) — number of context chunks to retrieve",
  "agentic_pattern": "string (optional, default: 'auto') — 'router', 'iterative', 'decomposition', or 'auto'"
}
```

**Response body:**

```json
{
  "answer": "string — the generated medical answer",
  "query_type": "string — detected or specified query type",
  "language": "string — detected source language code",
  "sources": [
    {
      "chunk_id": "string",
      "document": "string — source document name",
      "category": "string — drug_monograph | clinical_guideline | research_paper | general",
      "text": "string — the retrieved chunk text",
      "score": "float — cross-encoder relevance score"
    }
  ],
  "agentic_pattern_used": "string — which pattern was applied",
  "rl_score": "float — RLAIF quality score in [0, 1]",
  "latency_ms": "integer — end-to-end latency in milliseconds",
  "model_used": "string — 'medgemma-4b' or 'claude-fallback'"
}
```

**curl example:**

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the mechanism of action of Warfarin?",
    "language": "auto",
    "query_type": "auto",
    "top_k": 5
  }'
```

**Python example:**

```python
import requests

resp = requests.post(
    "http://localhost:8000/api/query",
    json={
        "query": "What is the mechanism of action of Warfarin?",
        "language": "auto",
        "query_type": "auto",
        "top_k": 5,
    },
    timeout=60,
)
resp.raise_for_status()
result = resp.json()
print(result["answer"])
```

---

### GET `/api/health`

Check API server and model status.

**Response:**

```json
{
  "status": "ok",
  "medgemma_loaded": true,
  "chromadb_connected": true,
  "chunks_128_count": 98432,
  "chunks_768_count": 98432,
  "uptime_seconds": 3672
}
```

---

### GET `/api/metrics`

Retrieve aggregated runtime metrics.

**Response:**

```json
{
  "total_queries": 1247,
  "avg_latency_ms": 1843,
  "p95_latency_ms": 4120,
  "avg_rl_score": 0.814,
  "fallback_rate": 0.032,
  "query_type_distribution": {
    "qa": 0.41,
    "mcqa": 0.28,
    "lfqa": 0.15,
    "jeopardy": 0.06,
    "fact_verification": 0.07,
    "fill_blank": 0.03
  },
  "language_distribution": {
    "en": 0.62,
    "hi": 0.18,
    "ta": 0.08,
    "te": 0.07,
    "hinglish": 0.05
  }
}
```

---

## Configuration

All configuration is managed via the `.env` file in the project root.

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key for Claude fallback and RLAIF scoring |
| `MEDGEMMA_MODEL_PATH` | Yes | — | Local path to MedGemma 4B weights (HuggingFace cache or custom path) |
| `MEDGEMMA_DEVICE` | No | `cuda` | Device for MedGemma: `cuda`, `cpu`, or `mps` (Apple Silicon) |
| `CHROMADB_PATH` | No | `./medical_vectordb` | Directory for ChromaDB persistent storage |
| `CHUNKS_DIR` | No | `./chunks` | Directory where serialized chunks are stored |
| `EMBEDDING_MODEL` | No | `nomic-ai/nomic-embed-text-v1.5` | HuggingFace embedding model ID |
| `RERANKER_MODEL` | No | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace cross-encoder model ID |
| `STAGE1_TOP_K` | No | `100` | Number of candidates retrieved in Stage 1 (128-dim) |
| `STAGE2_TOP_K` | No | `20` | Number of candidates after Stage 2 reranking (768-dim) |
| `STAGE3_TOP_K` | No | `5` | Final candidates after cross-encoder reranking |
| `CLAUDE_MODEL` | No | `claude-sonnet-4-6` | Claude model ID for fallback generation and RLAIF |
| `RLAIF_ENABLED` | No | `true` | Whether to score every response with RLAIF |
| `API_HOST` | No | `0.0.0.0` | FastAPI host address |
| `API_PORT` | No | `8000` | FastAPI port |
| `LOG_LEVEL` | No | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `MAX_CHUNK_SIZE` | No | `512` | Token limit per chunk during ingestion |
| `CHUNK_OVERLAP` | No | `64` | Token overlap between adjacent chunks |

**.env example:**

```bash
ANTHROPIC_API_KEY=sk-ant-...
MEDGEMMA_MODEL_PATH=/models/medgemma-4b
MEDGEMMA_DEVICE=cuda
CHROMADB_PATH=./medical_vectordb
STAGE1_TOP_K=100
STAGE2_TOP_K=20
STAGE3_TOP_K=5
CLAUDE_MODEL=claude-sonnet-4-6
RLAIF_ENABLED=true
LOG_LEVEL=INFO
```

---

## Troubleshooting

### 1. `OSError: [E050] Can't find model 'en_core_sci_md'`

The scispaCy medical model is not installed from the standard pip registry. Install it directly:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz
```

If your scispaCy version is different from 0.5.3, check the [scispaCy releases page](https://allenai.github.io/scispacy/) for the matching model URL.

---

### 2. `torch.cuda.OutOfMemoryError` when loading MedGemma

MedGemma 4B requires approximately 8 GB of VRAM in float16. Solutions:

```bash
# Option A: use CPU (slower but no VRAM requirement)
MEDGEMMA_DEVICE=cpu

# Option B: use 4-bit quantization (add bitsandbytes to requirements)
# Set in config.py: MEDGEMMA_LOAD_IN_4BIT=True

# Option C: use Apple Silicon MPS (Mac M1/M2/M3)
MEDGEMMA_DEVICE=mps
```

---

### 3. ChromaDB returns zero results after `build_index.py`

This usually means the embedding or upsert step failed silently. Check:

```bash
python -c "
import chromadb
client = chromadb.PersistentClient(path='./medical_vectordb')
for col in client.list_collections():
    print(col.name, col.count())
"
```

If counts are 0, re-run `build_index.py` with `--log-level DEBUG` and check for embedding errors. Also verify that `chunks/` is non-empty after running `run_chunking.py`.

---

### 4. `anthropic.AuthenticationError`: invalid API key

Ensure your `.env` file is in the project root and `python-dotenv` is loading it:

```python
from dotenv import load_dotenv
load_dotenv()
import os
print(os.getenv("ANTHROPIC_API_KEY")[:10])  # should print sk-ant-api0
```

If the key is correct but the error persists, check that the key has not been rotated in the Anthropic console.

---

### 5. Translation returns the original query unchanged for Hindi input

`deep-translator` requires an internet connection. If the translation silently fails:

```bash
python -c "from deep_translator import GoogleTranslator; print(GoogleTranslator(source='hi', target='en').translate('नमस्ते'))"
```

If this fails, check your network proxy settings or switch to the Claude translation path by setting `USE_CLAUDE_TRANSLATION=true` in `.env`.

---

## Contributing

### Adding New Documents

1. Place documents in the appropriate `data/` subdirectory.
2. Re-run chunking: `python run_chunking.py --data-dir data/ --output-dir chunks/`
3. Re-run indexing: `python build_index.py --chunks-dir chunks/ --vectordb-dir medical_vectordb/`

Incremental indexing (only new documents) is supported with the `--incremental` flag on `build_index.py`.

### Adding a New Language

1. Add a detection rule in `src/laqa.py` in the `detect_language()` method.
2. Add a translation handler in the `translate_to_english()` method.
3. Add the new language code to the `SUPPORTED_LANGUAGES` list in `config.py`.
4. Add test cases to `tests/test_laqa.py`.
5. Update the [Supported Languages](#supported-languages) table in this README.

### Adding a New Query Type

1. Add the new type label to `QUERY_TYPES` in `config.py`.
2. Add a prompt template in `src/prompt_builder.py` in the `build_prompt()` method.
3. Define the retrieval strategy override (if any) in `src/agentic_rag.py`.
4. Add BART-MNLI classification label in `src/laqa.py`.
5. Add end-to-end test cases in `tests/test_e2e.py`.
6. Update the [Query Types](#query-types) table in this README.

### Code Style

- Follow PEP 8; use `black` for formatting (`black src/ tests/`)
- Type-annotate all public functions
- Run tests before submitting a PR: `pytest tests/ -v`

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 MedRAG Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

> **Medical Disclaimer**: This system is intended for informational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.

---

## Acknowledgements

### Datasets

| Dataset | Source | Use |
|---|---|---|
| MedQA (USMLE) | Jin et al., 2021 | MCQA benchmark and fine-tuning |
| MedMCQA | Paillard et al., 2022 | Multi-subject MCQA evaluation |
| PubMedQA | Jin et al., 2019 | Biomedical research Q&A |
| BioASQ | Tsatsaronis et al., 2015 | Biomedical semantic indexing and QA |
| MIMIC-III (de-identified) | Johnson et al., 2016 | Clinical note vocabulary and NER |

### Models and Libraries

| Resource | Authors / Organization | License |
|---|---|---|
| `nomic-embed-text-v1.5` | Nomic AI | Apache 2.0 |
| `ms-marco-MiniLM-L-6-v2` | Microsoft / Hugging Face | Apache 2.0 |
| `en_core_sci_md` (scispaCy) | Allen Institute for AI | MIT |
| `facebook/bart-large-mnli` | Meta AI | MIT |
| MedGemma 4B | Google DeepMind | Gemma Terms of Use |
| Claude API | Anthropic | Commercial API Terms |
| ChromaDB | Chroma | Apache 2.0 |
| RAGAS | Explodinggradients | Apache 2.0 |
| FastAPI | Sebastián Ramírez | MIT |

### Inspiration and Prior Work

- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS 2020)
- Kusupati et al., "Matryoshka Representation Learning" (NeurIPS 2022)
- Ouyang et al., "Training language models to follow instructions with human feedback" (NeurIPS 2022)
- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (NeurIPS 2023)
