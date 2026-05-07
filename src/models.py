"""
Shared dataclasses, enums, and Pydantic models for MedRAG.
All other modules import types from here — never define types inline elsewhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Language(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"
    TAMIL = "ta"
    TELUGU = "te"
    HINGLISH = "hinglish"
    UNKNOWN = "unknown"


class QueryType(str, Enum):
    QA = "qa"
    MCQA = "mcqa"
    LFQA = "lfqa"
    JEOPARDY = "jeopardy"
    FACT_VERIFICATION = "fact_verification"
    FILL_BLANK = "fill_blank"
    UNKNOWN = "unknown"


class DocCategory(str, Enum):
    DRUG_MONOGRAPH = "drug_monograph"
    CLINICAL_GUIDELINE = "clinical_guideline"
    RESEARCH_PAPER = "research_paper"
    GENERAL = "general"


class AgenticPattern(str, Enum):
    ROUTER = "router"
    ITERATIVE = "iterative"
    DECOMPOSITION = "decomposition"
    STANDARD = "standard"


class ChunkStrategy(str, Enum):
    SENTENCE = "sentence"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    AGENTIC = "agentic"


# ---------------------------------------------------------------------------
# Core document and chunk dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RawDocument:
    """A document as loaded from disk, before chunking."""
    doc_id: str
    file_path: str
    category: str                    # one of DocCategory values
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # populated by document_loader
    page_count: int = 0
    word_count: int = 0


@dataclass
class Chunk:
    """A single text chunk ready for embedding and storage."""
    chunk_id: str                    # e.g. "doc_id__0042"
    doc_id: str
    text: str
    category: str                    # DocCategory value
    source_file: str
    chunk_index: int                 # position within parent document
    strategy: str                    # ChunkStrategy value used to produce this chunk
    metadata: dict[str, Any] = field(default_factory=dict)
    # optional fields populated after embedding
    embedding_128: list[float] = field(default_factory=list)
    embedding_768: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LAQA output
# ---------------------------------------------------------------------------

@dataclass
class MedicalEntity:
    """A single entity extracted by scispaCy NER."""
    text: str
    label: str           # e.g. "DISEASE", "CHEMICAL", "PROCEDURE"
    start: int           # char offset in (translated) query
    end: int


@dataclass
class AnalysedQuery:
    """Full output of the LAQA pipeline for one user query."""
    original_query: str
    translated_query: str            # English version (same as original if already English)
    detected_language: str           # Language enum value
    query_type: str                  # QueryType enum value
    query_type_confidence: float     # 0.0 – 1.0
    medical_entities: list[MedicalEntity] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)   # filled by query_expansion
    entity_texts: list[str] = field(default_factory=list)       # convenience: just the strings


# ---------------------------------------------------------------------------
# Retrieval results
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A chunk returned by any retrieval stage, with its relevance score."""
    chunk_id: str
    doc_id: str
    text: str
    category: str
    source_file: str
    score: float                     # higher = more relevant
    stage: str                       # "stage1" | "stage2" | "stage3"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Container for the output of the full 3-stage retrieval pipeline."""
    analysed_query: AnalysedQuery
    stage1_chunks: list[RetrievedChunk] = field(default_factory=list)
    stage2_chunks: list[RetrievedChunk] = field(default_factory=list)
    stage3_chunks: list[RetrievedChunk] = field(default_factory=list)
    agentic_pattern_used: str = AgenticPattern.STANDARD.value
    retrieval_latency_ms: float = 0.0

    @property
    def final_chunks(self) -> list[RetrievedChunk]:
        """The top chunks after all reranking stages."""
        return self.stage3_chunks if self.stage3_chunks else self.stage2_chunks


# ---------------------------------------------------------------------------
# Generation output
# ---------------------------------------------------------------------------

@dataclass
class GeneratedAnswer:
    """Raw output from the generator before RL scoring."""
    answer_text: str
    model_used: str                  # "medgemma-4b" | "claude-fallback"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_latency_ms: float = 0.0
    passed_safety_check: bool = True
    safety_flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RL / scoring
# ---------------------------------------------------------------------------

@dataclass
class RLScore:
    """RLAIF scores for a single answer on five dimensions."""
    medical_accuracy: float   # 0.0 – 1.0
    faithfulness: float
    completeness: float
    safety: float
    clarity: float
    composite: float          # weighted sum using RLAIF_WEIGHTS from config
    raw_feedback: str = ""    # Claude's textual justification


# ---------------------------------------------------------------------------
# Full pipeline response (internal)
# ---------------------------------------------------------------------------

@dataclass
class PipelineResponse:
    """Complete response object produced by pipeline.answer_query()."""
    query: str
    answer: str
    query_type: str
    detected_language: str
    retrieved_chunks: list[RetrievedChunk]
    agentic_pattern_used: str
    model_used: str
    rl_score: Optional[RLScore]
    total_latency_ms: float
    agent_iterations: int = 0           # number of retrieval iterations (iterative agent)
    query_type_confidence: float = 0.0  # LAQA confidence score
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# FastAPI request / response schemas (Pydantic)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request body for POST /api/query."""
    query: str = Field(..., min_length=3, max_length=2000, description="Medical question in any supported language")
    language: str = Field("auto", description="'auto' or a Language enum value")
    query_type: str = Field("auto", description="'auto' or a QueryType enum value")
    top_k: int = Field(5, ge=1, le=10, description="Number of context chunks to return")
    agentic_pattern: str = Field("auto", description="'auto' or an AgenticPattern enum value")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        valid = {"auto"} | {lang.value for lang in Language}
        if v not in valid:
            raise ValueError(f"language must be one of {sorted(valid)}")
        return v

    @field_validator("query_type")
    @classmethod
    def validate_query_type(cls, v: str) -> str:
        valid = {"auto"} | {qt.value for qt in QueryType}
        if v not in valid:
            raise ValueError(f"query_type must be one of {sorted(valid)}")
        return v

    @field_validator("agentic_pattern")
    @classmethod
    def validate_agentic_pattern(cls, v: str) -> str:
        valid = {"auto"} | {ap.value for ap in AgenticPattern}
        if v not in valid:
            raise ValueError(f"agentic_pattern must be one of {sorted(valid)}")
        return v


class SourceChunk(BaseModel):
    """A single source chunk returned in the API response."""
    chunk_id: str
    document: str
    category: str
    text: str
    score: float


class QueryResponse(BaseModel):
    """Response body for POST /api/query."""
    answer: str
    query_type: str
    language: str
    sources: list[SourceChunk]
    agentic_pattern_used: str
    rl_score: Optional[float]
    latency_ms: float
    model_used: str


class DualQueryResponse(BaseModel):
    """Response body for POST /api/query/dual."""
    patient_answer: str
    doctor_answer: str
    query_type: str
    language: str
    sources: list[SourceChunk]
    agentic_pattern_used: str
    agent_iterations: int
    latency_ms: float
    model_used: str


class HealthResponse(BaseModel):
    """Response body for GET /api/health."""
    status: str
    medgemma_loaded: bool
    chromadb_connected: bool
    chunks_128_count: int
    chunks_768_count: int
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Response body for GET /api/metrics."""
    total_queries: int
    avg_latency_ms: float
    p95_latency_ms: float
    avg_rl_score: float
    fallback_rate: float
    query_type_distribution: dict[str, float]
    language_distribution: dict[str, float]
