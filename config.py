"""
Central configuration for MedRAG.
All modules import from here — never read os.environ directly elsewhere.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root (the directory containing this file)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Apple Silicon MPS — disable memory upper limit to prevent OOM on large batches
# Must be set BEFORE torch is imported, so we set it here at config load time.
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("medrag.config")

# ---------------------------------------------------------------------------
# Anthropic / Claude
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

if not ANTHROPIC_API_KEY:
    logger.warning(
        "ANTHROPIC_API_KEY is not set. Claude fallback and RLAIF scoring will fail."
    )

# ---------------------------------------------------------------------------
# MedGemma
# ---------------------------------------------------------------------------
MEDGEMMA_MODEL_PATH: str = os.getenv("MEDGEMMA_MODEL_PATH", "google/medgemma-4b-it")
MEDGEMMA_DEVICE: str = os.getenv("MEDGEMMA_DEVICE", "cuda")

# ---------------------------------------------------------------------------
# Embedding and reranker models
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
# NOTE: Switch to "nomic-ai/nomic-embed-text-v1.5" when a CUDA GPU is available.
# nomic produces 768-dim MRL vectors; MiniLM produces 384-dim.
# The MRL truncation logic in embedder.py handles both sizes automatically.
RERANKER_MODEL: str = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ---------------------------------------------------------------------------
# Paths — resolved to absolute so all modules work regardless of cwd
# ---------------------------------------------------------------------------
CHROMADB_PATH: Path = Path(
    os.getenv("CHROMADB_PATH", str(_PROJECT_ROOT / "medical_vectordb"))
).resolve()

CHUNKS_DIR: Path = Path(
    os.getenv("CHUNKS_DIR", str(_PROJECT_ROOT / "chunks"))
).resolve()

DATA_DIR: Path = Path(
    os.getenv("DATA_DIR", str(_PROJECT_ROOT / "data"))
).resolve()

RESULTS_DIR: Path = Path(
    os.getenv("RESULTS_DIR", str(_PROJECT_ROOT / "results"))
).resolve()

REWARDS_DB_PATH: Path = Path(
    os.getenv("REWARDS_DB_PATH", str(RESULTS_DIR / "rewards.db"))
).resolve()

# Ensure output directories exist at import time
for _dir in (CHUNKS_DIR, RESULTS_DIR, CHROMADB_PATH):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data subdirectories — oncology corpus only (no drug monographs)
# ---------------------------------------------------------------------------
DATA_CLINICAL_GUIDELINES: Path = DATA_DIR / "clinical_guidelines"
DATA_RESEARCH_PAPERS: Path = DATA_DIR / "research_papers"
DATA_GENERAL: Path = DATA_DIR / "general"

# ---------------------------------------------------------------------------
# Retrieval hyper-parameters
# ---------------------------------------------------------------------------
STAGE1_TOP_K: int = int(os.getenv("STAGE1_TOP_K", "100"))
STAGE2_TOP_K: int = int(os.getenv("STAGE2_TOP_K", "20"))
STAGE3_TOP_K: int = int(os.getenv("STAGE3_TOP_K", "5"))

# MRL embedding dimensions
# MiniLM default: DIM_COARSE=64, DIM_FINE=384
# nomic (GPU):    DIM_COARSE=128, DIM_FINE=768
DIM_COARSE: int = int(os.getenv("DIM_COARSE", "64"))
DIM_FINE: int   = int(os.getenv("DIM_FINE",   "384"))

# ChromaDB collection names — named after actual dims in use
COLLECTION_COARSE: str = f"medrag_{DIM_COARSE}"
COLLECTION_FINE: str   = f"medrag_{DIM_FINE}"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# ---------------------------------------------------------------------------
# Query types
# ---------------------------------------------------------------------------
QUERY_TYPES: list[str] = [
    "qa",
    "mcqa",
    "lfqa",
    "jeopardy",
    "fact_verification",
    "fill_blank",
]

# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES: list[str] = ["en", "hi", "ta", "te", "hinglish"]

# ---------------------------------------------------------------------------
# Document categories (used as ChromaDB metadata filter values)
# Oncology-only corpus — drug_monograph removed
# ---------------------------------------------------------------------------
DOC_CATEGORIES: list[str] = [
    "clinical_guideline",
    "research_paper",
    "general",
]

# Category → data directory mapping
CATEGORY_TO_DIR: dict[str, Path] = {
    "clinical_guideline": DATA_CLINICAL_GUIDELINES,
    "research_paper":     DATA_RESEARCH_PAPERS,
    "general":            DATA_GENERAL,
}

# ---------------------------------------------------------------------------
# RL scoring
# ---------------------------------------------------------------------------
RLAIF_ENABLED: bool = os.getenv("RLAIF_ENABLED", "true").lower() == "true"

# RLAIF dimension weights (must sum to 1.0)
RLAIF_WEIGHTS: dict[str, float] = {
    "medical_accuracy": 0.35,
    "faithfulness": 0.25,
    "completeness": 0.20,
    "safety": 0.15,
    "clarity": 0.05,
}

# ---------------------------------------------------------------------------
# API server
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ---------------------------------------------------------------------------
# Sanity-check print (DEBUG only)
# ---------------------------------------------------------------------------
logger.debug("Config loaded — project root: %s", _PROJECT_ROOT)
logger.debug("CHROMADB_PATH : %s", CHROMADB_PATH)
logger.debug("CHUNKS_DIR    : %s", CHUNKS_DIR)
logger.debug("DATA_DIR      : %s", DATA_DIR)
logger.debug("RESULTS_DIR   : %s", RESULTS_DIR)
logger.debug("CLAUDE_MODEL  : %s", CLAUDE_MODEL)
logger.debug("EMBEDDING_MODEL: %s", EMBEDDING_MODEL)
