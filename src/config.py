# Central configuration: paths, model names, hyperparameters, roles, GPU device, and logger.

import os
import logging
from pathlib import Path

# Directory layout (paths resolve relative to the project, wherever it lives).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(PROJECT_ROOT / "data" / "raw_docs")
CORPUS_DIR = str(PROJECT_ROOT / "data" / "corpus")
ARTIFACTS_DIR = str(PROJECT_ROOT / "artifacts")
VECTOR_DB_DIR = str(PROJECT_ROOT / "artifacts" / "vector_db")
EVAL_REPORTS_DIR = str(PROJECT_ROOT / "artifacts" / "eval_reports")
AUDIT_DIR = str(PROJECT_ROOT / "artifacts" / "audit")
LOG_FILE = str(PROJECT_ROOT / "artifacts" / "pipeline_run.log")

_ALL_DIRS = [DATA_DIR, CORPUS_DIR, ARTIFACTS_DIR, VECTOR_DB_DIR, EVAL_REPORTS_DIR, AUDIT_DIR]

# RAG hyperparameters
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
TOP_K_VECTORS = 7
RRF_K = 60
MAX_CHUNKS_PER_CASE = 3          # balance cap so one long chargesheet can't dominate

# Models (current on Groq, Sept 2026; old llama-3.3-70b and qwen3-32b were deprecated)
GENERATOR_MODEL = "openai/gpt-oss-120b"
EVALUATOR_MODEL = "qwen/qwen3.6-27b"     # different family from generator (cross-family judge)
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"     # multilingual: English + Hindi + code-mixed

EMBEDDING_DEVICE = "cpu"

# API reliability
MAX_API_RETRIES = 3
API_RETRY_MIN_WAIT = 2
API_RETRY_MAX_WAIT = 10

# Access-control roles (RBAC)
ROLES = ["IO", "Prosecutor", "Court", "RTI_Public"]

def setup_environment():
    # Create every project directory if missing; safe on Google Drive where exist_ok can misfire.
    for directory in _ALL_DIRS:
        if not os.path.isdir(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except FileExistsError:
                pass

def get_logger(name="anukram"):
    # Return a configured, idempotent logger writing to console and file.
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    try:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Could not create log file at %s. Console logging only.", LOG_FILE)
    return logger

logger = get_logger("anukram")