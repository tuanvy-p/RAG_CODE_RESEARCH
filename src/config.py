import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables (.env with fallback to .env.example)
env_file = BASE_DIR / ".env"
if env_file.is_file():
    load_dotenv(env_file, override=True)
else:
    load_dotenv(BASE_DIR / ".env.example", override=True)

# Clean and synchronize API keys in environment variables
_raw_groq_key = (os.getenv("GROQ_API_KEY") or "").strip('"\' \n\t')
if _raw_groq_key:
    os.environ["GROQ_API_KEY"] = _raw_groq_key

_raw_voyage_key = (os.getenv("VOYAGE_API_KEY") or "").strip('"\' \n\t')
if _raw_voyage_key:
    os.environ["VOYAGE_API_KEY"] = _raw_voyage_key

DATA_DIR = BASE_DIR / "data"
RAW_JSONL_DIR = DATA_DIR / "raw_jsonl"
TARGET_REPOS_DIR = DATA_DIR / "target_repos"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure output directory exists
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    # Groq API for LLM Generation + Voyage AI API for Code Embeddings
    api_key: str = (os.getenv("GROQ_API_KEY") or "").strip('"\' \n\t')
    voyage_api_key: str = (os.getenv("VOYAGE_API_KEY") or "").strip('"\' \n\t')
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "voyage-code-4")
    temperature: float = 0.2
    max_tokens: int = 512


@dataclass
class RetrieverConfig:
    top_k: int = 5
    dense_weight: float = 0.5   # Weight for vector similarity
    sparse_weight: float = 0.5  # Weight for BM25 score
    graph_weight: float = 0.3   # Weight bonus for graph-connected nodes
    graph_expansion_hops: int = 1  # 1-hop or 2-hop neighbor expansion in Dependency Graph


@dataclass
class RepoCoderConfig:
    max_iterations: int = 2  # RepoCoder iterative retrieval-generation loops
    sliding_window_size: int = 10  # Context lines before cursor
    min_chunk_size: int = 3  # Minimum lines for a code chunk


@dataclass
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    repocoder: RepoCoderConfig = field(default_factory=RepoCoderConfig)
    target_repos_dir: Path = TARGET_REPOS_DIR
    raw_jsonl_dir: Path = RAW_JSONL_DIR
    outputs_dir: Path = OUTPUTS_DIR


config = ProjectConfig()