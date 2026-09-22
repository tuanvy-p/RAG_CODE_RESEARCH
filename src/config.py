import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables (.env with fallback to .env.example)
env_file = BASE_DIR / ".env"
if env_file.is_file():
    load_dotenv(env_file, override=True)
else:
    load_dotenv(BASE_DIR / ".env.example", override=True)

# Define directories
DATA_DIR = BASE_DIR / "data"
RAW_JSONL_DIR = DATA_DIR / "raw_jsonl"
TARGET_REPOS_DIR = DATA_DIR / "target_repos"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# BỔ SUNG: Tự động khởi tạo cả 2 thư mục chứa output và vector database
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    # Model local chính thức cho Kaggle GPU / Local
    llm_model: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    
    temperature: float = 0.2
    max_tokens: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: str = "float16"
    use_4bit: bool = False


@dataclass
class RetrieverConfig:
    top_k: int = 5
    dense_weight: float = 0.5   # Weight for vector similarity
    sparse_weight: float = 0.5  # Weight for BM25 score
    graph_weight: float = 0.3   # Weight bonus for graph-connected nodes
    graph_expansion_hops: int = 1  # 1-hop neighbor expansion in Dependency Graph


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
    chroma_db_dir: Path = CHROMA_DB_DIR  # BỔ SUNG: Khai báo đường dẫn ChromaDB vào config


config = ProjectConfig()