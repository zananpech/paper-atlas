import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
CONFIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONFIG_DIR.parent.parent

# Load local environment overrides if present
for _env_file in [
    CONFIG_DIR / ".env.local",
    REPO_ROOT / ".env.local",
    CONFIG_DIR / ".env",
    REPO_ROOT / ".env",
]:
    if _env_file.is_file():
        load_dotenv(dotenv_path=_env_file, override=False)

SUPPORTED_TYPES = {"pdf"}
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DB_PATH = "chroma_db"

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")