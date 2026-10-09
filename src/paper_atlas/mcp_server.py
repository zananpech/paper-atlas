"""MCP server for searching and ingesting papers in Paper Atlas."""

import logging
from pathlib import Path

from fastmcp import FastMCP

from paper_atlas.ingestion import Ingestion
from paper_atlas.retriever import Retriever
from paper_atlas.config import DB_PATH

logger = logging.getLogger(__name__)


# FastMCP provides the MCP server runtime. LangChain is used for the vector
# store, embeddings, and hybrid retrieval pipeline.
mcp = FastMCP("paper-atlas")
_retrieval: Retriever | None = None
_ingestion: Ingestion | None = None

# lazy initialization of Retriever and Ingestion to avoid ChromaDB connection on import
def _get_retrieval() -> Retriever:
    global _retrieval
    if _retrieval is None:
        _retrieval = Retriever(db_path=Path(DB_PATH))
    return _retrieval


def _get_ingestion() -> Ingestion:
    global _ingestion
    if _ingestion is None:
        _ingestion = Ingestion(ingestion_type="pdf", db_path=Path(DB_PATH))
    return _ingestion


@mcp.tool()
def search_papers(query: str, n_results: int = 10) -> list[dict]:
    """Search indexed paper chunks for a natural-language query."""
    results = _get_retrieval().retrieve(query, n_results=n_results)
    return [
        {
            "title": result["metadata"].get("title"),
            "text": result["text"],
            "section_title": result["metadata"].get("section_title"),
            "page_start": result["metadata"].get("page_start"),
            "page_end": result["metadata"].get("page_end"),
        }
        for result in results
    ]

@mcp.tool()
def ingest_pdf(path: str) -> str:
    """Ingest a local PDF into the Paper Atlas ChromaDB index."""
    pdf_path = Path(path).expanduser()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    result = _get_ingestion().ingest(pdf_path)
    if result is None:
        _get_retrieval().refresh_index()
        return f"Skipped already-indexed PDF: {pdf_path}"

    _get_retrieval().refresh_index()
    return f"Ingested PDF: {pdf_path}"


def main() -> None:
    """Start the MCP server over stdio."""
    logging.basicConfig(level=logging.INFO)
    mcp.run(transport="http", host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
