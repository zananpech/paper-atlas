# Paper Atlas

Paper Atlas is a lightweight research-paper ingestion and retrieval project built around PDF parsing, chunking, and local hybrid search with LangChain and ChromaDB.

The project supports two core workflows:

- ingest a PDF
- parse and chunk it with Docling through LangChain's `DoclingLoader`
- embed and store those chunks in a persistent local ChromaDB collection
- search indexed papers with hybrid semantic and keyword retrieval
- expose ingestion and retrieval through an MCP server

## Project status

This repository is a working prototype in active development. LangChain provides the document loader, embeddings, and Chroma vector-store integration. FastMCP hosts the MCP tools.

## Features

- PDF ingestion via LangChain's Docling integration
- Docling `HybridChunker` chunking through `DoclingLoader` (the default chunker)
- Persistent local vector storage through LangChain's Chroma integration
- Deduplication based on SHA-256 file hash
- Chunk metadata for document ID, version ID, page range, and section information
- LangChain Hugging Face embeddings with semantic and BM25 retrieval fused by reciprocal rank fusion
- A lightweight ingestion guardrail flags chunks matching prompt-injection patterns; it records a metadata flag but does not block or modify content
- FastMCP server for use with MCP-compatible clients
- Local storage under `./chroma_db`

## Repository structure

```text
paper-atlas/
├── README.md
├── pyproject.toml
├── uv.lock
├── chroma_db/
│   └── persistent ChromaDB data
├── papers/
│   └── research PDFs go here
├── src/
│   └── paper_atlas/
│       ├── __init__.py
│       ├── ingestion.py
│       ├── mcp_server.py
│       ├── retrieval.py
│       └── schemas/
│           └── ingestion.py
└── .venv/
```

## Tech stack

- Python 3.11+
- LangChain Chroma for vector storage and semantic retrieval
- LangChain Hugging Face for local sentence-transformer embeddings
- LangChain Docling (`DoclingLoader`) and Docling for PDF parsing and chunking
- Pydantic models for metadata validation
- FastMCP for the local MCP server
- uv for dependency and project management

## Installation

Using `uv` is the recommended path for this project:

```bash
cd paper-atlas
uv sync
```

If you prefer a standard virtual environment:

```bash
cd paper-atlas
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows
pip install -U pip
pip install -e .
```

## Usage

### Ingest a PDF

```python
from pathlib import Path
from paper_atlas.ingestion import Ingestion

pipeline = Ingestion(ingestion_type="pdf", db_path="./chroma_db")
result = pipeline.ingest(Path("papers/sample_research_paper.pdf"))

print(result)  # None (the current method does not return an ingestion status)
```

`DoclingLoader` uses Docling's `HybridChunker` by default and returns LangChain `Document` objects. Paper Atlas adds stable IDs and metadata to each chunk before indexing it. The `prompt_injection_detected` metadata field is set by a small regex-based check; a match is recorded, not blocked.

This creates or reuses a ChromaDB collection named `research_papers` and stores indexed chunks with metadata such as:

- `doc_id`
- `version_id`
- `title`
- `chunk_id`
- `page_start`
- `page_end`
- `section_title`
- `content_type`
- `ingested_at`

### Running the ingestion module directly

```bash
python -m paper_atlas.ingestion
```

This will attempt to ingest a sample file from `papers/sample_research_paper.pdf` if it exists.

### Run the MCP server

Start the FastMCP server over HTTP on `127.0.0.1:8000`:

```bash
uv run paper-atlas-mcp
```

The server exposes these tools:

- `ingest_pdf(path)`: ingest a local PDF into the ChromaDB index
- `search_papers(query, n_results=10)`: search indexed paper chunks and return their title, text, section, and page range

FastMCP provides the MCP server runtime; LangChain's MCP adapters are not used to host it. The server uses the database path configured by `paper_atlas.config.DB_PATH` and can be connected to from an MCP-compatible client using the server URL exposed by FastMCP.

## Notes on behavior

- Files are de-duplicated using a SHA-256 hash, so re-ingesting the same PDF does not create duplicate chunks.
- The implementation stores data locally and persists it across runs.
- The collection is created automatically if it does not already exist.

## Future directions

Planned work includes:

- a proper CLI for ingestion and querying
- filtering by title, section, and page range
- ranking and answer generation on top of retrieved chunks

## Contributing

Contributions are welcome. If you want to improve the pipeline, focus on:

1. adding a usable command-line interface
2. improving metadata and search filtering
3. testing ingestion and indexing workflows on real paper sets

## License

This project does not currently specify a license in the repository metadata. If you plan to share or publish it externally, add a license file and update the package metadata accordingly.
