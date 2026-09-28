# Paper Atlas

Paper Atlas is a lightweight research-paper ingestion and retrieval project built around PDF parsing, chunking, and local semantic search with ChromaDB.

The project is currently focused on one core workflow:

- ingest a PDF
- convert it into document chunks with Docling
- embed and store those chunks in a persistent local ChromaDB collection
- support future retrieval/search over the indexed papers

## Project status

This repository is a working prototype in active development. The ingestion pipeline is implemented in `src/paper_atlas/ingestion.py`, while the retrieval layer is still a scaffold and the package entrypoint is minimal.

## Features

- PDF ingestion via Docling
- Content chunking with a hybrid chunker
- Persistent local vector storage with ChromaDB
- Deduplication based on SHA-256 file hash
- Chunk metadata for document ID, version ID, page range, and section information
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
│       ├── retrieval.py
│       └── schemas/
│           └── ingestion.py
└── .venv/
```

## Tech stack

- Python 3.11+
- ChromaDB for vector storage
- Docling for PDF parsing and chunking
- Pydantic models for metadata validation
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

print(result)  # IngestResult.INGESTED or IngestResult.SKIPPED
```

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

## Notes on behavior

- Files are de-duplicated using a SHA-256 hash, so re-ingesting the same PDF does not create duplicate chunks.
- The implementation stores data locally and persists it across runs.
- The collection is created automatically if it does not already exist.

## Future directions

Planned work includes:

- a proper CLI for ingestion and querying
- semantic retrieval over indexed papers
- filtering by title, section, and page range
- ranking and answer generation on top of retrieved chunks

## Contributing

Contributions are welcome. If you want to improve the pipeline, focus on:

1. finishing the retrieval layer
2. adding a usable command-line interface
3. improving metadata and search filtering
4. testing ingestion and indexing workflows on real paper sets

## License

This project does not currently specify a license in the repository metadata. If you plan to share or publish it externally, add a license file and update the package metadata accordingly.
