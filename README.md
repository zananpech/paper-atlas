# Paper Atlas

Paper Atlas is a lightweight research-paper ingestion and retrieval project built around PDF parsing, chunking, local hybrid search with LangChain and ChromaDB, cross-encoder re-ranking, and Langfuse observability.

The project supports core workflows:

- Ingest research PDFs, parse and chunk them with Docling through LangChain's `DoclingLoader`
- Embed and store chunks in a persistent local ChromaDB collection
- Search indexed papers with hybrid retrieval (semantic vector search + BM25 keyword search fused by RRF)
- Re-rank candidate chunks using a cross-encoder model
- Evaluate retrieval quality with an offline benchmark dataset and metrics (Precision@K, Recall@K, HitRate@K, MRR, NDCG@K)
- Trace queries and log evaluation metrics to Langfuse
- Expose ingestion and retrieval tools through a FastMCP server

## Project status

This repository is a working prototype in active development. LangChain provides the document loader, embeddings, and Chroma vector-store integration. FastMCP hosts the MCP tools, and Langfuse tracks observability and evaluation metrics.

## Features

- **PDF Ingestion**: Fast parsing and chunking via LangChain's Docling integration (`DoclingLoader` with Docling `HybridChunker`).
- **Vector Storage**: Persistent local storage through LangChain's Chroma integration under `./chroma_db`.
- **Deduplication**: SHA-256 file hash checks ensure re-ingesting the same PDF does not duplicate chunks.
- **Rich Metadata**: Captures `doc_id`, `version_id`, `title`, `chunk_id`, page ranges, section titles, and content type.
- **Hybrid Retrieval + Re-ranking**:
  - Semantic embeddings via `BAAI/bge-small-en-v1.5`
  - Lexical keyword search via BM25
  - Reciprocal Rank Fusion (RRF) to combine candidate sets
  - Re-ranking via Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Ingestion Guardrail**: Lightweight regex check flagging prompt-injection patterns in metadata without mutating content.
- **Retrieval Evaluation**: Built-in evaluation runner measuring Precision@K, Recall@K, HitRate@K, MRR, and NDCG@K across categorized scenarios.
- **Langfuse Observability**: Traces `retrieve()` queries and records per-query evaluation scores with metadata.
- **MCP Server**: FastMCP server exposing tools for MCP-compatible clients.

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
│       ├── .env.example
│       ├── __init__.py
│       ├── config.py
│       ├── guardrail.py
│       ├── ingestion.py
│       ├── langfuse_client.py
│       ├── mcp_server.py
│       ├── retriever.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── evaluation_dataset.json
│       │   ├── metrics.py
│       │   └── retriever_eval.py
│       └── schemas/
│           ├── evaluation.py
│           └── ingestion.py
└── .venv/
```

## Tech stack

- Python 3.11+
- LangChain Chroma for vector storage and semantic retrieval
- LangChain Hugging Face for local sentence-transformer embeddings
- LangChain Docling (`DoclingLoader`) and Docling for PDF parsing and chunking
- Sentence Transformers Cross-Encoder for candidate re-ranking
- Rank-BM25 for lexical keyword matching
- Langfuse for tracing and evaluation logging
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

### Environment Configuration (Langfuse)

To enable Langfuse tracing and evaluation metric logging, create a `.env.local` file from the provided example:

```bash
cp src/paper_atlas/.env.example src/paper_atlas/.env.local
```

Configure your credentials inside `src/paper_atlas/.env.local`:

```env
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_BASE_URL="https://us.cloud.langfuse.com"
```

Verify your connection anytime:

```bash
uv run python -m paper_atlas.langfuse_client
```

## Usage

### Ingest a PDF

```python
from pathlib import Path
from paper_atlas.ingestion import Ingestion

pipeline = Ingestion(ingestion_type="pdf", db_path="./chroma_db")
pipeline.ingest(Path("papers/sample_research_paper.pdf"))
```

Or run the ingestion CLI command:

```bash
uv run paper-atlas-ingest
```

### Search Papers (Hybrid + Re-ranking)

```bash
uv run paper-atlas-retrieve "your query here"
```

In Python:

```python
from pathlib import Path
from paper_atlas.retriever import Retriever

retriever = Retriever(db_path=Path("chroma_db"))
results = retriever.retrieve("What is the proposed attention mechanism?", n_results=5)

for res in results:
    print(f"[{res['id']}] Rerank Score: {res['rerank_score']:.4f} | RRF Score: {res['rrf_score']:.4f}")
    print(res["text"][:200])
```

### Evaluate Retrieval Quality

Run the benchmark evaluation across test queries and log scores to Langfuse:

```bash
uv run paper-atlas-eval -k 5
```

Options:
- `-k`: Cutoff depth for retrieval and metrics (default: `5`).
- `--dataset`: Path to a custom evaluation JSON dataset (defaults to `evaluation_dataset.json`).

Output includes per-query metrics, overall summary (Precision, Recall, Hit Rate, MRR, NDCG), and category breakdowns, with metrics automatically sent to Langfuse.

### Run the MCP Server

Start the FastMCP server over HTTP on `127.0.0.1:8000`:

```bash
uv run paper-atlas-mcp
```

The server exposes:
- `ingest_pdf(path)`: Ingest a local PDF into the ChromaDB index.
- `search_papers(query, n_results=10)`: Search indexed chunks with hybrid retrieval and re-ranking.

## License

This project does not currently specify a license in the repository metadata. If you plan to share or publish it externally, add a license file and update the package metadata accordingly.
