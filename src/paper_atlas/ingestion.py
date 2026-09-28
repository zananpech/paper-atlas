import hashlib
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import BaseChunk
from docling_core.types.doc import DoclingDocument

from paper_atlas.schemas.ingestion import ChunkingMetadata
from paper_atlas.guardrail import IngestionGuardrail

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = {"pdf"}
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class IngestResult(Enum):
    INGESTED = "ingested"
    SKIPPED = "skipped"


def hash_file(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Ingestion:
    """Class for handling different types of data ingestion."""

    def __init__(self, ingestion_type: str, db_path: str | Path = "./chroma_db") -> None:
        """Initialize an ingestion handler.

        Args:
            ingestion_type: The type of data to ingest. Currently only ``"pdf"``.
            db_path: Directory for the persistent ChromaDB store.

        Raises:
            ValueError: If the ingestion type is unsupported.
        """
        if ingestion_type not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported ingestion type: {ingestion_type}")

        self.ingestion_type = ingestion_type
        self.converter = DocumentConverter()
        self.chunker = HybridChunker()  # reused for chunk() and contextualize()
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)  
        self.collection = self.chroma_client.get_or_create_collection(COLLECTION_NAME, embedding_function=self.embedding_function)
        self.ingestion_guardrail = IngestionGuardrail()

    def ingest(self, path: Path) -> IngestResult:
        """Ingest a document.

        Returns:
            ``SKIPPED`` if this exact file was already ingested, else ``INGESTED``.

        Raises:
            Exception: Parse, chunk, and index errors propagate to the caller.
        """
        version_id = hash_file(path)  # hashed once, passed down

        if self._has_already_been_ingested(version_id):
            logger.info("Document %s has already been ingested.", path)
            return IngestResult.SKIPPED

        self._ingest_pdf(path, version_id)
        return IngestResult.INGESTED

    def _has_already_been_ingested(self, version_id: str) -> bool:
        result = self.collection.get(where={"version_id": version_id}, limit=1)
        return len(result["ids"]) > 0

    def _parse_pdf(self, path: Path) -> DoclingDocument:
        logger.info("Processing PDF document: %s", path)
        return self.converter.convert(path).document

    def _chunk_document(self, document: DoclingDocument) -> list[BaseChunk]:
        # chunk() is lazy, so materialize it here where errors actually surface
        logger.info("Chunking document...")
        return list(self.chunker.chunk(document))

    def _index_chunks(
        self, chunks: Iterable[BaseChunk], path: Path, version_id: str
    ) -> None:
        logger.info("Indexing chunks into ChromaDB...")

        title = path.stem
        doc_id = f"{title}-{version_id}"
        ingested_at = datetime.now(timezone.utc)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for i, chunk in enumerate(chunks):
            pages = sorted(
                {
                    prov.page_no
                    for item in chunk.meta.doc_items
                    for prov in (item.prov or [])
                }
            )
            headings = getattr(chunk.meta, "headings", None) or []
            labels = sorted(
                {
                    item.label.value if hasattr(item.label, "value") else str(item.label)
                    for item in chunk.meta.doc_items
                    if hasattr(item, "label")
                }
            )

            chunk_id = f"{doc_id}-chunk-{i}"
            ids.append(chunk_id)
            # headings + text, so section context is part of the embedding
            documents.append(self.chunker.contextualize(chunk))

            metadata = ChunkingMetadata(
                doc_id=doc_id,
                version_id=version_id,
                title=title,
                chunk_id=chunk_id,
                chunk_size=len(chunk.text),
                embedding_model=EMBEDDING_MODEL,
                prompt_injection_detected=self.ingestion_guardrail.has_suspicious_instructions(chunk.text),
                ingested_at=ingested_at,
                page_start=pages[0] if pages else None,
                page_end=pages[-1] if pages else None,
                section_title=headings[-1] if headings else None,
                content_type=", ".join(labels) if labels else None,
            )
            # Chroma rejects None metadata values
            metadatas.append(metadata.model_dump(mode="json", exclude_none=True))

        if not ids:
            logger.info("No chunks to index for %s", path)
            return

        batch = self.chroma_client.get_max_batch_size()
        for start in range(0, len(ids), batch):
            end = start + batch
            # upsert: safe to re-run after a partial failure
            self.collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
        logger.info("Indexed %d chunks successfully.", len(ids))

    def _ingest_pdf(self, path: Path, version_id: str) -> None:
        logger.info("Ingesting PDF data from %s", path)
        document = self._parse_pdf(path)
        chunks = self._chunk_document(document)
        self._index_chunks(chunks, path, version_id)
        logger.info("Ingestion completed for %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = Ingestion(ingestion_type="pdf")
    print(pipeline.ingest(Path("papers/sample_research_paper.pdf")))