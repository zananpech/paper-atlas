import hashlib
import logging
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_docling import DoclingLoader
from langchain_docling.loader import BaseMetaExtractor, ExportType

from paper_atlas.schemas.ingestion import ChunkingMetadata
from paper_atlas.guardrail import IngestionGuardrail
from paper_atlas.config import COLLECTION_NAME, EMBEDDING_MODEL, SUPPORTED_TYPES

logger = logging.getLogger(__name__)


class PaperMetaExtractor(BaseMetaExtractor):
    """Expose Docling chunk metadata as simple LangChain document fields."""

    def extract_chunk_meta(self, file_path: str, chunk) -> dict:
        meta = chunk.meta
        pages = sorted(
            {
                provenance.page_no
                for item in meta.doc_items
                for provenance in (item.prov or [])
            }
        )
        headings = getattr(meta, "headings", None) or []
        labels = sorted(
            {
                item.label.value if hasattr(item.label, "value") else str(item.label)
                for item in meta.doc_items
                if hasattr(item, "label")
            }
        )
        return {
            "source": file_path,
            "page_start": pages[0] if pages else None,
            "page_end": pages[-1] if pages else None,
            "section_title": headings[-1] if headings else None,
            "content_type": ", ".join(labels) if labels else None,
            "chunk_size": len(chunk.text),
            "dl_meta": meta.export_json_dict(),
        }

    def extract_dl_doc_meta(self, file_path: str, dl_doc) -> dict:
        return {"source": file_path}


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
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(db_path),
            embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        )
        self.batch_size = self.vector_store._client.get_max_batch_size()
        self.ingestion_guardrail = IngestionGuardrail()

    def ingest(self, path: Path):
        """Ingest a document.

        Returns:
            ``SKIPPED`` if this exact file was already ingested, else ``INGESTED``.

        Raises:
            Exception: Parse, chunk, and index errors propagate to the caller.
        """
        version_id = hash_file(path)  # hashed once, passed down

        if self._has_already_been_ingested(version_id):
            logger.info("Document %s has already been ingested.", path)
            return 

        self._ingest_pdf(path, version_id)
        return

    def _has_already_been_ingested(self, version_id: str) -> bool:
        result = self.vector_store.get(where={"version_id": version_id}, limit=1)
        return bool(result["ids"])

    def _load_documents(self, path: Path) -> list[Document]:
        logger.info("Parsing and chunking PDF with DoclingLoader: %s", path)
        loader = DoclingLoader(
            file_path=str(path),
            export_type=ExportType.DOC_CHUNKS,
            meta_extractor=PaperMetaExtractor(),
        )
        return loader.load()

    def _index_documents(
        self, source_documents: Iterable[Document], path: Path, version_id: str
    ) -> None:
        logger.info("Indexing chunks into ChromaDB...")

        title = path.stem
        doc_id = f"{title}-{version_id}"
        ingested_at = datetime.now(timezone.utc)

        ids: list[str] = []
        documents: list[Document] = []

        for i, source_document in enumerate(source_documents):
            chunk_id = f"{doc_id}-chunk-{i}"
            ids.append(chunk_id)
            source_metadata = source_document.metadata

            metadata = ChunkingMetadata(
                doc_id=doc_id,
                version_id=version_id,
                title=title,
                chunk_id=chunk_id,
                chunk_size=source_metadata.get("chunk_size", len(source_document.page_content)),
                embedding_model=EMBEDDING_MODEL,
                prompt_injection_detected=self.ingestion_guardrail.has_suspicious_instructions(source_document.page_content),
                ingested_at=ingested_at,
                page_start=source_metadata.get("page_start"),
                page_end=source_metadata.get("page_end"),
                section_title=source_metadata.get("section_title"),
                content_type=source_metadata.get("content_type"),
            )
            # Chroma rejects None metadata values
            documents.append(
                Document(
                    page_content=source_document.page_content,
                    metadata=metadata.model_dump(mode="json", exclude_none=True),
                )
            )

        if not ids:
            logger.info("No chunks to index for %s", path)
            return

        # add_documents updates existing IDs, making retries safe after partial failure.
        for start in range(0, len(ids), self.batch_size):
            end = start + self.batch_size
            self.vector_store.add_documents(documents[start:end], ids=ids[start:end])
        logger.info("Indexed %d chunks successfully.", len(ids))

    def _ingest_pdf(self, path: Path, version_id: str) -> None:
        logger.info("Ingesting PDF data from %s", path)
        documents = self._load_documents(path)
        self._index_documents(documents, path, version_id)
        logger.info("Ingestion completed for %s", path)

def main() -> None:
    """CLI entry point for paper ingestion."""

    logging.basicConfig(level=logging.INFO)
    pdf_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("papers/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks-with-annotations.pdf")
    )

    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        sys.exit(1)

    pipeline = Ingestion(ingestion_type="pdf")
    pipeline.ingest(pdf_path)


if __name__ == "__main__":
    main()

