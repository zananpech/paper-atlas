
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ChunkingMetadata(BaseModel):
    """
    Metadata for a chunk of text.
    """

    doc_id: str
    version_id: str
    title: str
    chunk_id: str
    chunk_size: int
    ingested_at: datetime
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_title: Optional[str] = None
    content_type: Optional[str] = None