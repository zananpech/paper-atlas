import json
from pathlib import Path
from pydantic import BaseModel, Field


class EvalQuery(BaseModel):
    """A single evaluation query sample."""

    query_id: str | None = Field(default=None, description="Unique identifier for the query")
    query: str = Field(min_length=1, description="Search query string")
    relevant_doc_ids: list[str] = Field(
        default_factory=list,
        description="Ground truth relevant document/chunk IDs (empty if doc does not exist)",
    )
    category: str | None = Field(
        default=None,
        description="Scenario type: 'one_hop', 'multi_hop', or 'not_exist'",
    )


class EvalDataset(BaseModel):
    """Collection of evaluation query samples."""

    samples: list[EvalQuery] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, index: int) -> EvalQuery:
        return self.samples[index]

    @classmethod
    def load(cls, path: Path | str) -> "EvalDataset":
        """Load and validate an evaluation dataset from a JSON file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Evaluation dataset file not found at: {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a list of query items in {file_path}, got {type(data).__name__}")

        samples = [EvalQuery(**item) for item in data]
        return cls(samples=samples)
