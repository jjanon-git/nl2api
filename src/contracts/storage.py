"""
Storage Models

Azure Table Storage specific models and helpers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from src.contracts.core import _now_utc


# =============================================================================
# Table Storage Entity
# =============================================================================


class TableStorageEntity(BaseModel):
    """
    Mixin for Azure Table Storage compatibility.

    Provides partition/row key computation.
    """

    model_config = ConfigDict(frozen=True)

    @computed_field
    @property
    def partition_key(self) -> str:
        """Override in subclass."""
        raise NotImplementedError

    @computed_field
    @property
    def row_key(self) -> str:
        """Override in subclass."""
        raise NotImplementedError

    def to_table_entity(self) -> dict[str, Any]:
        """Convert to Azure Table Storage entity format."""
        data = self.model_dump(mode="json")
        data["PartitionKey"] = self.partition_key
        data["RowKey"] = self.row_key
        return data


# =============================================================================
# Idempotency Record
# =============================================================================


class IdempotencyRecord(BaseModel):
    """
    Record for tracking processed tasks (idempotency).

    Stored in Azure Table Storage.
    """

    model_config = ConfigDict(frozen=True)

    idempotency_key: str = Field(..., description="Unique processing key")
    scorecard_id: str = Field(..., description="Resulting scorecard ID")
    processed_at: datetime = Field(default_factory=_now_utc)
    worker_id: str = Field(..., description="Worker that processed this")

    @computed_field
    @property
    def partition_key(self) -> str:
        """Partition by first 2 chars of key for distribution."""
        return self.idempotency_key[:2] if len(self.idempotency_key) >= 2 else "00"

    @computed_field
    @property
    def row_key(self) -> str:
        """Full idempotency key as row key."""
        return self.idempotency_key


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TableStorageEntity",
    "IdempotencyRecord",
]
