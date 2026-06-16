from pydantic import BaseModel, Field
from typing import Any


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    document_id: str | None = None
    source_type: str | None = None


class IngestResponse(BaseModel):
    document_id: str
    total_chunks: int
    blocked_chunks: int
    saved_chunks: int
    evidence_id: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    evidence_id: str | None = None
    verified_count: int = 0
    blocked_count: int = 0
    sanitized_count: int = 0
    raw_retrieved: int = 0


class VerifyResponse(BaseModel):
    chunk_id: str
    exists: bool
    hash_valid: bool
    signature_valid: bool
    trust_level: str


class HealthResponse(BaseModel):
    status: str
    details: dict[str, Any]
