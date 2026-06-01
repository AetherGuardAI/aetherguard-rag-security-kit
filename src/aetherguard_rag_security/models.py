"""
Request/response dataclasses for aetherguard-rag-security SDK.

These models mirror the JSON shapes exchanged with backend-api's
/api/v1/rag/* endpoints.  They are plain Python dataclasses (no Pydantic
dependency at the model layer) so they remain lightweight and easy to
serialise/deserialise manually in the HTTP transport layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Ingestion models
# ---------------------------------------------------------------------------


@dataclass
class ChunkInput:
    """A chunk + embedding pair sent for ingestion."""

    content: str
    embedding: list[float]
    document_id: str | None = None
    position: int = 0


@dataclass
class ChunkMetadata:
    """
    Metadata returned by backend-api after ingestion.

    Store this alongside the vector in your vector database using
    :meth:`to_dict` so that AetherGuard can verify the chunk during
    retrieval.
    """

    chunk_id: str
    chunk_hash: str          # SHA-256 hex of content
    embedding_hash: str      # SHA-256 hex of normalised embedding
    signature: str           # base64 ECDSA-P256 signature
    classification: str      # public | internal | confidential | regulated
    trust_level: str         # verified | trusted | unverified
    injection_score: float   # 0.0–1.0
    pii_detected: bool
    secrets_detected: bool
    toxicity_score: float
    token_count: int
    region: str
    signed_at: str           # ISO 8601 timestamp

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a flat dict suitable for vector DB metadata storage.

        All keys are prefixed with ``ag_`` so they are namespaced and
        easy to identify in any vector database.
        """
        return {
            "ag_chunk_id": self.chunk_id,
            "ag_chunk_hash": self.chunk_hash,
            "ag_embedding_hash": self.embedding_hash,
            "ag_signature": self.signature,
            "ag_classification": self.classification,
            "ag_trust_level": self.trust_level,
            "ag_injection_score": self.injection_score,
            "ag_pii_detected": self.pii_detected,
            "ag_secrets_detected": self.secrets_detected,
            "ag_toxicity_score": self.toxicity_score,
            "ag_token_count": self.token_count,
            "ag_region": self.region,
            "ag_signed_at": self.signed_at,
        }


@dataclass
class IngestResult:
    """Result of a :meth:`~aetherguard_rag_security.AetherGuardRAG.secure_ingest` call."""

    metadata: list[ChunkMetadata]
    document_id: str
    total_chunks: int
    blocked_chunks: int   # chunks that failed scanning
    evidence_id: str      # immudb evidence reference


# ---------------------------------------------------------------------------
# Retrieval models
# ---------------------------------------------------------------------------


@dataclass
class SafeChunk:
    """A verified, scanned, and sanitised chunk returned during retrieval."""

    chunk_id: str
    content: str           # sanitised content
    trust_level: str
    classification: str
    relevance_score: float
    token_count: int


@dataclass
class RetrieveResult:
    """Result of a :meth:`~aetherguard_rag_security.AetherGuardRAG.secure_retrieve` call."""

    text: str                  # assembled safe context string
    chunks: list[SafeChunk]   # individual verified chunks
    total_retrieved: int       # raw chunks submitted
    verified_count: int        # passed verification
    blocked_count: int         # failed verification or scanning
    sanitized_count: int       # required sanitisation
    token_count: int           # total tokens in assembled context
    evidence_id: str           # immudb evidence reference


# ---------------------------------------------------------------------------
# Authorization / verification models
# ---------------------------------------------------------------------------


@dataclass
class AuthorizeResult:
    """Result of an :meth:`~aetherguard_rag_security.AetherGuardRAG.authorize` call."""

    authorized: bool
    namespace: str | None = None
    allowed_classifications: list[str] = field(default_factory=list)
    denial_reason: str | None = None


@dataclass
class VerifyResult:
    """Result of a :meth:`~aetherguard_rag_security.AetherGuardRAG.verify_chunk` call."""

    chunk_id: str
    hash_valid: bool
    signature_valid: bool
    trust_level: str
    exists: bool   # whether the chunk exists in PostgreSQL
