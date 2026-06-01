"""
aetherguard-rag-security — AetherGuard RAG Pipeline Security SDK.

Public API
----------
The following names are the stable public surface of this package:

- :class:`AetherGuardRAG`   — main async SDK client
- :class:`IngestResult`     — result of ``secure_ingest``
- :class:`RetrieveResult`   — result of ``secure_retrieve``
- :class:`ChunkMetadata`    — per-chunk metadata returned after ingestion
- :class:`SafeChunk`        — verified, sanitised chunk returned during retrieval
- :class:`AuthorizeResult`  — result of ``authorize``
- :class:`VerifyResult`     — result of ``verify_chunk``

Exceptions
----------
- :class:`AetherGuardError`     — base exception
- :class:`IngestError`          — ingestion failure
- :class:`RetrievalDeniedError` — retrieval authorisation denied (403)
- :class:`ConnectionError`      — backend-api unreachable
- :class:`AuthorizationError`   — authorisation failure (non-retrieval)
"""
from __future__ import annotations

from .client import AetherGuardRAG
from .exceptions import (
    AetherGuardError,
    AuthorizationError,
    ConnectionError,
    IngestError,
    RetrievalDeniedError,
)
from .models import (
    AuthorizeResult,
    ChunkInput,
    ChunkMetadata,
    IngestResult,
    RetrieveResult,
    SafeChunk,
    VerifyResult,
)

__all__ = [
    # Main client
    "AetherGuardRAG",
    # Input models
    "ChunkInput",
    # Result models
    "IngestResult",
    "RetrieveResult",
    "ChunkMetadata",
    "SafeChunk",
    "AuthorizeResult",
    "VerifyResult",
    # Exceptions
    "AetherGuardError",
    "IngestError",
    "RetrievalDeniedError",
    "ConnectionError",
    "AuthorizationError",
]

__version__ = "0.1.0"
