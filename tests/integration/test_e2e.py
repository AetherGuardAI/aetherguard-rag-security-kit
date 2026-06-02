"""
Integration tests for AetherGuardRAG SDK (using respx HTTP mocking).

Tasks 5.1–5.4:
  5.1 — secure_ingest integration tests (Requirements 1.1–1.5, 12.1)
  5.2 — secure_retrieve integration tests (Requirements 2.1–2.4, 12.1)
  5.3 — authorize and verify_chunk integration tests (Requirements 6.1–6.7)
  5.4 — async context manager integration tests (Requirements 13.1, 13.2)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from aetherguard_rag_security import AetherGuardRAG
from aetherguard_rag_security.exceptions import (
    ConnectionError as AetherConnectionError,
    IngestError,
    RetrievalDeniedError,
)

# ---------------------------------------------------------------------------
# Shared mock response bodies
# ---------------------------------------------------------------------------

INGEST_200 = {
    "metadata": [
        {
            "chunk_id": "chunk-001",
            "chunk_hash": "a" * 64,
            "embedding_hash": "b" * 64,
            "signature": "sig1==",
            "classification": "public",
            "trust_level": "verified",
            "injection_score": 0.01,
            "pii_detected": False,
            "secrets_detected": False,
            "toxicity_score": 0.02,
            "token_count": 50,
            "region": "us-east-1",
            "signed_at": "2024-01-01T00:00:00",
        },
        {
            "chunk_id": "chunk-002",
            "chunk_hash": "c" * 64,
            "embedding_hash": "d" * 64,
            "signature": "sig2==",
            "classification": "internal",
            "trust_level": "verified",
            "injection_score": 0.0,
            "pii_detected": False,
            "secrets_detected": False,
            "toxicity_score": 0.0,
            "token_count": 60,
            "region": "us-east-1",
            "signed_at": "2024-01-01T00:00:01",
        },
    ],
    "document_id": "doc-001",
    "total_chunks": 2,
    "blocked_chunks": 0,
    "evidence_id": "ev-001",
}

RETRIEVE_200 = {
    "text": "Safe context text",
    "chunks": [
        {
            "chunk_id": "chunk-001",
            "content": "Safe chunk 1",
            "trust_level": "verified",
            "classification": "public",
            "relevance_score": 0.95,
            "token_count": 80,
        },
        {
            "chunk_id": "chunk-002",
            "content": "Safe chunk 2",
            "trust_level": "verified",
            "classification": "internal",
            "relevance_score": 0.88,
            "token_count": 70,
        },
    ],
    "total_retrieved": 2,
    "verified_count": 2,
    "blocked_count": 0,
    "sanitized_count": 0,
    "token_count": 150,
    "evidence_id": "ev-002",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_URL = "https://api.aetherguard.test"


def _make_client(**kwargs) -> AetherGuardRAG:
    return AetherGuardRAG(api_url=_BASE_URL, api_key="test-key", **kwargs)


# ===========================================================================
# Task 5.1 — secure_ingest integration tests
# Requirements 1.1–1.5, 12.1
# ===========================================================================


async def test_secure_ingest_happy_path() -> None:
    """
    Mock 200 response; verify IngestResult fields are parsed correctly.

    Validates: Requirements 1.1, 1.2
    """
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/api/v1/rag/secure-ingest").mock(
            return_value=httpx.Response(200, json=INGEST_200)
        )

        async with _make_client() as ag:
            result = await ag.secure_ingest(
                chunks=["chunk text one", "chunk text two"],
                embeddings=[[0.1, 0.2], [0.3, 0.4]],
                tenant_id="acme",
                region="us-east-1",
                document_id="doc-001",
            )

    assert result.total_chunks == 2
    assert result.blocked_chunks == 0
    assert result.document_id == "doc-001"
    assert result.evidence_id == "ev-001"
    assert len(result.metadata) == 2
    assert result.metadata[0].chunk_id == "chunk-001"
    assert result.metadata[0].trust_level == "verified"
    assert result.metadata[0].pii_detected is False


async def test_secure_ingest_400_raises_ingest_error() -> None:
    """
    Mock 400 with {"detail": "invalid tenant"}; verify IngestError raised
    with status_code=400 and detail containing "invalid tenant".

    Validates: Requirement 1.4
    """
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/api/v1/rag/secure-ingest").mock(
            return_value=httpx.Response(400, json={"detail": "invalid tenant"})
        )

        async with _make_client() as ag:
            with pytest.raises(IngestError) as exc_info:
                await ag.secure_ingest(
                    chunks=["text"],
                    embeddings=[[0.1]],
                    tenant_id="bad tenant!",
                    region="us-east-1",
                )

    err = exc_info.value
    assert err.status_code == 400
    assert "invalid tenant" in (err.detail or "")


async def test_secure_ingest_length_mismatch_no_http() -> None:
    """
    3 chunks, 2 embeddings → ValueError raised, 0 HTTP calls.

    Validates: Requirement 1.3
    """
    with respx.mock(base_url=_BASE_URL, assert_all_called=False) as mock:
        mock.post("/api/v1/rag/secure-ingest").mock(
            return_value=httpx.Response(200, json=INGEST_200)
        )

        async with _make_client() as ag:
            with pytest.raises(ValueError):
                await ag.secure_ingest(
                    chunks=["a", "b", "c"],
                    embeddings=[[0.1], [0.2]],
                    tenant_id="acme",
                    region="us-east-1",
                )

        # No HTTP call should have been made
        assert mock.calls.call_count == 0


async def test_secure_ingest_timeout_raises_connection_error() -> None:
    """
    Mock TimeoutException; verify ConnectionError raised.
    Uses max_retries=1 and patches asyncio.sleep to avoid real delays.

    Validates: Requirement 12.1
    """
    with patch(
        "aetherguard_rag_security._http.asyncio.sleep",
        new=AsyncMock(return_value=None),
    ):
        with respx.mock(base_url=_BASE_URL) as mock:
            mock.post("/api/v1/rag/secure-ingest").mock(
                side_effect=httpx.TimeoutException("timeout")
            )

            async with _make_client(max_retries=1) as ag:
                with pytest.raises(AetherConnectionError):
                    await ag.secure_ingest(
                        chunks=["text"],
                        embeddings=[[0.1]],
                        tenant_id="acme",
                        region="us-east-1",
                    )


# ===========================================================================
# Task 5.2 — secure_retrieve integration tests
# Requirements 2.1–2.4, 12.1
# ===========================================================================


async def test_secure_retrieve_happy_path() -> None:
    """
    Mock 200 response; verify RetrieveResult fields are parsed correctly.

    Validates: Requirements 2.1, 2.2
    """
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/api/v1/rag/secure-retrieve").mock(
            return_value=httpx.Response(200, json=RETRIEVE_200)
        )

        async with _make_client() as ag:
            result = await ag.secure_retrieve(
                raw_results=[
                    {"chunk_id": "chunk-001", "content": "Safe chunk 1", "metadata": {}},
                    {"chunk_id": "chunk-002", "content": "Safe chunk 2", "metadata": {}},
                ],
                email="user@acme.com",
                region="us-east-1",
            )

    assert result.text == "Safe context text"
    assert result.total_retrieved == 2
    assert result.verified_count == 2
    assert result.blocked_count == 0
    assert result.sanitized_count == 0
    assert result.token_count == 150
    assert result.evidence_id == "ev-002"
    assert len(result.chunks) == 2
    assert result.chunks[0].chunk_id == "chunk-001"
    assert result.chunks[0].trust_level == "verified"


async def test_secure_retrieve_403_raises_retrieval_denied() -> None:
    """
    Mock 403 with {"reason": "insufficient role"}; verify RetrievalDeniedError
    raised with denial_reason="insufficient role".

    Validates: Requirement 2.3
    """
    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/api/v1/rag/secure-retrieve").mock(
            return_value=httpx.Response(403, json={"reason": "insufficient role"})
        )

        async with _make_client() as ag:
            with pytest.raises(RetrievalDeniedError) as exc_info:
                await ag.secure_retrieve(
                    raw_results=[{"chunk_id": "c1", "content": "x", "metadata": {}}],
                    email="guest@acme.com",
                    region="us-east-1",
                )

    err = exc_info.value
    assert err.denial_reason == "insufficient role"


async def test_secure_retrieve_timeout_raises_connection_error() -> None:
    """
    Mock TimeoutException; verify ConnectionError raised.
    Uses max_retries=1 and patches asyncio.sleep.

    Validates: Requirement 12.1
    """
    with patch(
        "aetherguard_rag_security._http.asyncio.sleep",
        new=AsyncMock(return_value=None),
    ):
        with respx.mock(base_url=_BASE_URL) as mock:
            mock.post("/api/v1/rag/secure-retrieve").mock(
                side_effect=httpx.TimeoutException("timeout")
            )

            async with _make_client(max_retries=1) as ag:
                with pytest.raises(AetherConnectionError):
                    await ag.secure_retrieve(
                        raw_results=[{"chunk_id": "c1", "content": "x", "metadata": {}}],
                        email="user@acme.com",
                        region="us-east-1",
                    )


# ===========================================================================
# Task 5.3 — authorize and verify_chunk integration tests
# Requirements 6.1–6.7
# ===========================================================================


async def test_authorize_happy_path() -> None:
    """
    Mock 200 with authorized=true; verify all AuthorizeResult fields.

    Validates: Requirements 6.1, 6.2
    """
    authorize_200 = {
        "authorized": True,
        "namespace": "acme_us-east-1",
        "allowed_classifications": ["public", "internal"],
        "denial_reason": None,
    }

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/api/v1/rag/authorize").mock(
            return_value=httpx.Response(200, json=authorize_200)
        )

        async with _make_client() as ag:
            result = await ag.authorize(
                tenant_id="acme",
                user_id="user-1",
                role="reader",
                region="us-east-1",
            )

    assert result.authorized is True
    assert result.namespace == "acme_us-east-1"
    assert result.allowed_classifications == ["public", "internal"]
    assert result.denial_reason is None


async def test_authorize_denied() -> None:
    """
    Mock 403 with authorized=false; verify AuthorizationError raised with
    denial_reason="role not permitted".

    Validates: Requirements 6.4, 6.5
    """
    authorize_403 = {
        "authorized": False,
        "denial_reason": "role not permitted",
    }

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.post("/api/v1/rag/authorize").mock(
            return_value=httpx.Response(403, json=authorize_403)
        )

        async with _make_client() as ag:
            from aetherguard_rag_security.exceptions import AuthorizationError
            with pytest.raises(AuthorizationError) as exc_info:
                await ag.authorize(
                    tenant_id="acme",
                    user_id="user-1",
                    role="unknown-role",
                    region="us-east-1",
                )

    err = exc_info.value
    assert err.denial_reason == "role not permitted"


async def test_verify_chunk_exists_and_valid() -> None:
    """
    Mock 200 with a valid chunk; verify all VerifyResult fields.

    Validates: Requirements 6.1, 6.2
    """
    verify_200 = {
        "chunk_id": "chunk-abc",
        "hash_valid": True,
        "signature_valid": True,
        "trust_level": "verified",
        "exists": True,
    }

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.get("/api/v1/rag/verify/chunk-abc").mock(
            return_value=httpx.Response(200, json=verify_200)
        )

        async with _make_client() as ag:
            result = await ag.verify_chunk("chunk-abc")

    assert result.chunk_id == "chunk-abc"
    assert result.hash_valid is True
    assert result.signature_valid is True
    assert result.trust_level == "verified"
    assert result.exists is True


async def test_verify_chunk_not_found() -> None:
    """
    Mock 404 for a missing chunk; verify result.exists=False.

    Validates: Requirement 6.4
    """
    verify_404 = {
        "chunk_id": "missing-chunk",
        "hash_valid": False,
        "signature_valid": False,
        "trust_level": "unverified",
        "exists": False,
    }

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.get("/api/v1/rag/verify/missing-chunk").mock(
            return_value=httpx.Response(404, json=verify_404)
        )

        async with _make_client() as ag:
            result = await ag.verify_chunk("missing-chunk")

    assert result.exists is False
    assert result.chunk_id == "missing-chunk"
    assert result.hash_valid is False
    assert result.signature_valid is False
    assert result.trust_level == "unverified"


# ===========================================================================
# Task 5.4 — async context manager integration tests
# Requirements 13.1, 13.2
# ===========================================================================


async def test_async_context_manager_closes_client() -> None:
    """
    Use `async with AetherGuardRAG(...) as ag:`, make one verify_chunk call
    inside, then verify the underlying httpx client is closed after the block.

    Validates: Requirements 13.1, 13.2
    """
    verify_200 = {
        "chunk_id": "chunk-abc",
        "hash_valid": True,
        "signature_valid": True,
        "trust_level": "verified",
        "exists": True,
    }

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.get("/api/v1/rag/verify/chunk-abc").mock(
            return_value=httpx.Response(200, json=verify_200)
        )

        ag = _make_client()
        async with ag:
            await ag.verify_chunk("chunk-abc")

    assert ag._transport._client.is_closed is True


async def test_explicit_close() -> None:
    """
    Create client, call `await ag.close()`, verify the underlying httpx
    client is closed.

    Validates: Requirement 13.2
    """
    ag = _make_client()
    assert ag._transport._client.is_closed is False

    await ag.close()

    assert ag._transport._client.is_closed is True


async def test_context_manager_closes_on_exception() -> None:
    """
    Use `async with` block that raises RuntimeError inside; catch it and
    verify the underlying httpx client is still closed.

    Validates: Requirements 13.1, 13.2
    """
    verify_200 = {
        "chunk_id": "chunk-abc",
        "hash_valid": True,
        "signature_valid": True,
        "trust_level": "verified",
        "exists": True,
    }

    with respx.mock(base_url=_BASE_URL) as mock:
        mock.get("/api/v1/rag/verify/chunk-abc").mock(
            return_value=httpx.Response(200, json=verify_200)
        )

        ag = _make_client()
        with pytest.raises(RuntimeError):
            async with ag:
                await ag.verify_chunk("chunk-abc")
                raise RuntimeError("simulated failure inside context block")

    assert ag._transport._client.is_closed is True
