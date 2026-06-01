"""
Unit tests for AetherGuardRAG client.

Implemented in tasks 4.x and 5.x.

**Validates: Requirement 1.3**
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import assume, given, settings
import hypothesis.strategies as st

from aetherguard_rag_security import AetherGuardRAG


# ---------------------------------------------------------------------------
# Property 13: SDK Input Validation — chunks/embeddings length mismatch
# ---------------------------------------------------------------------------
# Validates: Requirement 1.3
# "WHEN the number of chunks does not equal the number of embeddings,
#  THE SDK SHALL reject the request locally before making any network call"
# ---------------------------------------------------------------------------

@given(
    chunks=st.lists(st.text(min_size=1), min_size=0, max_size=10),
    embeddings=st.lists(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=5,
        ),
        min_size=0,
        max_size=10,
    ),
)
@settings(max_examples=50, deadline=None)
async def test_secure_ingest_length_mismatch_raises_value_error_no_http(
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    """
    **Validates: Requirement 1.3**

    Property: For ANY combination where len(chunks) != len(embeddings),
    calling secure_ingest MUST raise ValueError BEFORE any HTTP call is made.
    """
    assume(len(chunks) != len(embeddings))

    with patch(
        "aetherguard_rag_security._http.HTTPTransport.post",
        new_callable=AsyncMock,
    ) as mock_post:
        client = AetherGuardRAG(
            api_url="https://api.aetherguard.example",
            api_key="test-key",
        )

        with pytest.raises(ValueError):
            await client.secure_ingest(
                chunks=chunks,
                embeddings=embeddings,
                tenant_id="test-tenant",
                region="us-east-1",
            )

        # The ValueError must be raised BEFORE any HTTP transport is touched.
        mock_post.assert_not_called()

        await client.close()


import httpx
import respx


# ---------------------------------------------------------------------------
# Property 16: SDK Bearer Token Authentication
# Validates: Requirement 13.3
# ---------------------------------------------------------------------------
# For ANY valid api_key string, every HTTP request made by the SDK must
# include an `Authorization: Bearer {api_key}` header.
# ---------------------------------------------------------------------------

@given(
    api_key=st.text(
        min_size=1,
        max_size=64,
        # Restrict to ASCII alphanumeric only (A-Z, a-z, 0-9) so the key is
        # always a valid ASCII HTTP header value.
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            max_codepoint=127,
        ),
    )
)
@settings(max_examples=50, deadline=None)
@pytest.mark.asyncio
async def test_bearer_token_on_every_request(api_key: str) -> None:
    """
    **Validates: Requirement 13.3**

    Property 16: SDK Bearer Token Authentication.

    For any alphanumeric api_key, the Authorization header on every outgoing
    HTTP request must equal ``Bearer {api_key}``.
    """
    verify_response_body = {
        "chunk_id": "test-chunk",
        "hash_valid": True,
        "signature_valid": True,
        "trust_level": "verified",
        "exists": True,
    }

    with respx.mock(base_url="https://api.test") as mock:
        mock.get("/api/v1/rag/verify/test-chunk").mock(
            return_value=httpx.Response(200, json=verify_response_body)
        )

        async with AetherGuardRAG(api_url="https://api.test", api_key=api_key) as ag:
            await ag.verify_chunk("test-chunk")

        assert len(mock.calls) == 1, "Expected exactly one HTTP call"
        request = mock.calls[0].request
        assert request.headers["authorization"] == f"Bearer {api_key}", (
            f"Expected 'Bearer {api_key}', got '{request.headers.get('authorization')}'"
        )


# ---------------------------------------------------------------------------
# Property: ChunkMetadata.to_dict() key prefixing
# ---------------------------------------------------------------------------
# Validates: Requirement 1.2 / Design: ChunkMetadata.to_dict()
# "All keys are prefixed with ag_ so they are namespaced and easy to
#  identify in any vector database."
# ---------------------------------------------------------------------------

from aetherguard_rag_security.models import ChunkMetadata

chunk_metadata_strategy = st.builds(
    ChunkMetadata,
    chunk_id=st.uuids().map(str),
    chunk_hash=st.text(alphabet="0123456789abcdef", min_size=64, max_size=64),
    embedding_hash=st.text(alphabet="0123456789abcdef", min_size=64, max_size=64),
    signature=st.text(min_size=1, max_size=256),
    classification=st.sampled_from(["public", "internal", "confidential", "regulated"]),
    trust_level=st.sampled_from(["verified", "trusted", "unverified"]),
    injection_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    pii_detected=st.booleans(),
    secrets_detected=st.booleans(),
    toxicity_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    token_count=st.integers(min_value=0, max_value=10000),
    region=st.text(min_size=1, max_size=64),
    signed_at=st.datetimes().map(lambda d: d.isoformat()),
)


@given(metadata=chunk_metadata_strategy)
@settings(max_examples=100)
def test_chunk_metadata_to_dict_all_keys_ag_prefixed(metadata: ChunkMetadata) -> None:
    """
    **Validates: Requirement 1.2**

    Property 1: Every key in ChunkMetadata.to_dict() output MUST start with "ag_".
    Property 2: Every value in ChunkMetadata.to_dict() output MUST be non-None
                for a fully-populated (valid) ChunkMetadata instance.
    """
    result = metadata.to_dict()

    # Property 1 — all keys carry the ag_ namespace prefix
    assert all(k.startswith("ag_") for k in result.keys()), (
        f"Found key(s) without 'ag_' prefix: "
        f"{[k for k in result.keys() if not k.startswith('ag_')]}"
    )

    # Property 2 — no value is None for a valid instance
    assert all(v is not None for v in result.values()), (
        f"Found None value(s) for key(s): "
        f"{[k for k, v in result.items() if v is None]}"
    )


# ---------------------------------------------------------------------------
# Property: Retry Behavior — transient and non-transient status codes
# Validates: Requirements 12.1, 13.5
# ---------------------------------------------------------------------------
# 12.1: IF Backend_API is unreachable, THEN THE SDK SHALL retry with
#        exponential backoff up to the configured max_retries before raising
#        a ConnectionError.
# 13.5: THE SDK SHALL use a configurable max_retries (default 3) for
#        transient failure recovery.
# ---------------------------------------------------------------------------

from aetherguard_rag_security._http import HTTPTransport
from aetherguard_rag_security.exceptions import ConnectionError as AetherConnectionError


@given(
    max_retries=st.integers(min_value=1, max_value=3),
    transient_code=st.sampled_from([502, 503, 504]),
)
@settings(max_examples=30, deadline=None)
@pytest.mark.asyncio
async def test_retries_exactly_max_retries_times_on_transient(
    max_retries: int,
    transient_code: int,
) -> None:
    """
    **Validates: Requirements 12.1, 13.5**

    Property: For any max_retries in [1, 2, 3] and any transient status code
    (502, 503, 504), when ALL requests return that transient status the
    transport MUST make exactly ``max_retries + 1`` total attempts before
    raising ConnectionError.

    The loop in _http.py is ``for attempt in range(self._max_retries + 1)``,
    so with max_retries=N it iterates N+1 times (attempts 0 … N).
    """
    with patch(
        "aetherguard_rag_security._http.asyncio.sleep",
        new=AsyncMock(return_value=None),
    ):
        with respx.mock(base_url="https://api.test") as mock:
            mock.get("/api/v1/rag/verify/chunk-1").mock(
                return_value=httpx.Response(transient_code)
            )
            transport = HTTPTransport(
                "https://api.test",
                "key",
                timeout=0.01,
                max_retries=max_retries,
            )
            with pytest.raises(AetherConnectionError):
                await transport.get("/api/v1/rag/verify/chunk-1")
            await transport.aclose()

            # Assert inside the context manager so mock.calls is still populated
            assert mock.calls.call_count == max_retries + 1, (
                f"Expected {max_retries + 1} attempts for max_retries={max_retries} "
                f"with transient status {transient_code}, "
                f"got {mock.calls.call_count}"
            )


@given(
    non_transient_code=st.sampled_from([400, 403, 404]),
)
@settings(max_examples=30, deadline=None)
@pytest.mark.asyncio
async def test_no_retry_on_non_transient_status(
    non_transient_code: int,
) -> None:
    """
    **Validates: Requirements 12.1, 13.5**

    Property: For any non-transient status code (400, 403, 404) the transport
    MUST return the response immediately after exactly 1 attempt — no retries.
    """
    with respx.mock(base_url="https://api.test") as mock:
        mock.get("/api/v1/rag/verify/chunk-1").mock(
            return_value=httpx.Response(non_transient_code, json={"detail": "error"})
        )
        transport = HTTPTransport(
            "https://api.test",
            "key",
            timeout=0.01,
            max_retries=3,
        )
        response = await transport.get("/api/v1/rag/verify/chunk-1")
        await transport.aclose()

        # Assert inside the context manager so mock.calls is still populated
        assert response.status_code == non_transient_code, (
            f"Expected status {non_transient_code}, got {response.status_code}"
        )
        assert mock.calls.call_count == 1, (
            f"Expected exactly 1 attempt for non-transient status {non_transient_code}, "
            f"got {mock.calls.call_count}"
        )
