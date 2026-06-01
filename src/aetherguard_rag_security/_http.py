"""
Internal HTTP transport for aetherguard-rag-security SDK.

This module is private (prefixed with ``_``) and should not be imported
directly by SDK consumers.  It wraps ``httpx.AsyncClient`` and adds:

- Bearer-token authentication on every request
- Configurable timeout
- Exponential-backoff retry for transient failures (502, 503, 504,
  connection errors)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from .exceptions import ConnectionError  # noqa: A004

logger = logging.getLogger(__name__)

# HTTP status codes that are considered transient and worth retrying.
_TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({502, 503, 504})

# Base delay (seconds) for exponential backoff: 1s, 2s, 4s, …
_BACKOFF_BASE: float = 1.0


class HTTPTransport:
    """
    Async HTTP transport with retry and authentication.

    Parameters
    ----------
    base_url:
        Base URL of backend-api (e.g. ``"https://api.aetherguard.ai"``).
    api_key:
        API key used in the ``Authorization: Bearer`` header.
    timeout:
        Per-request timeout in seconds (default 30).
    max_retries:
        Maximum number of retry attempts for transient failures (default 3).
        A value of 0 means no retries.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Public request helpers
    # ------------------------------------------------------------------

    async def post(self, path: str, json: dict[str, Any]) -> httpx.Response:
        """Send a POST request with JSON body, retrying on transient errors."""
        return await self._request("POST", path, json=json)

    async def get(self, path: str) -> httpx.Response:
        """Send a GET request, retrying on transient errors."""
        return await self._request("GET", path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying httpx client and release connections."""
        await self._client.aclose()

    async def __aenter__(self) -> HTTPTransport:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """
        Execute an HTTP request with exponential-backoff retry.

        Retries are attempted only for transient failures:
        - ``httpx.ConnectError`` / ``httpx.TimeoutException``
        - HTTP 502, 503, 504 responses

        Non-transient HTTP errors (4xx, 500, etc.) are returned immediately
        without retrying so the caller can inspect and raise appropriate
        SDK exceptions.
        """
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(method, path, json=json)

                if response.status_code in _TRANSIENT_STATUS_CODES:
                    if attempt < self._max_retries:
                        delay = _BACKOFF_BASE * (2 ** attempt)
                        logger.warning(
                            "Transient %s from %s %s — retrying in %.1fs (attempt %d/%d)",
                            response.status_code,
                            method,
                            path,
                            delay,
                            attempt + 1,
                            self._max_retries,
                        )
                        await asyncio.sleep(delay)
                        continue
                    # Exhausted retries — return the last bad response so the
                    # caller can raise ConnectionError.
                    raise ConnectionError(
                        f"backend-api returned {response.status_code} after "
                        f"{self._max_retries} retries for {method} {path}",
                        attempts=attempt + 1,
                    )

                return response

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt < self._max_retries:
                    delay = _BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "Connection error on %s %s — retrying in %.1fs (attempt %d/%d): %s",
                        method,
                        path,
                        delay,
                        attempt + 1,
                        self._max_retries,
                        exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise ConnectionError(
                        f"backend-api unreachable after {self._max_retries} retries "
                        f"for {method} {path}: {exc}",
                        attempts=attempt + 1,
                    ) from exc

        # Should be unreachable, but satisfy the type checker.
        raise ConnectionError(  # pragma: no cover
            f"Unexpected retry exhaustion for {method} {path}",
            attempts=self._max_retries + 1,
        )
