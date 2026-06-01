"""
SDK-specific exceptions for aetherguard-rag-security.

All exceptions inherit from AetherGuardError so callers can catch the
base class or specific subclasses as needed.
"""
from __future__ import annotations


class AetherGuardError(Exception):
    """Base exception for all AetherGuard SDK errors."""


class IngestError(AetherGuardError):
    """
    Raised when backend-api returns an error during secure_ingest.

    Attributes:
        status_code: HTTP status code returned by backend-api (if available).
        detail: Error detail string from the response body.
    """

    def __init__(self, message: str, *, status_code: int | None = None, detail: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class RetrievalDeniedError(AetherGuardError):
    """
    Raised when backend-api returns 403 during secure_retrieve.

    Attributes:
        denial_reason: Human-readable reason for the denial.
    """

    def __init__(self, message: str, *, denial_reason: str | None = None) -> None:
        super().__init__(message)
        self.denial_reason = denial_reason


class ConnectionError(AetherGuardError):  # noqa: A001 — intentional shadowing of built-in
    """
    Raised when backend-api is unreachable after all retry attempts.

    Attributes:
        attempts: Number of attempts made before giving up.
    """

    def __init__(self, message: str, *, attempts: int = 0) -> None:
        super().__init__(message)
        self.attempts = attempts


class AuthorizationError(AetherGuardError):
    """
    Raised when an authorization check fails (non-retrieval context).

    Attributes:
        denial_reason: Human-readable reason for the auth failure.
    """

    def __init__(self, message: str, *, denial_reason: str | None = None) -> None:
        super().__init__(message)
        self.denial_reason = denial_reason
