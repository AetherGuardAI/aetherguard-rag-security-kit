from __future__ import annotations

from typing import Any

from aetherguard_rag_security import (
    AetherGuardRAG,
    AuthorizationError,
    AuthorizeResult,
    IngestError,
    IngestResult,
    RetrievalDeniedError,
    RetrieveResult,
    VerifyResult,
)

from .config import Settings


class AetherGuardService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def client(self) -> AetherGuardRAG:
        return AetherGuardRAG(
            api_url=self.settings.aetherguard_api_url,
            api_key=self.settings.aetherguard_api_key,
            timeout=self.settings.aetherguard_timeout,
        )

    async def secure_ingest(
        self,
        *,
        chunks: list[str],
        embeddings: list[list[float]],
        document_id: str | None,
        source_type: str,
    ) -> IngestResult:
        async with self.client() as ag:
            return await ag.secure_ingest(
                chunks=chunks,
                embeddings=embeddings,
                region=self.settings.region,
                document_id=document_id,
                source_type=source_type,
                embedding_model_id=self.settings.ollama_embed_model,
                embedding_model_version='ollama-local',
            )

    async def authorize(self, *, email: str) -> AuthorizeResult:
        async with self.client() as ag:
            return await ag.authorize(
                email=email,
                region=self.settings.region,
            )

    async def secure_retrieve(
        self,
        *,
        raw_results: list[dict[str, Any]],
        email: str,
    ) -> RetrieveResult:
        async with self.client() as ag:
            return await ag.secure_retrieve(
                raw_results=raw_results,
                email=email,
                region=self.settings.region,
                max_tokens=self.settings.max_context_tokens,
                trust_threshold=self.settings.trust_threshold,
                max_injection_score=self.settings.max_injection_score,
            )

    async def verify_chunk(self, chunk_id: str) -> VerifyResult:
        async with self.client() as ag:
            return await ag.verify_chunk(chunk_id)


__all__ = [
    'AetherGuardService',
    'AuthorizationError',
    'IngestError',
    'RetrievalDeniedError',
]
