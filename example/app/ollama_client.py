from __future__ import annotations

import hashlib
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str, *, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Create embeddings using Ollama. Falls back to deterministic local vectors.

        The fallback keeps the demo runnable if the selected Ollama model does not expose
        embeddings. For real production, use a proper embedding model such as nomic-embed-text.
        """
        if not texts:
            return []

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Newer Ollama API supports batch input at /api/embed.
                response = await client.post(
                    f'{self.base_url}/api/embed',
                    json={'model': model, 'input': texts},
                )
                if response.status_code == 200:
                    data = response.json()
                    vectors = data.get('embeddings')
                    if isinstance(vectors, list) and len(vectors) == len(texts):
                        return vectors

                logger.warning('Ollama /api/embed failed: %s %s', response.status_code, response.text[:300])

                # Older endpoint supports one prompt at a time.
                vectors = []
                for text in texts:
                    r = await client.post(
                        f'{self.base_url}/api/embeddings',
                        json={'model': model, 'prompt': text},
                    )
                    r.raise_for_status()
                    vectors.append(r.json()['embedding'])
                return vectors
        except Exception as exc:  # keep local demo usable
            logger.warning('Ollama embedding failed; using deterministic fallback vectors: %s', exc)
            return [self._fallback_embedding(t) for t in texts]

    async def answer(self, *, question: str, safe_context: str, model: str) -> str:
        prompt = f"""You are a careful assistant. Answer using ONLY the provided secure context.
If the answer is not present in the context, say: "I don't know from the provided documents."

SECURE CONTEXT:
{safe_context}

QUESTION:
{question}
"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f'{self.base_url}/api/chat',
                json={
                    'model': model,
                    'stream': False,
                    'messages': [{'role': 'user', 'content': prompt}],
                },
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return data.get('message', {}).get('content', '').strip()

    def _fallback_embedding(self, text: str, dimensions: int = 384) -> list[float]:
        vector = [0.0] * dimensions
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode('utf-8')).digest()
            idx = int.from_bytes(digest[:4], 'big') % dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]
