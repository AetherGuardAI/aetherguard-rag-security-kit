from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection


class ChromaStore:
    def __init__(self, host: str, port: int, collection_name: str) -> None:
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={'hnsw:space': 'cosine'},
        )

    def upsert_secure_chunks(
        self,
        *,
        ids: list[str],
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, *, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances'],
        )

        documents = result.get('documents', [[]])[0]
        metadatas = result.get('metadatas', [[]])[0]
        distances = result.get('distances', [[]])[0]
        ids = result.get('ids', [[]])[0]

        raw_results: list[dict[str, Any]] = []
        for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            raw_results.append(
                {
                    # SDK says raw result must contain chunk_id, content, metadata.
                    'chunk_id': metadata.get('ag_chunk_id') or chunk_id,
                    'content': content,
                    'metadata': metadata,
                    # Extra fields are useful for audit/debug and should be tolerated by SDK backend.
                    'score': 1.0 - float(distance),
                    'distance': float(distance),
                }
            )
        return raw_results

    def count(self) -> int:
        return self.collection.count()
