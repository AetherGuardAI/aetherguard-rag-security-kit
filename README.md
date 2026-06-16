<img width="120" height="120" alt="ag_security_logo - github" src="https://github.com/user-attachments/assets/0d5b299d-c9e9-4024-9356-25216f9740ca" /><br/>
# AetherGuard RAG Security Kit

Zero-trust retrieval governance for Retrieval-Augmented Generation (RAG) pipelines.

The SDK is a **thin async REST client** — it does no hashing, signing, scanning, or
database work itself. All security operations are performed server-side by the
AetherGuard backend-api. The SDK works with **any vector database** (Pinecone,
Chroma, pgvector, Weaviate, Qdrant, OpenSearch, …) and **any embedding model**
(OpenAI, Cohere, local models, …).

---

## Getting Started

1. Go to [https://aetherguard.ai](https://aetherguard.ai)
2. Sign up for a free account
3. Generate an API key from the portal dashboard
4. Use the API key in the SDK as shown below

---

## Installation

```bash
pip install aetherguard-rag-security
```

Python 3.10 or later is required.

---

## Quick Start

> **Note:** Replace `https://example.com` with the actual API URL provided in the **Settings** section of the [AetherGuard AI Control Portal](https://genesis.aetherguard.ai).

### 1. Ingestion Flow

Split your document, generate embeddings with your preferred model, then hand
the chunks and embeddings to the SDK. Store the returned metadata alongside
your vectors in your vector database.

```python
import asyncio
from aetherguard_rag_security import AetherGuardRAG

async def ingest_document(text: str) -> None:
    # 1. Split and embed with your own tools
    chunks = my_splitter.split(text)                  # list[str]
    embeddings = my_embedding_model.embed(chunks)     # list[list[float]]

    # 2. Send to AetherGuard for security processing
    async with AetherGuardRAG(
        api_url="https://example.com",
        api_key="YOUR_API_KEY",
    ) as ag:
        result = await ag.secure_ingest(
            chunks=chunks,
            embeddings=embeddings,
            tenant_id="acme-corp",
            region="us-east-1",
            document_id="doc-001",
            source_type="confluence",
        )

    # 3. Store vectors + AetherGuard metadata in your vector DB
    for chunk, embedding, meta in zip(chunks, embeddings, result.metadata):
        vector_db.upsert(
            id=meta.chunk_id,
            vector=embedding,
            metadata=meta.to_dict(),   # all keys prefixed with "ag_"
        )

    print(f"Ingested {result.total_chunks} chunks, blocked {result.blocked_chunks}")
    print(f"Evidence ID: {result.evidence_id}")

asyncio.run(ingest_document("..."))
```

### 2. Retrieval Flow

Query your vector database as usual, then pass the raw results to the SDK.
AetherGuard verifies integrity, scans for threats, sanitises content, and
returns safe context ready for your LLM.

```python
import asyncio
from aetherguard_rag_security import AetherGuardRAG, RetrievalDeniedError

async def retrieve_context(query: str, email: str) -> str:
    # 1. Query your vector DB as normal
    query_embedding = my_embedding_model.embed([query])[0]
    raw_results = vector_db.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True,
    )

    # 2. Send raw results to AetherGuard for verification + sanitisation
    async with AetherGuardRAG(
        api_url="https://example.com",
        api_key="YOUR_API_KEY",
    ) as ag:
        try:
            result = await ag.secure_retrieve(
                raw_results=raw_results,   # list of dicts from your vector DB
                tenant_id="acme-corp",
                email=email,
                region="us-east-1",
                max_tokens=4096,
                trust_threshold="trusted",
            )
        except RetrievalDeniedError as exc:
            print(f"Access denied: {exc.denial_reason}")
            return ""

    # 3. Pass the safe context to your LLM
    print(f"Verified {result.verified_count}/{result.total_retrieved} chunks")
    print(f"Evidence ID: {result.evidence_id}")
    return result.text   # safe, sanitised context string

context = asyncio.run(retrieve_context("What is our refund policy?", "user@acme.com"))
```

### 3. On-Demand Chunk Verification

Verify the integrity of a specific chunk at any time — useful for compliance
checks or debugging.

```python
import asyncio
from aetherguard_rag_security import AetherGuardRAG

async def check_chunk(chunk_id: str) -> None:
    async with AetherGuardRAG(
        api_url="https://example.com",
        api_key="YOUR_API_KEY",
    ) as ag:
        result = await ag.verify_chunk(chunk_id)

    if result.exists and result.hash_valid and result.signature_valid:
        print(f"Chunk {chunk_id} is intact (trust_level={result.trust_level})")
    elif not result.exists:
        print(f"Chunk {chunk_id} not found in AetherGuard")
    else:
        print(f"Chunk {chunk_id} FAILED verification — possible tampering!")

asyncio.run(check_chunk("550e8400-e29b-41d4-a716-446655440000"))
```

### 4. Pre-flight Authorisation Check

Check whether a user is authorised to retrieve data before querying your
vector database.

```python
import asyncio
from aetherguard_rag_security import AetherGuardRAG, AuthorizationError

async def check_access(user_id: str, role: str) -> bool:
    async with AetherGuardRAG(
        api_url="https://example.com",
        api_key="YOUR_API_KEY",
    ) as ag:
        try:
            result = await ag.authorize(
                tenant_id="acme-corp",
                user_id=user_id,
                role=role,
                region="us-east-1",
            )
        except AuthorizationError as exc:
            print(f"Denied: {exc.denial_reason}")
            return False

    print(f"Authorised — namespace={result.namespace}, "
          f"classifications={result.allowed_classifications}")
    return result.authorized

asyncio.run(check_access("user-42", "support"))
```

### 5. Async Context Manager (Recommended Pattern)

The `async with` pattern ensures the underlying HTTP connection pool is
always closed, even if an exception is raised.

```python
import asyncio
from aetherguard_rag_security import AetherGuardRAG

async def main() -> None:
    async with AetherGuardRAG(
        api_url="https://example.com",
        api_key="YOUR_API_KEY",
        timeout=30.0,      # seconds per request
        max_retries=3,     # retries on 502/503/504 with exponential backoff
    ) as ag:
        # All calls share the same connection pool
        ingest_result = await ag.secure_ingest(...)
        retrieve_result = await ag.secure_retrieve(...)
        verify_result = await ag.verify_chunk(...)
    # HTTP client is closed here automatically

asyncio.run(main())
```

---

## Error Handling

```python
from aetherguard_rag_security import (
    AetherGuardError,
    IngestError,
    RetrievalDeniedError,
    ConnectionError,
    AuthorizationError,
)

try:
    result = await ag.secure_ingest(chunks, embeddings, tenant_id="acme", region="us-east-1")
except ValueError as exc:
    # chunks/embeddings length mismatch — caught before any network call
    print(f"Input error: {exc}")
except IngestError as exc:
    print(f"Ingestion failed (HTTP {exc.status_code}): {exc.detail}")
except ConnectionError as exc:
    print(f"backend-api unreachable after {exc.attempts} attempts: {exc}")
except AetherGuardError as exc:
    print(f"Unexpected SDK error: {exc}")
```

---

## Configuration

| Parameter    | Default | Description                                              |
|--------------|---------|----------------------------------------------------------|
| `api_url`    | —       | Base URL of backend-api (required)                       |
| `api_key`    | —       | Bearer token for authentication (required)               |
| `timeout`    | `30.0`  | Per-request timeout in seconds                           |
| `max_retries`| `3`     | Max retries on transient failures (502, 503, 504, conn.) |

Retries use exponential backoff: 1 s → 2 s → 4 s → …

---

## Requirements

- Python >= 3.10
- `httpx >= 0.25.0`

---

## License

Proprietary — AetherGuard AI. All rights reserved.
