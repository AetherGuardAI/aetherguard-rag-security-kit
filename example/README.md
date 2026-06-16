# AetherGuard SDK + Local ChromaDB + Ollama RAG

This project is intentionally simple and practical. It implements the required SDK flow while storing vectors in **ChromaDB** and answering with **Ollama llama3.1**.

## What this project does

### Ingestion

1. Split input text into chunks.
2. Generate embeddings using Ollama.
3. Call:

```python
await AetherGuardRAG(...).secure_ingest(chunks, embeddings, region, ...)
```

4. Save chunks into ChromaDB only after `secure_ingest()` succeeds.
5. Store each returned `ChunkMetadata.to_dict()` beside the Chroma vector.

### Retrieval

1. Call:

```python
await AetherGuardRAG(...).authorize(email, region)
```

2. Query ChromaDB only if authorization succeeds.
3. Convert Chroma results into raw SDK retrieval format.
4. Call:

```python
await AetherGuardRAG(...).secure_retrieve(raw_results, email, region)
```

5. Send only `RetrieveResult.text` to Ollama.
6. Return Ollama's answer and audit counts.

### Debug/compliance

The API exposes:

```http
GET /verify/{chunk_id}
```

which calls SDK `verify_chunk()`.

---

## Important note about the SDK

The wheel `aetherguard_rag_security-0.1.0-py3-none-any.whl` is a thin async REST client. It does not perform hashing/signing/scanning by itself. It calls an AetherGuard backend API using these endpoints:

- `/api/v1/rag/secure-ingest`
- `/api/v1/rag/authorize`
- `/api/v1/rag/secure-retrieve`
- `/api/v1/rag/verify/{chunk_id}`

Set the backend URL, API key, region, email, Chroma, and Ollama values in `app/config.py`.

If you do not have the backend running, the app will start, but `/ingest`, `/ask`, and `/verify` will fail when the SDK tries to call AetherGuard.

---

## Step-by-step implementation

### Step 1: Check configuration

Review `app/config.py` and update the defaults if your AetherGuard backend, email, ChromaDB, or Ollama settings are different.

### Step 2: Install requirements

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

---

### Step 3: Start dependencies

Start ChromaDB on `http://localhost:8000` and Ollama on `http://localhost:11434`, then pull the required Ollama models:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

Start the API:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 9000
```

---

### Step 4: Check health

```bash
curl http://localhost:9000/health
```

Expected response includes Chroma collection name and vector count.

---

### Step 5: Ingest text

```bash
curl -X POST http://localhost:9000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "company-policy-001",
    "source_type": "local-demo",
    "text": "Acme vacation policy: Employees receive 18 annual leave days per year. Sick leave is 10 days per year. Confidential payroll data must not be shared outside HR."
  }'
```

Expected response:

```json
{
  "document_id": "company-policy-001",
  "total_chunks": 1,
  "blocked_chunks": 0,
  "saved_chunks": 1,
  "evidence_id": "..."
}
```

Behind the scenes, the app calls `secure_ingest()` first, then stores the returned AetherGuard metadata in ChromaDB with keys like:

```text
ag_chunk_id
ag_chunk_hash
ag_embedding_hash
ag_signature
ag_classification
ag_trust_level
ag_injection_score
ag_pii_detected
ag_secrets_detected
ag_toxicity_score
ag_region
ag_signed_at
```

---

### Step 6: Ask a question

```bash
curl -X POST http://localhost:9000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many annual leave days do employees get?"
  }'
```

The app flow is:

1. `authorize()`
2. ChromaDB query
3. `secure_retrieve(raw_results=...)`
4. Ollama receives only sanitized `RetrieveResult.text`

---

### Step 7: Verify a chunk

Use a chunk ID returned in Chroma metadata or AetherGuard logs:

```bash
curl http://localhost:9000/verify/<chunk_id>
```

---
## Project structure

```text
.
|-- app/
|   |-- aetherguard_service.py   # SDK wrapper
|   |-- chroma_store.py          # Local ChromaDB HTTP client
|   |-- config.py                # Environment config
|   |-- main.py                  # FastAPI endpoints
|   |-- ollama_client.py         # Ollama embedding/chat client
|   |-- schemas.py               # Request/response models
|   `-- text_splitter.py         # Simple chunking
|-- wheels/
|   `-- aetherguard_rag_security-0.1.0-py3-none-any.whl
|-- requirements.txt
`-- README.md
```

---

## API endpoints
### `POST /ingest`

Request:

```json
{
  "text": "your document text",
  "document_id": "optional-id",
  "source_type": "local-demo"
}
```

### `POST /ask`

Request:

```json
{
  "question": "your question"
}
```

### `GET /verify/{chunk_id}`

Calls SDK `verify_chunk()`.

---

## Notes for production

- Use a dedicated embedding model such as `nomic-embed-text` or `mxbai-embed-large` instead of `llama3.1` for better vector search quality.
- Keep ChromaDB behind private networking.
- Do not log raw confidential documents.
- Configure the authorization email in the application settings.
- Add authentication to the FastAPI app before exposing it.
- Use a persistent production Chroma deployment or another managed vector database if needed.
