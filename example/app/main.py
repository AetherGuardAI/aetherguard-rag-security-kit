from __future__ import annotations

import logging
import sys
import uuid
import time
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pypdf import PdfReader
from io import BytesIO

from .aetherguard_service import (
    AetherGuardService,
    AuthorizationError,
    IngestError,
    RetrievalDeniedError,
)
from .chroma_store import ChromaStore
from .config import get_settings
from .ollama_client import OllamaClient
from .schemas import AskRequest, AskResponse, HealthResponse, IngestRequest, IngestResponse, VerifyResponse
from .text_splitter import split_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def log_info(*values: object, sep: str = ' ', end: str = '\n', flush: bool = False) -> None:
    """Route existing ingest diagnostics through the configured application logger."""
    del flush
    message = sep.join(str(value) for value in values)
    if end != '\n':
        message += end
    logger.info(message.rstrip('\n'))

settings = get_settings()
app = FastAPI(title=settings.app_name, version='1.0.0')

store = ChromaStore(settings.chroma_host, settings.chroma_port, settings.chroma_collection)
ollama = OllamaClient(settings.ollama_base_url)
ag_service = AetherGuardService(settings)


@app.get('/health', response_model=HealthResponse)
def health() -> HealthResponse:
    logger.info('Health check endpoint called')
    chroma_count = store.count()
    logger.debug(f'Chroma store count: {chroma_count}')
    response = HealthResponse(
        status='ok',
        details={
            'chroma_collection': settings.chroma_collection,
            'chroma_count': chroma_count,
            'ollama_chat_model': settings.ollama_chat_model,
            'ollama_embed_model': settings.ollama_embed_model,
            'region': settings.region,
        },
    )
    logger.info('Health check completed successfully')
    return response


@app.post('/ingest', response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    chunks = split_text(request.text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail='No text chunks produced')

    document_id = request.document_id or str(uuid.uuid4())
    source_type = request.source_type or settings.source_type

    embeddings = await ollama.embed(chunks, settings.ollama_embed_model)
    # REQUIRED FLOW 1: call SDK secure_ingest BEFORE saving to ChromaDB.
    try:
        result = await ag_service.secure_ingest(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            source_type=source_type,
        )
        log_info("Metadata:", result.metadata)
    except IngestError as exc:
        raise HTTPException(status_code=502, detail=f'AetherGuard secure_ingest failed: {exc.detail or exc}') from exc
    except Exception as exc:
        logger.exception('Unexpected ingest failure')
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    ids: list[str] = []
    saved_chunks: list[str] = []
    saved_embeddings: list[list[float]] = []
    saved_metadatas: list[dict] = []

    # Store only chunks that AetherGuard returned metadata for.
    # If backend blocks a chunk, it should not return metadata for that chunk.
    for idx, meta in enumerate(result.metadata):
        metadata = meta.to_dict()
        metadata.update(
            {
                'document_id': result.document_id,
                'source_type': source_type,
                'region': settings.region,
                'chunk_position': idx,
            }
        )
        ids.append(meta.chunk_id)
        saved_chunks.append(chunks[idx])
        saved_embeddings.append(embeddings[idx])
        saved_metadatas.append(metadata)

    # REQUIRED FLOW 2: save returned AetherGuard metadata alongside Chroma vectors.
    store.upsert_secure_chunks(ids=ids, chunks=saved_chunks, embeddings=saved_embeddings, metadatas=saved_metadatas)

    return IngestResponse(
        document_id=result.document_id,
        total_chunks=result.total_chunks,
        blocked_chunks=result.blocked_chunks,
        saved_chunks=len(ids),
        evidence_id=result.evidence_id,
    )


@app.post('/ask', response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    total_start = time.perf_counter()

    logger.info("========== ASK REQUEST START ==========")

    auth_start = time.perf_counter()
    try:
        authorization = await ag_service.authorize(
            email=settings.email,
        )
    except AuthorizationError as exc:
        raise HTTPException(
            status_code=403,
            detail=exc.denial_reason or "Authorization denied",
        ) from exc

    if not authorization.authorized:
        raise HTTPException(
            status_code=403,
            detail=authorization.denial_reason or "Authorization denied",
        )

    auth_time = time.perf_counter() - auth_start
    logger.info(f"AUTHORIZATION: {auth_time:.3f}s")

    # EMBEDDING
    embed_start = time.perf_counter()

    query_embedding = (
        await ollama.embed(
            [request.question],
            settings.ollama_embed_model
        )
    )[0]

    embed_time = time.perf_counter() - embed_start
    logger.info(f"EMBEDDING: {embed_time:.3f}s")

    # CHROMADB QUERY
    chroma_start = time.perf_counter()

    raw_results = store.query(
        query_embedding=query_embedding,
        top_k=settings.top_k
    )

    chroma_time = time.perf_counter() - chroma_start
    logger.info(f"CHROMADB QUERY: {chroma_time:.3f}s")
    logger.info(f"Raw ChromaDB results: {raw_results}")

    # SECURE RETRIEVE
    secure_start = time.perf_counter()
    
    try:
        secure = await ag_service.secure_retrieve(
            raw_results=raw_results,
            email=settings.email,
        )

    except RetrievalDeniedError as exc:
        raise HTTPException(
            status_code=403,
            detail=exc.denial_reason or "Secure retrieval denied"
        ) from exc

    secure_time = time.perf_counter() - secure_start

    logger.info(f"SECURE RETRIEVE: {secure_time:.3f}s")
    logger.info(f"Secure retrieve result: {secure}")

    if not secure.text.strip():
        total_time = time.perf_counter() - total_start

        logger.info(f"TOTAL REQUEST TIME: {total_time:.3f}s")
        logger.info("========== ASK REQUEST END ==========")

        return AskResponse(
            answer="I don't know from the provided documents.",
            evidence_id=secure.evidence_id
        )

    # LLM GENERATION
    llm_start = time.perf_counter()

    answer = await ollama.answer(
        question=request.question,
        safe_context=secure.text,
        model=settings.ollama_chat_model,
    )

    llm_time = time.perf_counter() - llm_start
    logger.info(f"OLLAMA GENERATION: {llm_time:.3f}s")

    total_time = time.perf_counter() - total_start

    logger.info(
        "\n===== LATENCY BREAKDOWN =====\n"
        f"Authorization   : {auth_time:.3f}s\n"
        f"Embedding       : {embed_time:.3f}s\n"
        f"ChromaDB Query  : {chroma_time:.3f}s\n"
        f"Secure Retrieve : {secure_time:.3f}s\n"
        f"LLM Generation  : {llm_time:.3f}s\n"
        f"TOTAL           : {total_time:.3f}s\n"
        "============================="
    )

    return AskResponse(
        answer=answer,
        evidence_id=secure.evidence_id,
        verified_count=secure.verified_count,
        blocked_count=secure.blocked_count,
        sanitized_count=secure.sanitized_count,
        raw_retrieved=secure.total_retrieved,
    )

@app.get('/verify/{chunk_id}', response_model=VerifyResponse)
async def verify_chunk(chunk_id: str) -> VerifyResponse:
    # Optional debugging/compliance endpoint.
    result = await ag_service.verify_chunk(chunk_id)
    return VerifyResponse(
        chunk_id=result.chunk_id,
        exists=result.exists,
        hash_valid=result.hash_valid,
        signature_valid=result.signature_valid,
        trust_level=result.trust_level,
    )




@app.post("/ingest-file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    document_id: str | None = Form(None),
    source_type: str | None = Form(None),
) -> IngestResponse:
    log_info("\n========== FILE INGEST START ==========")
    log_info("Filename:", file.filename)
    log_info("Content Type:", file.content_type)

    content = await file.read()
    log_info("File Size:", len(content), "bytes")

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        if file.filename and file.filename.lower().endswith(".pdf"):
            log_info("Detected PDF file. Extracting text...")

            reader = PdfReader(BytesIO(content))
            text = ""

            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                log_info(f"Page {page_number} text length:", len(page_text))
                text += page_text + "\n"

            source_type = source_type or "pdf"

        else:
            log_info("Detected text file. Decoding as UTF-8...")
            text = content.decode("utf-8")
            source_type = source_type or file.content_type or settings.source_type

    except UnicodeDecodeError:
        log_info("ERROR: File is not valid UTF-8 text")
        raise HTTPException(
            status_code=400,
            detail="Only UTF-8 text files or PDF files are supported"
        )
    except Exception as exc:
        log_info("ERROR: Failed to read uploaded file:", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {exc}"
        ) from exc

    log_info("Extracted Text Length:", len(text))
    log_info("Text Preview:", text[:300])

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text found in uploaded file"
        )

    chunks = split_text(
        text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    log_info("Chunks Created:", len(chunks))

    if chunks:
        log_info("First Chunk Preview:", chunks[0][:300])

    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks produced")

    document_id = document_id or str(uuid.uuid4())

    log_info("Document ID:", document_id)
    log_info("Source Type:", source_type)

    log_info("Generating embeddings...")
    embeddings = await ollama.embed(
        chunks,
        settings.ollama_embed_model,
    )

    log_info("Embeddings Generated:", len(embeddings))

    if embeddings:
        log_info("Embedding Dimension:", len(embeddings[0]))

    log_info("Calling AetherGuard secure_ingest...")

    try:
        result = await ag_service.secure_ingest(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            source_type=source_type,
        )

        log_info("secure_ingest SUCCESS")
        log_info("Returned Document ID:", result.document_id)
        log_info("Evidence ID:", result.evidence_id)
        log_info("Total Chunks:", result.total_chunks)
        log_info("Blocked Chunks:", result.blocked_chunks)
        log_info("Metadata Count:", len(result.metadata))
        log_info("Metadata:", result.metadata)

    except IngestError as exc:
        log_info("secure_ingest FAILED:", exc)
        raise HTTPException(
            status_code=502,
            detail=f"AetherGuard secure_ingest failed: {exc.detail or exc}",
        ) from exc
    except Exception as exc:
        log_info("Unexpected ingest failure:", exc)
        logger.exception("Unexpected ingest failure")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    ids: list[str] = []
    saved_chunks: list[str] = []
    saved_embeddings: list[list[float]] = []
    saved_metadatas: list[dict] = []

    log_info("Preparing chunks for ChromaDB...")

    for idx, meta in enumerate(result.metadata):
        metadata = meta.to_dict()

        metadata.update(
            {
                "document_id": result.document_id,
                "source_type": source_type,
                "filename": file.filename,
                "region": settings.region,
                "chunk_position": idx,
            }
        )

        ids.append(meta.chunk_id)
        saved_chunks.append(chunks[idx])
        saved_embeddings.append(embeddings[idx])
        saved_metadatas.append(metadata)

        log_info(
            f"Chunk {idx}: "
            f"chunk_id={meta.chunk_id}, "
            f"text_length={len(chunks[idx])}"
        )

    log_info("Saving chunks to ChromaDB:", len(ids))

    store.upsert_secure_chunks(
        ids=ids,
        chunks=saved_chunks,
        embeddings=saved_embeddings,
        metadatas=saved_metadatas,
    )

    log_info("ChromaDB save complete")

    response = IngestResponse(
        document_id=result.document_id,
        total_chunks=result.total_chunks,
        blocked_chunks=result.blocked_chunks,
        saved_chunks=len(ids),
        evidence_id=result.evidence_id,
    )

    log_info("Response:", response.model_dump())
    log_info("========== FILE INGEST END ==========\n")

    return response

@app.post("/ingest-file-direct", response_model=IngestResponse)
async def ingest_file_direct(
    file: UploadFile = File(...),
    document_id: str | None = Form(None),
    source_type: str | None = Form(None),
) -> IngestResponse:
    log_info("\n========== DIRECT FILE INGEST START ==========")
    log_info("Filename:", file.filename)
    log_info("Content Type:", file.content_type)

    content = await file.read()
    log_info("File Size:", len(content), "bytes")

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        if file.filename and file.filename.lower().endswith(".pdf"):
            log_info("Detected PDF file. Extracting text...")

            reader = PdfReader(BytesIO(content))
            text = ""

            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                log_info(f"Page {page_number} text length:", len(page_text))
                text += page_text + "\n"

            source_type = source_type or "pdf"

        else:
            log_info("Detected text file. Decoding as UTF-8...")
            text = content.decode("utf-8")
            source_type = source_type or file.content_type or settings.source_type

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Only UTF-8 text files or PDF files are supported",
        )
    except Exception as exc:
        log_info("ERROR: Failed to read uploaded file:", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {exc}",
        ) from exc

    log_info("Extracted Text Length:", len(text))
    log_info("Text Preview:", text[:300])

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text found in uploaded file",
        )

    chunks = split_text(
        text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    log_info("Chunks Created:", len(chunks))

    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks produced")

    document_id = document_id or str(uuid.uuid4())

    log_info("Document ID:", document_id)
    log_info("Source Type:", source_type)

    log_info("Generating embeddings...")
    embeddings = await ollama.embed(
        chunks,
        settings.ollama_embed_model,
    )

    log_info("Embeddings Generated:", len(embeddings))

    if embeddings:
        log_info("Embedding Dimension:", len(embeddings[0]))

    ids: list[str] = []
    saved_metadatas: list[dict] = []

    log_info("Preparing direct ChromaDB records...")

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{document_id}-chunk-{idx}"

        metadata = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "source_type": source_type,
            "filename": file.filename,
            "region": settings.region,
            "chunk_position": idx,
            "ingest_mode": "direct_without_aetherguard",
        }

        ids.append(chunk_id)
        saved_metadatas.append(metadata)

        log_info(
            f"Chunk {idx}: "
            f"chunk_id={chunk_id}, "
            f"text_length={len(chunk)}"
        )

    log_info("Saving chunks directly to ChromaDB:", len(ids))

    store.upsert_secure_chunks(
        ids=ids,
        chunks=chunks,
        embeddings=embeddings,
        metadatas=saved_metadatas,
    )

    log_info("Direct ChromaDB save complete")

    response = IngestResponse(
        document_id=document_id,
        total_chunks=len(chunks),
        blocked_chunks=0,
        saved_chunks=len(ids),
        evidence_id="direct-ingest-no-aetherguard",
    )

    log_info("Response:", response.model_dump())
    log_info("========== DIRECT FILE INGEST END ==========\n")

    return response
