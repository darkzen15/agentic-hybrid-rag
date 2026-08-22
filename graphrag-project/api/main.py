import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

import vector_store
import graph_store
import jobs
import file_parsers
import llm
import ollama_client
from config import settings
from ingest import ingest_document, process_uploaded_file
from retrieval import hybrid_retrieve, format_as_context
from agentic_retrieval import agentic_retrieve

logger = logging.getLogger("graphrag")
logging.basicConfig(level=logging.INFO)
# The neo4j driver logs query-planner advisories (e.g. "cartesian product"
# notices) at INFO level. These are informational, not errors — harmless
# here since Entity.name has a uniqueness constraint, so the flagged
# queries only ever match a single node per side. Quieted so they don't
# bury actual errors in the logs.
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

app = FastAPI(
    title="Hybrid GraphRAG API",
    description=(
        "Retrieval API combining vector search (Qdrant), entity-name lookup, "
        "knowledge-graph expansion, and keyword search (Neo4j), fused via "
        "Reciprocal Rank Fusion. Use /retrieve to fetch context for a query."
    ),
    version="1.0.0",
)

# Allow OpenWebUI (or any browser-based caller) to reach this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


STATIC_DIR = Path(__file__).parent / "static"


@app.on_event("startup")
def startup():
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    try:
        graph_store.ensure_constraints()
    except Exception as e:
        print(
            f"[startup] Could not reach Neo4j yet ({e}). The API will still "
            f"start, but ingestion/retrieval will fail until Neo4j is "
            f"reachable — check `docker compose logs neo4j` and "
            f"`docker compose ps` if this persists."
        )

    try:
        vector_store.ensure_collection()
    except Exception as e:
        print(
            f"[startup] Could not initialize the Qdrant collection yet ({e}). "
            f"This is expected if '{settings.EMBEDDING_MODEL}' hasn't been "
            f"pulled into Ollama yet — it'll be created automatically on the "
            f"first successful ingest/retrieve once the model is available."
        )


@app.on_event("shutdown")
async def shutdown():
    graph_store.close_driver()
    ollama_client.close_sync_client()
    await ollama_client.close_async_client()


@app.get("/health", operation_id="health_check", summary="Health check")
def health():
    return {"status": "ok"}


@app.get(
    "/health/detailed",
    operation_id="health_check_detailed",
    summary="Detailed health check: verifies Qdrant, Neo4j, and Ollama connectivity + model availability",
    description=(
        "Checks each dependency directly rather than just returning ok, and "
        "specifically checks whether OLLAMA_MODEL / EMBEDDING_MODEL / "
        "GRAPH_EXTRACTION_MODEL are pulled with the *exact* tag configured "
        "(e.g. 'llama3.1:8b', not just 'llama3.1') - an untagged mismatch "
        "otherwise surfaces later as a confusing 404 from Ollama on the "
        "first request that actually needs the model, rather than here."
    ),
)
async def health_detailed():
    result: Dict[str, Any] = {"status": "ok", "checks": {}}

    try:
        vector_store.get_client().get_collections()
        result["checks"]["qdrant"] = {"ok": True}
    except Exception as e:
        result["checks"]["qdrant"] = {"ok": False, "error": str(e)}
        result["status"] = "degraded"

    try:
        driver = graph_store.get_driver()
        with driver.session() as session:
            session.run("RETURN 1").consume()
        result["checks"]["neo4j"] = {"ok": True}
    except Exception as e:
        result["checks"]["neo4j"] = {"ok": False, "error": str(e)}
        result["status"] = "degraded"

    try:
        pulled_models = await llm.list_ollama_models()
        ollama_check: Dict[str, Any] = {"ok": True, "pulled_models": pulled_models}

        expected_models = {
            "OLLAMA_MODEL": settings.OLLAMA_MODEL,
            "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
        }
        if settings.GRAPH_EXTRACTION_MODEL:
            expected_models["GRAPH_EXTRACTION_MODEL"] = settings.GRAPH_EXTRACTION_MODEL
        # Only check RAG_ASSIST_MODEL if something actually uses it - if
        # none of the LLM-assist retrieval steps are enabled, it's dead
        # config and checking it would just be a false-positive nag.
        assist_features_enabled = (
            settings.RAG_CONDENSE_ENABLED
            or settings.RAG_RERANK_ENABLED
            or settings.RAG_QUERY_EXPANSION_ENABLED
        )
        if settings.RAG_ASSIST_MODEL and assist_features_enabled:
            expected_models["RAG_ASSIST_MODEL"] = settings.RAG_ASSIST_MODEL

        missing = {name: model for name, model in expected_models.items() if model not in pulled_models}
        if missing:
            ollama_check["ok"] = False
            ollama_check["missing_models"] = missing
            ollama_check["hint"] = (
                "A configured model isn't pulled under this exact tag. Run "
                "`docker compose exec ollama ollama list` and compare against "
                "the values above - Ollama treats an untagged name as ':latest', "
                "a different tag than e.g. ':8b' unless you specifically pulled "
                "':latest'."
            )
            result["status"] = "degraded"
        result["checks"]["ollama"] = ollama_check
    except Exception as e:
        result["checks"]["ollama"] = {"ok": False, "error": str(e)}
        result["status"] = "degraded"

    return JSONResponse(status_code=200 if result["status"] == "ok" else 503, content=result)


@app.get("/", include_in_schema=False)
def upload_page():
    """Serves the upload webpage."""
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    doc_id: str = Field(..., description="Unique identifier for this document")
    title: Optional[str] = Field("", description="Human-readable title")
    text: str = Field(..., description="Raw text content to ingest")


class IngestResponse(BaseModel):
    doc_id: str
    chunks_ingested: int
    duplicate_of: Optional[str] = None


@app.post(
    "/ingest",
    response_model=IngestResponse,
    operation_id="ingest_text",
    summary="Ingest raw text into the vector store and knowledge graph",
)
def ingest_text(req: IngestRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    result = ingest_document(doc_id=req.doc_id, title=req.title or "", text=req.text)
    return result


class JobResponse(BaseModel):
    job_id: str
    filename: str
    doc_id: str
    status: str
    chunks_ingested: Optional[int] = None
    total_chunks: Optional[int] = None
    chunks_processed: Optional[int] = None
    duplicate_of: Optional[str] = None
    cancel_requested: Optional[bool] = None
    error: Optional[str] = None
    created_at: float
    updated_at: float


@app.post(
    "/ingest/file",
    response_model=JobResponse,
    operation_id="ingest_file",
    summary="Upload a file to ingest",
    description=(
        "Streams the upload to disk (safe for large files), then ingests it "
        "in the background so the request returns immediately. Poll "
        "/jobs/{job_id} for status. Supported extensions: "
        + ", ".join(sorted(file_parsers.SUPPORTED_EXTENSIONS))
    ),
)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    title: Optional[str] = Form(""),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in file_parsers.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: "
            f"{', '.join(sorted(file_parsers.SUPPORTED_EXTENSIONS))}",
        )

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    saved_path = os.path.join(settings.UPLOAD_DIR, f"{uuid.uuid4()}{ext}")

    # Stream to disk in chunks so large files never sit fully in memory.
    bytes_written = 0
    chunk_size = 1024 * 1024
    try:
        with open(saved_path, "wb") as out_file:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds MAX_UPLOAD_MB ({settings.MAX_UPLOAD_MB} MB)",
                    )
                out_file.write(chunk)
    except HTTPException:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise
    except Exception:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    final_doc_id = doc_id or Path(file.filename).stem
    final_title = title or file.filename

    job_id = jobs.create_job(filename=file.filename, doc_id=final_doc_id)
    background_tasks.add_task(
        process_uploaded_file, job_id, saved_path, file.filename, final_doc_id, final_title
    )

    return jobs.get_job(job_id)


@app.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    operation_id="get_ingest_job",
    summary="Check the status of a file ingestion job",
)
def get_job(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get(
    "/jobs",
    response_model=List[JobResponse],
    operation_id="list_ingest_jobs",
    summary="List recent file ingestion jobs",
)
def list_jobs(limit: int = 50):
    return jobs.list_jobs(limit=limit)


@app.post(
    "/jobs/{job_id}/cancel",
    response_model=JobResponse,
    operation_id="cancel_ingest_job",
    summary="Stop a running (queued/processing) ingestion job",
    description=(
        "Cancellation is cooperative, not immediate: the batch of chunks "
        "already in flight finishes normally (an Ollama call already "
        "underway can't be aborted mid-request), then ingestion stops "
        "before starting the next batch. Chunks already written before "
        "that point remain in Qdrant/Neo4j - this does not roll back "
        "partial ingestion, it just stops adding more."
    ),
)
def cancel_job(job_id: str):
    job = jobs.request_cancel(job_id)
    if job:
        return job
    existing = jobs.get_job(job_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Job not found")
    raise HTTPException(
        status_code=409,
        detail=f"Job is already '{existing['status']}' - nothing to cancel",
    )


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------

class DocumentInfo(BaseModel):
    doc_id: str
    title: Optional[str] = None


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    chunks_deleted: int
    entities_deleted: int
    relationships_deleted: int


@app.get(
    "/documents",
    response_model=List[DocumentInfo],
    operation_id="list_documents",
    summary="List all currently ingested documents",
)
def list_documents():
    return graph_store.list_documents()


@app.delete(
    "/documents/{doc_id}",
    response_model=DeleteDocumentResponse,
    operation_id="delete_document",
    summary="Delete a document and all its data from Qdrant and Neo4j",
    description=(
        "Removes every chunk belonging to this document from Qdrant, then "
        "cleans up the corresponding graph data in Neo4j: entities that "
        "only ever appeared in this document are deleted, relationships "
        "whose evidence chunks were all in this document are deleted, and "
        "entities/relationships shared with other documents are kept "
        "(with this document's contribution stripped out)."
    ),
)
def delete_document(doc_id: str):
    chunk_ids = vector_store.delete_by_doc_id(doc_id)
    graph_result = graph_store.delete_document(doc_id, chunk_ids)
    return {
        "doc_id": doc_id,
        "chunks_deleted": len(chunk_ids),
        "entities_deleted": graph_result["entities_deleted"],
        "relationships_deleted": graph_result["relationships_deleted"],
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    query: str = Field(..., description="User's natural-language query")
    top_k: int = Field(5, ge=1, le=20, description="Number of vector search results")
    graph_expand: int = Field(
        5, ge=0, le=20, description="Number of additional chunks to pull in via graph expansion"
    )
    doc_id: Optional[str] = Field(None, description="Optional: restrict search to one document")
    min_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Minimum vector similarity (0-1) to keep a match; overrides RAG_MIN_SCORE for this request",
    )
    history: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Optional prior turns ([{role, content}, ...]) used for query condensation "
        "if RAG_CONDENSE_ENABLED is set; ignored otherwise",
    )


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    source: str  # "vector", "entity_name", "graph", "keyword", "path", or a "+"-joined combination


class RetrieveResponse(BaseModel):
    query: str
    context: str
    results: List[RetrievedChunk]
    related_entities: List[dict]
    timings: Optional[Dict[str, float]] = None


@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    operation_id="retrieve_context",
    summary="Retrieve relevant context for a query using hybrid vector + graph search",
    description=(
        "Runs multiple retrieval channels (vector similarity, entity-name "
        "lookup, graph expansion, keyword search) and fuses them with "
        "Reciprocal Rank Fusion. Also adds a direct relationship-path "
        "lookup when the query names two or more known entities. Returns "
        "both a ready-to-use context string and structured results."
    ),
)
def retrieve_context(req: RetrieveRequest):
    result = hybrid_retrieve(
        query=req.query,
        top_k=req.top_k,
        graph_expand=req.graph_expand,
        doc_id=req.doc_id,
        min_score=req.min_score,
        history=req.history,
    )
    context = format_as_context(result)
    return {
        "query": req.query,
        "context": context,
        "results": result["results"],
        "related_entities": result["related_entities"],
        "timings": result.get("timings"),
    }


# Simple GET variant — convenient for quick testing / simple tool callers
@app.get(
    "/retrieve",
    response_model=RetrieveResponse,
    operation_id="retrieve_context_get",
    summary="GET variant of /retrieve for quick testing",
)
def retrieve_context_get(query: str, top_k: int = 5, graph_expand: int = 5, min_score: Optional[float] = None):
    return retrieve_context(
        RetrieveRequest(query=query, top_k=top_k, graph_expand=graph_expand, min_score=min_score)
    )


# ---------------------------------------------------------------------------
# RAG chat — retrieval + generation via a local Ollama model
# ---------------------------------------------------------------------------

class RagChatRequest(BaseModel):
    query: str = Field(..., description="User's question")
    top_k: int = Field(5, ge=1, le=20)
    graph_expand: int = Field(5, ge=0, le=20)
    doc_id: Optional[str] = Field(None, description="Optional: restrict retrieval to one document")
    min_score: Optional[float] = Field(None, ge=0, le=1, description="Overrides RAG_MIN_SCORE for this request")
    model: Optional[str] = Field(None, description="Ollama model name; defaults to OLLAMA_MODEL")
    temperature: Optional[float] = Field(None, ge=0, le=2)
    top_p: Optional[float] = Field(None, ge=0, le=1)
    max_tokens: Optional[int] = Field(None, ge=1, description="Maps to Ollama's num_predict")
    history: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Optional prior turns ([{role, content}, ...]) - used both for query "
        "condensation (if RAG_CONDENSE_ENABLED) and as conversational context for generation",
    )


class RagChatResponse(BaseModel):
    answer: str
    context: str
    sources: List[RetrievedChunk]
    related_entities: List[dict]
    timings: Optional[Dict[str, float]] = None


def _smart_retrieve(query, top_k, graph_expand, doc_id, min_score, history):
    """Picks agentic or single-shot retrieval based on the toggle."""
    if settings.RAG_AGENTIC_ENABLED:
        return agentic_retrieve(
            query=query, top_k=top_k, graph_expand=graph_expand,
            doc_id=doc_id, min_score=min_score, history=history,
        )
    return hybrid_retrieve(
        query=query, top_k=top_k, graph_expand=graph_expand,
        doc_id=doc_id, min_score=min_score, history=history,
    )


@app.post(
    "/rag/chat",
    response_model=RagChatResponse,
    operation_id="rag_chat",
    summary="Ask a question — retrieves hybrid context and generates an answer with a local Ollama model",
)
async def rag_chat(req: RagChatRequest):
    retrieval = _smart_retrieve(
        query=req.query, top_k=req.top_k, graph_expand=req.graph_expand,
        doc_id=req.doc_id, min_score=req.min_score, history=req.history,
    )
    context = format_as_context(retrieval)
    messages = llm.build_rag_messages(query=req.query, context=context, history=req.history)
    options = llm.build_options(
        temperature=req.temperature, top_p=req.top_p, max_tokens=req.max_tokens
    )

    try:
        answer = await llm.ollama_chat(messages, model=req.model, options=options)
    except Exception as e:
        logger.exception(f"/rag/chat: Ollama request failed (model={req.model or settings.OLLAMA_MODEL})")
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {e}")

    return {
        "answer": answer,
        "context": context,
        "sources": retrieval["results"],
        "related_entities": retrieval["related_entities"],
        "timings": retrieval.get("timings"),
    }


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints — lets OpenWebUI (or any OpenAI-client) use
# this whole pipeline as a regular selectable chat model. The model dropdown
# is populated from whatever's pulled in the connected Ollama instance;
# every request gets hybrid retrieval injected before hitting Ollama.
# ---------------------------------------------------------------------------

class OAChatMessage(BaseModel):
    role: str
    content: str
    model_config = ConfigDict(extra="allow")


class OAChatCompletionRequest(BaseModel):
    model: str
    messages: List[OAChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    # Optional extensions a caller can set to tune retrieval per-request
    top_k: Optional[int] = None
    graph_expand: Optional[int] = None
    doc_id: Optional[str] = None
    min_score: Optional[float] = None  # overrides RAG_MIN_SCORE for this request
    include_sources: Optional[bool] = None  # overrides SHOW_SOURCES_IN_CHAT for this request
    model_config = ConfigDict(extra="allow")


def _last_user_message(messages: List[OAChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return messages[-1].content if messages else ""


_SOURCE_ICONS = {
    "vector": "🔎",
    "entity_name": "🎯",
    "graph": "🕸️",
    "keyword": "🔤",
    "path": "🧭",
}


def _icon_for_source(source: str) -> str:
    """Retrieval now runs multiple channels and a chunk can be found by
    more than one (e.g. "vector+graph"), so `source` may be a "+"-joined
    combination rather than a single channel name - shows one icon per
    contributing channel, deduped, in a stable order."""
    parts = source.split("+")
    icons = []
    for p in parts:
        icon = _SOURCE_ICONS.get(p, "🔎")
        if icon not in icons:
            icons.append(icon)
    return "".join(icons)


def _build_sources_footer(results: List[Dict[str, Any]]) -> str:
    """A short, human-readable list of what was actually retrieved, so you
    can see in the chat itself whether the answer is grounded in your data —
    not just take the model's word for it. Includes a text snippet per
    chunk (not just the doc_id) since multiple chunks from the same
    document are otherwise indistinguishable in the footer."""
    if not results:
        return "\n\n---\n_No matching context was found in the knowledge base for this question._"

    lines = []
    for i, r in enumerate(results, 1):
        icon = _icon_for_source(r.get("source", "vector"))
        snippet = (r.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > 100:
            snippet = snippet[:100] + "…"
        lines.append(f"{i}. {icon} `{r['doc_id']}` — {snippet}")
    return "\n\n---\n**Sources retrieved:**\n" + "\n".join(lines)


@app.get(
    "/v1/models",
    operation_id="list_models",
    summary="OpenAI-compatible model list (proxies models pulled in Ollama)",
    include_in_schema=False,
)
async def list_models():
    try:
        names = await llm.list_ollama_models()
    except Exception:
        names = [settings.OLLAMA_MODEL]
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "owned_by": "ollama", "created": int(time.time())}
            for name in names
        ],
    }


@app.post(
    "/v1/chat/completions",
    operation_id="openai_chat_completions",
    summary="OpenAI-compatible chat endpoint, retrieval-augmented via the hybrid graph RAG",
    include_in_schema=False,
)
async def chat_completions(req: OAChatCompletionRequest):
    query = _last_user_message(req.messages)
    top_k = req.top_k or settings.RAG_TOP_K
    graph_expand = (
        req.graph_expand if req.graph_expand is not None else settings.RAG_GRAPH_EXPAND
    )

    # Prior turns minus the trailing user message (build_rag_messages appends
    # the current query itself) and minus any client-sent system message
    # (we supply our own, with the retrieved context baked in). Built before
    # the retrieval call since hybrid_retrieve uses it for query condensation
    # when RAG_CONDENSE_ENABLED is set.
    history = [{"role": m.role, "content": m.content} for m in req.messages if m.role != "system"]
    if history and history[-1]["role"] == "user":
        history = history[:-1]

    try:
        retrieval = _smart_retrieve(
            query=query, top_k=top_k, graph_expand=graph_expand,
            doc_id=req.doc_id, min_score=req.min_score, history=history,
        )
    except Exception as e:
        logger.exception("/v1/chat/completions: retrieval failed")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"Retrieval failed (embedding model may not be loaded in Ollama yet): {e}",
                    "type": "upstream_error",
                    "param": None,
                    "code": "retrieval_error",
                }
            },
        )
    context = format_as_context(retrieval)

    rag_messages = llm.build_rag_messages(query=query, context=context, history=history)
    options = llm.build_options(
        temperature=req.temperature, top_p=req.top_p, max_tokens=req.max_tokens, seed=req.seed
    )

    show_sources = req.include_sources if req.include_sources is not None else settings.SHOW_SOURCES_IN_CHAT
    sources_footer = _build_sources_footer(retrieval["results"]) if show_sources else ""

    completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())

    if req.stream:
        async def event_stream() -> AsyncGenerator[str, None]:
            try:
                async for delta in llm.ollama_chat_stream(rag_messages, model=req.model, options=options):
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [
                            {"index": 0, "delta": {"content": delta}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                if sources_footer:
                    footer_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [
                            {"index": 0, "delta": {"content": sources_footer}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(footer_chunk)}\n\n"
            except Exception as e:
                logger.exception(f"/v1/chat/completions (stream): Ollama request failed (model={req.model})")
                error_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [
                        {"index": 0, "delta": {"content": f"\n\n[error: {e}]"}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        answer = await llm.ollama_chat(rag_messages, model=req.model, options=options)
    except Exception as e:
        logger.exception(f"/v1/chat/completions: Ollama request failed (model={req.model})")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"Ollama request failed: {e}",
                    "type": "upstream_error",
                    "param": None,
                    "code": "ollama_error",
                }
            },
        )

    if sources_footer:
        answer += sources_footer

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
