import hashlib
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from config import settings
from embeddings import embed_texts
import vector_store
import graph_store
import graph_extraction
import file_parsers
import jobs

logger = logging.getLogger("graphrag")


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # try to break on a sentence/paragraph boundary near `end`
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1:
            boundary = text.rfind(". ", start, end)
        if boundary != -1 and boundary > start + chunk_size * 0.5:
            end = boundary + 1

        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return [c for c in chunks if c]


def chunk_blocks(blocks: List[Dict[str, Any]], chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
    """
    Chunks each block independently (rather than concatenating every block
    into one string first) so a chunk never spans two unrelated blocks -
    e.g. never spans two PDF pages or two docx sections. Each resulting
    chunk carries its source block's metadata (page number, section
    heading, sheet name, etc.) forward into the chunk payload.
    """
    all_chunks: List[Dict[str, Any]] = []
    for block in blocks:
        text = (block.get("text") or "").strip()
        if not text:
            continue
        metadata = block.get("metadata") or {}
        for piece in chunk_text(text, chunk_size, overlap):
            all_chunks.append({"text": piece, "metadata": metadata})
    return all_chunks


def stable_chunk_id(doc_id: str, index: int) -> str:
    raw = f"{doc_id}:{index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _content_hash(blocks: List[Dict[str, Any]]) -> str:
    joined = "\n".join(b.get("text") or "" for b in blocks)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _extract_batch_concurrently(batch: List[Dict[str, Any]], doc_id: str, batch_start: int) -> List[Dict[str, Any]]:
    """
    Resolves entity/relationship extraction for every chunk in a batch,
    via one of two paths per chunk:

    - If the chunk carries a precomputed structural_extraction result in
      its metadata (JSON blocks - see file_parsers._json_block_metadata
      and structural_extraction.py) AND that result's `coverage` meets
      JSON_STRUCTURAL_MIN_COVERAGE, it's used directly with no Ollama call
      at all: JSON's own key names and nesting already encode most of
      what the LLM would otherwise have to infer from prose. Low coverage
      means most of the record's fields didn't match any known key hint -
      a strong signal this schema uses field-naming conventions
      structural_extraction.py hasn't been taught, not that the record
      genuinely has nothing worth extracting - so that case falls through
      to the LLM path instead of silently keeping a thin result.
    - Otherwise, falls back to the LLM-based graph_extraction.extract_graph,
      run concurrently across the batch (bounded by
      GRAPH_EXTRACTION_CONCURRENCY) as before - this is the dominant
      per-chunk cost during ingestion (one Ollama round-trip per chunk),
      so concurrency here is the main lever for large-file ingestion
      throughput. Real-world speedup depends on OLLAMA_NUM_PARALLEL
      actually being raised above 1 on the Ollama side, otherwise
      requests still queue there even if issued concurrently here.

    Only chunks that actually need an LLM call get submitted to the
    thread pool - high-coverage structural-extraction chunks resolve
    immediately with no round trip at all.
    """
    results: List[Optional[Dict[str, Any]]] = [None] * len(batch)
    to_extract: Dict[int, str] = {}

    for i, chunk in enumerate(batch):
        structural = (chunk.get("metadata") or {}).get("structural_extraction")
        if structural is not None and structural.get("coverage", 0.0) >= settings.JSON_STRUCTURAL_MIN_COVERAGE:
            results[i] = structural
        else:
            to_extract[i] = chunk["text"]

    if to_extract:
        max_workers = max(1, settings.GRAPH_EXTRACTION_CONCURRENCY)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(graph_extraction.extract_graph, text): i
                for i, text in to_extract.items()
            }
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    results[i] = future.result()
                except Exception:
                    logger.exception(
                        f"Graph extraction failed for chunk {batch_start + i} of {doc_id}, continuing without it"
                    )
                    results[i] = {"entities": [], "relationships": []}
    return results


def ingest_document(
    doc_id: str,
    title: str,
    text: Optional[str] = None,
    blocks: Optional[List[Dict[str, Any]]] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Chunks + embeds + writes a document to Qdrant (vectors + chunk text) and
    Neo4j (entities + relationships extracted from each chunk by the LLM).
    Neo4j never stores chunk text - Entity nodes carry a chunk_ids list that
    points back to Qdrant, which stays the single source of truth for text.

    Accepts either raw `text` (wrapped as a single block, e.g. the /ingest
    raw-text API) or pre-structured `blocks` from file_parsers.extract_text
    (each carrying page/section/sheet metadata). Exactly one should be
    provided; if `blocks` is omitted, `text` is used as a single block with
    no metadata.

    Processes chunks in bounded batches (EMBED_BATCH_SIZE) rather than
    embedding/upserting the entire document in one shot — for large files
    (tens of thousands of chunks), a single giant embedding request can
    take far longer than any reasonable HTTP timeout and gives no visibility
    into progress. Batching keeps each individual Ollama/Qdrant call small
    and fast, and lets us report real progress via the job record.
    """
    if blocks is None:
        blocks = [{"text": text or "", "metadata": {}}]

    content_hash = _content_hash(blocks)
    duplicate_of = None
    try:
        duplicate_of = graph_store.find_document_by_hash(content_hash)
        if duplicate_of and duplicate_of != doc_id:
            logger.warning(
                f"ingest_document: content of '{doc_id}' matches the already-ingested "
                f"document '{duplicate_of}' (identical content hash) - ingesting anyway, "
                f"but this is very likely a duplicate."
            )
        else:
            duplicate_of = None
    except Exception:
        logger.warning("ingest_document: duplicate-content check failed, continuing without it", exc_info=True)

    vector_store.ensure_collection()
    graph_store.ensure_constraints()
    graph_store.upsert_document(doc_id=doc_id, title=title, content_hash=content_hash)

    chunks = chunk_blocks(blocks)
    total = len(chunks)
    if job_id:
        jobs.update_job(job_id, total_chunks=total, chunks_processed=0, duplicate_of=duplicate_of)

    if total == 0:
        return {"doc_id": doc_id, "chunks_ingested": 0, "duplicate_of": duplicate_of}

    batch_size = max(1, settings.EMBED_BATCH_SIZE)
    chunks_written = 0
    cancelled = False

    for start in range(0, total, batch_size):
        # Checked at the top of the loop, before doing any work for this
        # batch, so a cancellation request stops ingestion as promptly as
        # possible - the batch already in flight when the request lands
        # will still finish (an Ollama call already underway can't be
        # aborted mid-request), but no *new* batch starts after this.
        if job_id and jobs.is_cancel_requested(job_id):
            cancelled = True
            logger.info(f"ingest {doc_id}: cancellation requested, stopping after {chunks_written}/{total} chunks")
            break

        batch = chunks[start : start + batch_size]
        batch_texts = [c["text"] for c in batch]

        t0 = time.perf_counter()
        vectors = embed_texts(batch_texts, prefix=settings.EMBEDDING_DOC_PREFIX)
        t_embed = time.perf_counter() - t0

        t1 = time.perf_counter()
        extractions = _extract_batch_concurrently(batch, doc_id, start)
        t_extract = time.perf_counter() - t1

        points = []
        graph_items = []
        for offset, (chunk, vector, extraction) in enumerate(zip(batch, vectors, extractions)):
            idx = start + offset
            chunk_id = stable_chunk_id(doc_id, idx)

            payload = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "index": idx,
                "text": chunk["text"],
            }
            # Carries page/section/sheet/etc metadata from file_parsers
            # through into the Qdrant payload, for citation purposes.
            payload.update(chunk.get("metadata") or {})
            points.append({"id": chunk_id, "vector": vector, "payload": payload})

            graph_items.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "entities": extraction["entities"],
                    "relationships": extraction["relationships"],
                }
            )

        # One Neo4j session/transaction for the whole batch (see
        # graph_store.upsert_batch). Wrapped in try/except deliberately -
        # previously a Neo4j hiccup here would raise straight out of this
        # loop and abort the entire ingest job, even though the batch's
        # embeddings were already computed and the vector upsert below
        # would otherwise succeed fine. Matches the fail-soft philosophy
        # used everywhere else in this codebase (retrieval channels,
        # extraction calls): vector search should keep working even if
        # graph data for this batch is temporarily unavailable.
        t_graph_start = time.perf_counter()
        try:
            graph_store.upsert_batch(graph_items)
        except Exception:
            logger.exception(
                f"Graph upsert failed for batch {start}-{start + len(batch) - 1} of {doc_id}, "
                f"continuing without it (vector chunks for this batch are still written)"
            )
        t_graph = time.perf_counter() - t_graph_start

        t2 = time.perf_counter()
        vector_store.upsert_chunks(points)
        t_upsert = time.perf_counter() - t2

        logger.info(
            f"ingest {doc_id}: batch {start}-{start + len(batch) - 1}/{total} "
            f"embed={t_embed:.2f}s extract={t_extract:.2f}s graph={t_graph:.2f}s upsert={t_upsert:.2f}s"
        )

        processed = min(start + batch_size, total)
        chunks_written = processed
        if job_id:
            batch_num = start // batch_size
            is_last_batch = processed >= total
            # Progress updates always land in memory (so GET /jobs/{id}
            # stays live), but only every 5th batch (or the final one)
            # gets written to disk - a large file can mean hundreds of
            # these calls, and jobs.py's persist writes the *entire* job
            # history file each time, not just this job. A crash between
            # persisted writes can only cost some progress-bar precision,
            # never a status transition (those always persist immediately
            # in process_uploaded_file below).
            jobs.update_job(
                job_id,
                chunks_processed=processed,
                persist=is_last_batch or batch_num % 5 == 0,
            )

    return {
        "doc_id": doc_id,
        "chunks_ingested": chunks_written,
        "duplicate_of": duplicate_of,
        "cancelled": cancelled,
    }


def process_uploaded_file(job_id: str, path: str, filename: str, doc_id: str, title: str) -> None:
    """
    Runs in a background task after a file has been saved to disk. Extracts
    blocks based on file type, ingests them, and updates job status
    throughout. The saved file is deleted once processing finishes
    (success or failure).
    """
    try:
        jobs.update_job(job_id, status="processing")
        blocks = file_parsers.extract_text(path, filename)
        if not any((b.get("text") or "").strip() for b in blocks):
            raise ValueError("No extractable text found in file")

        result = ingest_document(doc_id=doc_id, title=title, blocks=blocks, job_id=job_id)
        final_status = "cancelled" if result.get("cancelled") else "done"
        jobs.update_job(
            job_id,
            status=final_status,
            chunks_ingested=result["chunks_ingested"],
            duplicate_of=result.get("duplicate_of"),
        )
    except Exception as e:  # noqa: BLE001 - surface any failure to the job record
        logger.exception(f"Ingestion failed for job {job_id} (doc_id={doc_id}, file={filename})")
        jobs.update_job(job_id, status="error", error=str(e) or repr(e))
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
