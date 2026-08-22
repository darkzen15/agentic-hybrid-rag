from typing import List, Dict, Any
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import settings
from embeddings import vector_size

logger = logging.getLogger("graphrag")

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    return _client


_collection_ensured = False


def ensure_collection():
    """
    Called at the start of every ingest_document() call - once per
    document, so someone uploading many files pays this cost once per
    file. Everything inside is already idempotent (create-if-missing,
    index-if-missing), but each check still means a network round trip to
    Qdrant even when nothing needs to change. _collection_ensured caches
    "already done this in this process" so only the first call after
    startup actually talks to Qdrant - later calls no-op immediately.
    Not lock-protected: a narrow race at startup could let two concurrent
    ingests both run the full (idempotent) check once, which is harmless
    given every operation inside is already safe to repeat - not worth a
    lock for that.
    """
    global _collection_ensured
    if _collection_ensured:
        return
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=vector_size(),
                distance=qmodels.Distance.COSINE,
            ),
        )
    _ensure_text_index(client)
    _collection_ensured = True


def _ensure_text_index(client: QdrantClient):
    """
    Full-text payload index on the chunk `text` field, needed for the
    keyword/lexical retrieval channel (keyword_search below) to use
    Qdrant's MatchText filter. Checked-then-created rather than blindly
    re-issued on every call (ensure_collection runs on every ingest) to
    avoid hammering Qdrant with a redundant index-create call per chunk
    batch. Any failure here is logged and swallowed rather than raised -
    this is a retrieval-quality optimization, not something ingestion
    should hard-fail on; the keyword channel just degrades to "no results"
    until the index exists.
    """
    try:
        info = client.get_collection(settings.QDRANT_COLLECTION)
        existing_schema = info.payload_schema or {}
        if "text" in existing_schema:
            return
        client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION,
            field_name="text",
            field_schema=qmodels.TextIndexParams(
                type=qmodels.TextIndexType.TEXT,
                tokenizer=qmodels.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True,
            ),
        )
    except Exception as e:
        logger.warning(
            f"vector_store: could not ensure text index on 'text' field ({e}); "
            f"the keyword search retrieval channel will return no results "
            f"until this index exists."
        )


def upsert_chunks(points: List[Dict[str, Any]]):
    """
    points: list of dicts with keys: id (str/int), vector (List[float]), payload (dict)
    """
    client = get_client()
    client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=[
            qmodels.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
            for p in points
        ],
    )


def search(
    vector: List[float], top_k: int = 5, doc_id: str = None, score_threshold: float = None
) -> List[Dict[str, Any]]:
    client = get_client()
    query_filter = None
    if doc_id:
        query_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
        )

    results = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        score_threshold=score_threshold,  # None = no filtering (Qdrant default)
    )
    return [
        {
            "chunk_id": r.payload.get("chunk_id"),
            "text": r.payload.get("text"),
            "doc_id": r.payload.get("doc_id"),
            "index": r.payload.get("index"),
            "score": r.score,
            "source": "vector",
        }
        for r in results
    ]


def keyword_search(keywords: List[str], limit: int = 20, doc_id: str = None) -> List[Dict[str, Any]]:
    """
    Lexical retrieval channel: finds chunks whose text contains any of the
    given keywords, via Qdrant's MatchText filter (requires the payload
    text index from _ensure_text_index). Exists because embeddings are
    weak at exact strings - IDs, codes, acronyms, quoted phrases - where
    the precise characters matter more than the meaning.

    MatchText is a boolean filter, not a scored search - Qdrant returns
    matches in storage order, not relevance order. Scrolling with
    limit=<final answer size> and ranking only what comes back would
    silently drop the true best matches whenever more than `limit` chunks
    match the filter, since scroll's cutoff happens before any ranking.
    Instead this pulls a much wider candidate pool
    (settings.RAG_KEYWORD_SCROLL_LIMIT) via scroll, ranks all of it
    client-side by keyword overlap, then truncates to `limit` afterward.
    Still bounded (not a full collection scan) - if a corpus regularly has
    more than RAG_KEYWORD_SCROLL_LIMIT chunks matching a single query's
    keywords, this should be replaced with a real BM25/full-text search
    engine rather than raising the constant further.
    """
    if not keywords:
        return []
    client = get_client()

    should_conditions = [
        qmodels.FieldCondition(key="text", match=qmodels.MatchText(text=kw))
        for kw in keywords
    ]
    must_conditions = []
    if doc_id:
        must_conditions.append(
            qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))
        )

    query_filter = qmodels.Filter(should=should_conditions, must=must_conditions or None)

    scroll_pool_size = max(settings.RAG_KEYWORD_SCROLL_LIMIT, limit)
    try:
        points, _ = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=query_filter,
            limit=scroll_pool_size,
            with_payload=True,
        )
    except Exception as e:
        logger.warning(f"vector_store.keyword_search: query failed, returning no keyword results: {e}")
        return []

    lowered_keywords = [kw.lower() for kw in keywords]
    scored = []
    for p in points:
        text = (p.payload.get("text") or "").lower()
        overlap = sum(1 for kw in lowered_keywords if kw in text)
        if overlap == 0:
            continue
        scored.append((overlap, p))
    scored.sort(key=lambda t: t[0], reverse=True)
    scored = scored[:limit]

    return [
        {
            "chunk_id": p.payload.get("chunk_id"),
            "text": p.payload.get("text"),
            "doc_id": p.payload.get("doc_id"),
            "index": p.payload.get("index"),
            "score": float(overlap),
            "source": "keyword",
        }
        for overlap, p in scored
    ]


def get_neighbors(doc_id: str, index: int, window: int = 1) -> List[Dict[str, Any]]:
    """
    Fetches the chunks immediately before/after a given chunk (same
    document, index within [index-window, index+window]) - used by
    retrieval's neighbor-window expansion to pull in context that a chunk
    boundary may have cut off from a strong hit. Filters on the payload's
    doc_id + index fields directly rather than recomputing chunk_id from
    the ingestion-time uuid5 scheme, so this stays correct even if that
    scheme ever changes.
    """
    if window <= 0 or index is None:
        return []
    client = get_client()
    query_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id)),
            qmodels.FieldCondition(key="index", range=qmodels.Range(gte=index - window, lte=index + window)),
        ]
    )
    try:
        points, _ = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=query_filter,
            limit=window * 2 + 5,
            with_payload=True,
        )
    except Exception as e:
        logger.warning(f"vector_store.get_neighbors: query failed, returning no neighbors: {e}")
        return []

    results = [
        {
            "chunk_id": p.payload.get("chunk_id"),
            "text": p.payload.get("text"),
            "doc_id": p.payload.get("doc_id"),
            "index": p.payload.get("index"),
        }
        for p in points
    ]
    results.sort(key=lambda r: r.get("index") or 0)
    return results


def get_by_ids(chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetches payloads (text, doc_id, ...) for specific chunk_ids directly,
    rather than by similarity. Used by graph expansion: Neo4j Entity nodes
    only store *which* chunk_ids they appear in, not the chunk text itself,
    so this is how retrieval gets the actual text back for a graph hit.
    Returns a dict keyed by chunk_id; ids no longer present in the
    collection (e.g. a re-ingested/deleted doc) are simply absent.
    """
    if not chunk_ids:
        return {}
    client = get_client()
    points = client.retrieve(
        collection_name=settings.QDRANT_COLLECTION,
        ids=chunk_ids,
        with_payload=True,
    )
    return {p.payload.get("chunk_id"): p.payload for p in points}


def get_all_chunk_ids_for_doc(doc_id: str) -> List[str]:
    """
    Fully paginates through every chunk belonging to a document. Used by
    document deletion, which needs the complete chunk_id set to know what
    to strip from the graph - a single scroll page isn't enough since a
    document can have far more chunks than one page holds.
    """
    client = get_client()
    query_filter = qmodels.Filter(
        must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))]
    )
    chunk_ids: List[str] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=query_filter,
            limit=500,
            with_payload=["chunk_id"],
            offset=offset,
        )
        chunk_ids.extend(p.payload.get("chunk_id") for p in points if p.payload.get("chunk_id"))
        if offset is None:
            break
    return chunk_ids


def delete_by_doc_id(doc_id: str) -> List[str]:
    """
    Deletes every chunk belonging to doc_id from Qdrant. Returns the
    chunk_ids that were deleted, so the caller (main.py) can pass them to
    graph_store.delete_document to clean up the corresponding graph data -
    Neo4j has no independent record of which chunk_ids belonged to which
    document.
    """
    chunk_ids = get_all_chunk_ids_for_doc(doc_id)
    if not chunk_ids:
        return []
    client = get_client()
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=qmodels.PointIdsList(points=chunk_ids),
    )
    return chunk_ids
